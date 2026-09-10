"""Local training of a Qwen model with transformers and PEFT, replacing Plunkett's OpenAI calls.

Reads: model_hyperparameters (model_name, quantization, finetune, lora_rank,
lora_alpha, learning_rate, batch_size, loss_mask_answer_only, training_steps,
checkpoint_every, introspection_steps, introspection_batch_size, seed);
compute (device, mixed_precision). Loss is masked to the answer tokens.
Checkpoints are saved every `checkpoint_every` steps and at the last step, and
accuracy and loss per step go to results/train_log.csv.
"""
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRAIN_LOG_COLUMNS = [
    "run_id",
    "stage",
    "fold",
    "step",
    "loss",
    "first_token_accuracy",
    "answer_token_accuracy",
    "val_loss",
    "val_first_token_accuracy",
    "learning_rate",
    "elapsed_seconds",
]


def _from_pretrained(name, **kwargs):
    """AutoModelForCausalLM.from_pretrained across the dtype / torch_dtype rename."""
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except TypeError as err:
        if "dtype" in kwargs and "torch_dtype" not in kwargs and "dtype" in str(err):
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            return AutoModelForCausalLM.from_pretrained(name, **kwargs)
        raise


def end_token_id(tok):
    """The token that ends an assistant turn: <|im_end|> for Qwen, else the tokenizer's EOS."""
    vocab = tok.get_vocab()
    return vocab["<|im_end|>"] if "<|im_end|>" in vocab else tok.eos_token_id


def gpu_dtype(cfg, device):
    """Weight and compute dtype on a GPU, following compute.mixed_precision.

    fp16 is the only half precision a T4 supports, so it stays the default.
    bf16 needs Ampere or newer and is the better choice on an A100 or H100: it
    carries fp32's exponent range, so training needs no loss scaling and cannot
    silently overflow. The gradient scaler below is built for fp16 only, which
    matches.
    """
    if device != "cuda":
        return torch.float32
    if cfg["compute"].get("mixed_precision") != "bf16":
        return torch.float16
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "compute.mixed_precision is 'bf16' but this GPU does not support it; "
            "use 'fp16' on Turing cards such as the T4"
        )
    return torch.bfloat16


def load_model_and_tokenizer(cfg, device):
    """Download the model named in the settings and quantize it if asked. Reads model_hyperparameters, compute."""
    from transformers import AutoTokenizer

    mh = cfg["model_hyperparameters"]
    name = mh["model_name"]
    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(name, token=token)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    half = gpu_dtype(cfg, device)
    dtype_name = str(half).replace("torch.", "")
    if mh["quantization"] != "none":
        if device != "cuda":
            raise RuntimeError(
                f"model_hyperparameters.quantization is '{mh['quantization']}' but compute.device is not 'cuda'; bitsandbytes needs a GPU"
            )
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=half,
            bnb_4bit_use_double_quant=True,
        )
        model = _from_pretrained(name, quantization_config=bnb, device_map={"": 0}, dtype=half, token=token)
    else:
        dtype = torch.float32 if (device == "cpu" or mh["finetune"] == "full") else half
        if device == "cuda":
            # Stream the weights straight onto the GPU. Loading on the CPU and then
            # moving needs the whole model in host RAM first, which a 14B model in
            # bf16 (about 28 GB) may not get on a shared box.
            model = _from_pretrained(name, dtype=dtype, device_map={"": 0}, token=token)
        else:
            model = _from_pretrained(name, dtype=dtype, token=token).to(device)
    model.config.use_cache = False
    print(f"model: {name} | quantization={mh['quantization']} | device={device} | dtype={dtype_name}")
    return model, tok


def apply_finetune(model, cfg):
    """LoRA at the given rank on every linear layer, or full fine-tuning. Reads model_hyperparameters."""
    mh = cfg["model_hyperparameters"]
    if mh["finetune"] == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        print(f"finetune: full, {sum(p.numel() for p in model.parameters()):,} trainable parameters")
        return model
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if mh["quantization"] != "none":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora = LoraConfig(
        r=mh["lora_rank"],
        lora_alpha=mh["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"finetune: lora rank {mh['lora_rank']} alpha {mh['lora_alpha']} on {LORA_TARGETS}; {trainable:,} trainable parameters")
    return model


def encode_example(tok, example, mask_answer_only=True):
    """Token ids and labels for one chat example: prompt via the chat template, answer + end token appended.

    Prompt and answer are tokenized separately and concatenated, so the label
    boundary is exact and matches what the model sees at inference. Reads
    model_hyperparameters.loss_mask_answer_only.
    """
    prompt_text = tok.apply_chat_template(
        example.messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tok(example.answer, add_special_tokens=False)["input_ids"] + [end_token_id(tok)]
    ids = prompt_ids + answer_ids
    labels = ([-100] * len(prompt_ids) if mask_answer_only else list(prompt_ids)) + answer_ids
    return ids, labels


def collate(encoded, pad_id, device):
    """Right-pad a list of (ids, labels) into tensors."""
    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(encoded), width), dtype=torch.long)
    labels = torch.full((len(encoded), width), -100, dtype=torch.long)
    for i, (ids, labs) in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attention[i, : len(ids)] = 1
        labels[i, : len(labs)] = torch.tensor(labs)
    return input_ids.to(device), attention.to(device), labels.to(device)


def _accuracies(logits, labels):
    """First-answer-token accuracy and all-answer-token accuracy from teacher-forced logits.

    Takes the logits in whatever dtype the model produced. Only an argmax is
    needed, so no fp32 copy of the (batch, sequence, vocabulary) tensor is made.
    """
    pred = logits[:, :-1, :].argmax(-1)
    target = labels[:, 1:]
    valid = target != -100
    correct = (pred == target) & valid
    answer_acc = correct.sum().item() / max(valid.sum().item(), 1)
    firsts = []
    for i in range(labels.shape[0]):
        positions = torch.nonzero(valid[i]).flatten()
        if len(positions):
            firsts.append(correct[i, positions[0]].item())
    first_acc = float(np.mean(firsts)) if firsts else float("nan")
    return first_acc, answer_acc


def _autocast(device, precision):
    if device == "cuda" and precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    if device == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def evaluate_examples(model, tok, encoded, batch_size, device, precision):
    """Mean loss over answer tokens and first-token accuracy on held-out examples."""
    model.eval()
    losses, weights, firsts = [], [], []
    for i in range(0, len(encoded), batch_size):
        input_ids, attention, labels = collate(encoded[i : i + batch_size], tok.pad_token_id, device)
        with _autocast(device, precision):
            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
        n_tokens = (labels[:, 1:] != -100).sum().item()
        losses.append(out.loss.item() * n_tokens)
        weights.append(n_tokens)
        first, _ = _accuracies(out.logits, labels)
        firsts.append(first)
    model.train()
    return sum(losses) / max(sum(weights), 1), float(np.nanmean(firsts)) if firsts else float("nan")


def _append_log(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_LOG_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def train_stage(
    model,
    tok,
    examples,
    val_examples,
    cfg,
    device,
    run_id,
    stage,
    fold,
    steps,
    batch_size,
    checkpoint_every,
    checkpoint_root="checkpoints",
    log_path="results/train_log.csv",
    on_checkpoint=None,
):
    """Train for `steps` optimizer steps, logging every step and checkpointing every `checkpoint_every`.

    Examples are shuffled anew each epoch. Reads model_hyperparameters
    (learning_rate, loss_mask_answer_only, seed), compute (mixed_precision).
    Returns the list of (step, checkpoint_dir).
    """
    mh, comp = cfg["model_hyperparameters"], cfg["compute"]
    precision = comp["mixed_precision"] if device == "cuda" else "no"
    mask = mh["loss_mask_answer_only"]
    encoded = [encode_example(tok, e, mask) for e in examples]
    encoded_val = [encode_example(tok, e, mask) for e in val_examples]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=mh["learning_rate"])
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    rng = random.Random(mh["seed"] + (0 if stage == "decision" else 100 * (fold + 1)))
    order = []
    checkpoints = []
    stage_dir = Path(checkpoint_root) / run_id / (stage if not fold else f"{stage}-fold{fold}")
    model.train()
    t0 = time.time()
    rows = []
    print(f"training stage '{stage}'{'' if not fold else f' fold {fold}'}: {len(encoded)} examples, {steps} steps, batch {batch_size}")
    for step in range(1, steps + 1):
        if len(order) < batch_size:
            fresh = list(range(len(encoded)))
            rng.shuffle(fresh)
            order += fresh
        batch = [encoded[i] for i in order[:batch_size]]
        order = order[batch_size:]
        input_ids, attention, labels = collate(batch, tok.pad_token_id, device)
        with _autocast(device, precision):
            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
        loss = out.loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        first_acc, answer_acc = _accuracies(out.logits.detach(), labels)
        loss_value = loss.item()
        # Drop the step's outputs before any evaluation below. The logits alone are
        # about 1 GB in bf16 for an 8B model at batch 10, and evaluation at batch 64
        # is the memory peak of the whole run.
        del out, loss
        row = {
            "run_id": run_id,
            "stage": stage,
            "fold": fold,
            "step": step,
            "loss": loss_value,
            "first_token_accuracy": first_acc,
            "answer_token_accuracy": answer_acc,
            "val_loss": "",
            "val_first_token_accuracy": "",
            "learning_rate": mh["learning_rate"],
            "elapsed_seconds": round(time.time() - t0, 2),
        }
        if step % checkpoint_every == 0 or step == steps:
            val_loss, val_first = evaluate_examples(model, tok, encoded_val, comp["eval_batch_size"], device, precision)
            row["val_loss"], row["val_first_token_accuracy"] = val_loss, val_first
            ckpt = stage_dir / f"step-{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt))
            checkpoints.append((step, ckpt))
            print(
                f"  step {step}/{steps} loss {loss_value:.4f} acc {first_acc:.3f} | val loss {val_loss:.4f} "
                f"val acc {val_first:.3f} | saved {ckpt} | {time.time() - t0:.0f}s"
            )
            rows.append(row)
            _append_log(log_path, rows)
            rows = []
            if on_checkpoint is not None:
                on_checkpoint(step, ckpt)
                model.train()
        else:
            rows.append(row)
            if step % 50 == 0:
                print(f"  step {step}/{steps} loss {loss_value:.4f} acc {first_acc:.3f} | {time.time() - t0:.0f}s")
    if rows:
        _append_log(log_path, rows)
    return checkpoints


def snapshot(model):
    """Copy the trainable state so a later stage can restart from it (PEFT adapter or full weights)."""
    try:
        from peft import get_peft_model_state_dict

        state = get_peft_model_state_dict(model)
    except (ImportError, AttributeError, ValueError):
        state = model.state_dict()
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def restore(model, state):
    """Undo training since `snapshot`."""
    try:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, state)
    except (ImportError, AttributeError, ValueError):
        model.load_state_dict(state, strict=False)
