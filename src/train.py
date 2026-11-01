"""Local training of a Qwen model with transformers and PEFT, replacing Plunkett's OpenAI calls.

Reads: model_hyperparameters (model_name, quantization, finetune, lora_rank,
lora_alpha, learning_rate, batch_size, loss_mask_answer_only, training_steps,
checkpoint_every, introspection_steps, introspection_batch_size, seed);
compute (device, mixed_precision). Loss is masked to the answer tokens.
Checkpoints are saved every `checkpoint_every` steps and at the last step, and
accuracy and loss per step go to results/train_log.csv. A decision checkpoint also
holds the optimizer state and the training state (optimizer.pt, training_state.pt),
so a run can be resumed from it exactly; see train_stage's start_step and resume.
"""
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import eval_batch_size, learning_rate_at

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRAIN_LOG_COLUMNS = [
    "run_id",
    "stage",
    "fold",
    "introspection_from_step",
    "step",
    "loss",
    "first_token_accuracy",
    "answer_token_accuracy",
    "val_loss",
    "val_first_token_accuracy",
    "val_answer_token_accuracy",
    "learning_rate",
    "elapsed_seconds",
    "resumed_from_step",
    "introspection_target",
]

OPTIMIZER_STATE = "optimizer.pt"            # AdamW's moments, saved beside a decision checkpoint's adapter
TRAINING_STATE = "training_state.pt"        # step, data order and random generators, for an exact resume


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

    # Gradient checkpointing recomputes activations in the backward pass instead
    # of storing them, roughly a quarter of the activation memory for about a
    # third more time per step. Off by default so Plunkett's settings are
    # unchanged; the *_20gb configs turn it on, since a 20 GB MIG slice cannot
    # hold a 4B model's activations at batch 10 otherwise. Needs use_cache off,
    # which load_model_and_tokenizer already sets.
    checkpointing = mh["gradient_checkpointing"]
    if mh["quantization"] != "none":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing)
    elif checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if checkpointing:
        print("gradient checkpointing: on")
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


def _accuracy_counts(logits, labels, first_token=True):
    """Per-example first-answer-token hits, plus correct and valid answer-token counts.

    Takes the logits in whatever dtype the model produced. Only an argmax is
    needed, so no fp32 copy of the (batch, sequence, vocabulary) tensor is made.
    Counts rather than means, so micro-batches of one step can be combined.
    `first_token` False skips the first-token hits and returns none.
    """
    pred = logits[:, :-1, :].argmax(-1)
    target = labels[:, 1:]
    valid = target != -100
    correct = (pred == target) & valid
    firsts = []
    for i in range(labels.shape[0] if first_token else 0):
        positions = torch.nonzero(valid[i]).flatten()
        if len(positions):
            firsts.append(correct[i, positions[0]].item())
    return firsts, correct.sum().item(), valid.sum().item()




def _autocast(device, precision):
    if device == "cuda" and precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    if device == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def _decoder_and_head(model):
    """The transformer body and the output layer, under any PEFT wrapper; (None, None) when it has neither."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    # the body by its declared name ("model" for Qwen3), not get_decoder(): in transformers 4.56 that
    # recurses one level too far and returns None for Qwen3ForCausalLM
    prefix = getattr(base, "base_model_prefix", None)
    decoder = getattr(base, prefix, None) if prefix else None
    head = base.get_output_embeddings() if hasattr(base, "get_output_embeddings") else None
    return (decoder, head) if decoder is not None and head is not None else (None, None)


@torch.no_grad()
def evaluate_examples(model, tok, encoded, batch_size, device, precision, first_token=True):
    """Mean loss over answer tokens, first-token accuracy and all-answer-token accuracy on held-out examples.

    `first_token` False leaves the first token unread and its accuracy NaN, for stage two.

    The output layer is applied at the answer positions only. A full forward pass makes logits
    for every prompt token over Qwen3's 152k-token vocabulary, then an fp32 copy for the loss:
    over a gigabyte for four long introspection prompts, which is what ran a 16 GB card out of
    memory. The loss and accuracies are the same numbers, since only answer positions were ever
    scored; what changes is that the rest are never made. A model without a separate body and
    output layer gets the full forward pass, read at the same positions.
    """
    model.eval()
    decoder, head = _decoder_and_head(model)
    total_loss, total_tokens, firsts, correct = 0.0, 0, [], 0
    for i in range(0, len(encoded), batch_size):
        input_ids, attention, labels = collate(encoded[i : i + batch_size], tok.pad_token_id, device)
        target = labels[:, 1:]
        valid = target != -100
        with _autocast(device, precision):
            if decoder is not None:
                hidden = decoder(input_ids=input_ids, attention_mask=attention).last_hidden_state
                logits = head(hidden[:, :-1][valid])    # (answer tokens, vocabulary), rows in batch order
                del hidden
            else:
                logits = model(input_ids=input_ids, attention_mask=attention, labels=labels).logits[:, :-1][valid]
        logits = logits.float()
        answers = target[valid]
        total_loss += F.cross_entropy(logits, answers, reduction="sum").item()
        total_tokens += answers.numel()
        hits = logits.argmax(-1) == answers
        correct += hits.sum().item()
        if first_token:
            start = 0
            for count in valid.sum(1).tolist():
                if count:
                    firsts.append(hits[start].item())
                start += count
        del logits
    model.train()
    first_acc = float(np.mean(firsts)) if firsts else float("nan")
    return total_loss / max(total_tokens, 1), first_acc, correct / max(total_tokens, 1)


def stage_label(stage, fold=0, from_step=None, target=None):
    """The name a stage's checkpoints and detail files go under.

    decision, introspection-fold1, or introspection-from-1500-fold1 when the folds branched
    from decision step 1500. The pipeline always names the step it branched from, so several
    introspection points in one run never share a folder. An introspection target other than
    hidden is named too, introspection-from-1500-recovered-fold1; hidden keeps the plain name it
    always had, so earlier runs' folders and files read as they were.
    """
    name = stage if from_step is None else f"{stage}-from-{from_step}"
    if target and target != "hidden":
        name = f"{name}-{target}"
    return name if not fold else f"{name}-fold{fold}"


def _append_log(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    if not new:
        with path.open(newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != TRAIN_LOG_COLUMNS:
            if not set(existing) <= set(TRAIN_LOG_COLUMNS):
                raise RuntimeError(f"{path} has columns {existing} but this pipeline writes {TRAIN_LOG_COLUMNS}; move the old file aside")
            # A code update added a column. Rewrite the old rows under the new header, as
            # append_results does for results.csv, so the appended rows line up.
            with path.open(newline="") as f:
                old = list(csv.DictReader(f))
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TRAIN_LOG_COLUMNS)
                writer.writeheader()
                writer.writerows(old)
            print(f"note: {path} migrated to {len(TRAIN_LOG_COLUMNS)} columns")
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
    extra_checkpoints=(),
    from_step=None,
    start_step=0,
    resume=None,
    target=None,
):
    """Train for `steps` optimizer steps, logging every step and checkpointing every `checkpoint_every`.

    Steps in `extra_checkpoints` are checkpointed too, so an introspection point need not fall on
    the regular schedule. `from_step` is the decision step an introspection stage branched from;
    it names the checkpoint folder and fills the log's introspection_from_step column.
    Examples are shuffled anew each epoch. Reads model_hyperparameters
    (learning_rate, loss_mask_answer_only, seed), compute (mixed_precision, and the
    decision or introspection eval batch size for the validation pass).
    Returns the list of (step, checkpoint_dir).

    `start_step` resumes: training continues at start_step + 1, with the adapter already loaded by
    the caller. `resume` is load_training_state's dict for that checkpoint. Its training state puts
    the data order back exactly; without one, the order is replayed from the seed, which gives the
    same batches. Its optimizer state restores AdamW's moments; without one the optimizer starts
    fresh, the one way such a resume differs from a run that never stopped, and it says so.
    Decision checkpoints save both files, so a resume from any of them is exact.

    The learning rate is set before every step from model_hyperparameters.lr_schedule
    (config.learning_rate_at), a function of the step alone, so a resume needs no scheduler state.
    Introspection stages keep the constant rate. `target` is an introspection stage's target (hidden,
    recovered, shuffled, random): it names the checkpoint folder and fills the log's column.
    """
    mh, comp = cfg["model_hyperparameters"], cfg["compute"]
    precision = comp["mixed_precision"] if device == "cuda" else "no"
    mask = mh["loss_mask_answer_only"]
    # Micro-batching: the optimizer still steps once per `batch_size` examples,
    # so the hyperparameter is unchanged, but the examples pass through the
    # model `micro` at a time and gradients accumulate. What it buys is memory:
    # cross entropy over Qwen3's 152k-token vocabulary needs several full fp32
    # copies of the logits, and those scale with the micro-batch, not the batch.
    micro = min(mh["micro_batch_size"] or batch_size, batch_size)
    encoded = [encode_example(tok, e, mask) for e in examples]
    encoded_val = [encode_example(tok, e, mask) for e in val_examples]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=mh["learning_rate"])
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    rng = random.Random(mh["seed"] + (0 if stage == "decision" else 100 * (fold + 1)))
    order = []
    # stage two's validation prompts are whole report questions with JSON answers, far longer than an A/B trial
    val_batch = eval_batch_size(cfg, "decision" if stage == "decision" else "introspection")
    resumed = cfg.get("_resumed_from_step", "")
    if start_step:
        state = (resume or {}).get("training_state")
        if state is not None:
            if state["step"] != start_step:
                raise ValueError(f"training state is for step {state['step']}, not {start_step}")
            rng.setstate(state["order_rng"])
            order = list(state["order"])
        else:
            for _ in range(start_step):      # replay the draws, so step start_step + 1 sees its own batch
                if len(order) < batch_size:
                    fresh = list(range(len(encoded)))
                    rng.shuffle(fresh)
                    order += fresh
                order = order[batch_size:]
        if (resume or {}).get("optimizer") is not None:
            optimizer.load_state_dict(resume["optimizer"])
            if scaler is not None and state is not None and state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            print(f"resumed at step {start_step}: optimizer and data order restored")
        else:
            print(f"resumed at step {start_step}: data order {'restored' if state is not None else 'replayed from the seed'}; "
                  "no optimizer state was saved with this checkpoint, so AdamW starts fresh")
    checkpoints = []
    stage_dir = Path(checkpoint_root) / run_id / stage_label(stage, fold, from_step, target)
    extra = set(extra_checkpoints)
    model.train()
    t0 = time.time()
    rows = []
    # Console lines report the mean over the steps since the previous line, so the
    # accuracy covers hundreds of examples rather than one batch of ten. The
    # per-step values still go to train_log.csv unchanged.
    # Stage one's answer is one letter, so its first token is the answer and its accuracy is the
    # A/B accuracy. Stage two's answer is a whole JSON object whose first token is always "{",
    # so stage two never reads the first token: it is measured over every answer token instead.
    first_token = stage == "decision"
    win_loss, win_firsts, win_correct, win_valid, win_start = 0.0, [], 0, 0, start_step + 1

    def window_text(step):
        nonlocal win_loss, win_firsts, win_correct, win_valid, win_start
        n = step - win_start + 1
        if first_token:
            acc = f"acc {float(np.mean(win_firsts)) if win_firsts else float('nan'):.3f}"
        else:
            acc = f"answer acc {win_correct / max(win_valid, 1):.3f}"
        text = f"loss {win_loss / n:.4f} {acc} [mean of steps {win_start}-{step}]"
        win_loss, win_firsts, win_correct, win_valid, win_start = 0.0, [], 0, 0, step + 1
        return text

    target_text = "" if stage == "decision" else ", target " + (target or "hidden")
    print(
        f"training stage '{stage}'{'' if not fold else f' fold {fold}'}"
        f"{'' if from_step is None else f' from decision step {from_step}'}{target_text}: {len(encoded)} examples, {steps} steps, "
        f"batch {batch_size}" + (f" in micro-batches of {micro}" if micro < batch_size else "")
        + (f", resuming after step {start_step}" if start_step else "")
    )
    for step in range(start_step + 1, steps + 1):
        if len(order) < batch_size:
            fresh = list(range(len(encoded)))
            rng.shuffle(fresh)
            order += fresh
        batch = [encoded[i] for i in order[:batch_size]]
        order = order[batch_size:]
        chunks = [batch[i : i + micro] for i in range(0, len(batch), micro)]
        loss_value, firsts, answer_correct, answer_valid = 0.0, [], 0, 0
        for chunk in chunks:
            input_ids, attention, labels = collate(chunk, tok.pad_token_id, device)
            with _autocast(device, precision):
                out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            # Each chunk's loss is a mean over its answer tokens; dividing by the
            # chunk count makes the accumulated gradient the mean over the step.
            # Exact when chunks hold equal token counts, which decision examples
            # always do (one letter plus the end token each).
            loss = out.loss / len(chunks)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_value += loss.item()
            f, c, v = _accuracy_counts(out.logits.detach(), labels, first_token)
            firsts += f
            answer_correct += c
            answer_valid += v
            # Drop each chunk's outputs at once: the logits are the largest tensor
            # in the step, and the checkpoint evaluation below is the memory peak.
            del out, loss
        rate = learning_rate_at(cfg, step, stage)
        for group in optimizer.param_groups:
            group["lr"] = rate
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        first_acc = float(np.mean(firsts)) if firsts else float("nan")
        answer_acc = answer_correct / max(answer_valid, 1)
        win_loss += loss_value
        win_firsts += firsts
        win_correct += answer_correct
        win_valid += answer_valid
        row = {
            "run_id": run_id,
            "stage": stage,
            "fold": fold,
            "introspection_from_step": "" if from_step is None else from_step,
            "step": step,
            "loss": loss_value,
            "first_token_accuracy": first_acc if first_token else "",
            "answer_token_accuracy": answer_acc,
            "val_loss": "",
            "val_first_token_accuracy": "",
            "val_answer_token_accuracy": "",
            "learning_rate": rate,
            "elapsed_seconds": round(time.time() - t0, 2),
            "resumed_from_step": resumed,
            "introspection_target": "" if stage == "decision" else (target or "hidden"),
        }
        if step % checkpoint_every == 0 or step == steps or step in extra:
            if device == "cuda":
                torch.cuda.empty_cache()     # hand back the training step's cached blocks before the validation pass
            val_loss, val_first, val_answer = evaluate_examples(
                model, tok, encoded_val, val_batch, device, precision, first_token)
            row["val_loss"], row["val_answer_token_accuracy"] = val_loss, val_answer
            row["val_first_token_accuracy"] = val_first if first_token else ""
            ckpt = stage_dir / f"step-{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt))
            if stage == "decision":
                save_training_state(ckpt, optimizer, scaler, step, rng, order)
            checkpoints.append((step, ckpt))
            print(
                f"  step {step}/{steps} {window_text(step)} | val loss {val_loss:.4f} "
                f"val {'acc' if first_token else 'answer acc'} {val_first if first_token else val_answer:.3f} "
                f"| saved {ckpt} | {time.time() - t0:.0f}s"
            )
            rows.append(row)
            _append_log(log_path, rows)
            rows = []
            if on_checkpoint is not None:
                on_checkpoint(step, ckpt)
                model.train()
            if stage == "decision":
                mark_evaluated(ckpt)
        else:
            rows.append(row)
            if step % 50 == 0:
                print(f"  step {step}/{steps} {window_text(step)} | {time.time() - t0:.0f}s")
    if rows:
        _append_log(log_path, rows)
    return checkpoints


def save_training_state(ckpt, optimizer, scaler, step, order_rng, order):
    """optimizer.pt and training_state.pt beside a checkpoint's adapter, so a resume from it is exact.

    Saved before the checkpoint is evaluated, so a crash during the evaluation still leaves a
    checkpoint to resume from. The random generators are saved again once the evaluation is done
    (mark_evaluated), since the run after it continues from their state then.
    """
    ckpt = Path(ckpt)
    torch.save(optimizer.state_dict(), ckpt / OPTIMIZER_STATE)
    torch.save({
        "step": step,
        "order_rng": order_rng.getstate(),
        "order": list(order),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": rng_state(),
        "evaluated": False,
    }, ckpt / TRAINING_STATE)


def mark_evaluated(ckpt):
    """Record that the checkpoint's evaluation finished, with the random generators as it left them."""
    path = Path(ckpt) / TRAINING_STATE
    if path.exists():
        state = torch.load(path, weights_only=False)
        state["rng"], state["evaluated"] = rng_state(), True
        torch.save(state, path)


def load_training_state(ckpt, device):
    """A checkpoint's saved optimizer and training state, each None where the checkpoint has none.

    Checkpoints written before these files existed hold the adapter only.
    """
    ckpt = Path(ckpt)
    optimizer = (torch.load(ckpt / OPTIMIZER_STATE, map_location=device, weights_only=False)
                 if (ckpt / OPTIMIZER_STATE).exists() else None)
    state = torch.load(ckpt / TRAINING_STATE, weights_only=False) if (ckpt / TRAINING_STATE).exists() else None
    return {"optimizer": optimizer, "training_state": state}


def load_checkpoint_weights(model, ckpt, cfg, device):
    """Put a saved checkpoint's trained weights into a model built by apply_finetune."""
    ckpt = Path(ckpt)
    if cfg["model_hyperparameters"]["finetune"] == "lora":
        from peft import set_peft_model_state_dict
        from peft.utils import load_peft_weights

        weights = load_peft_weights(str(ckpt), device=device)
        result = set_peft_model_state_dict(model, weights)
        missing = [k for k in getattr(result, "missing_keys", []) if "lora_" in k]
        unexpected = list(getattr(result, "unexpected_keys", []))
        if missing or unexpected:
            raise RuntimeError(f"{ckpt} does not fit this model: missing {missing[:5]}, unexpected {unexpected[:5]}")
        print(f"loaded adapter {ckpt} ({len(weights)} tensors)")
    else:
        from safetensors.torch import load_file

        files = sorted(ckpt.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no weights in {ckpt}")
        state = {}
        for f in files:
            state.update(load_file(str(f), device=device))
        model.load_state_dict(state, strict=True)
        print(f"loaded full weights {ckpt}")
    return model


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


def rng_state():
    """Every random number generator training draws from, so a detour can be undone exactly."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    """Put back what `rng_state` saved."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
