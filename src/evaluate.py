"""Evaluation of a trained model or checkpoint: decisions, recovery, reports, faithfulness, chance.

Per checkpoint:
  1. decision accuracy on fresh verification trials (Plunkett's seed-5 trials), scored
     against the data folder's hidden latent, checked against gates.min_decision_accuracy;
  2. the same choices are fed to the estimator to recover the predicted learned latent;
  3. recovered versus hidden;
  4. reports in fresh contexts at report_temperature, samples_per_report per persona,
     each with its own freshly drawn option pair, parse rate against gates.min_parse_rate;
  5. faithfulness: report versus recovered latent, never versus the hidden latent;
  6. chance: step 5 with random latents from the rule;
  7. rows appended to results/results.csv, one per report schema, block and checkpoint.
Every distance is computed per block and the blocks are never combined.

Reads: model_estimating (decision_temperature, samples_per_trial,
estimation_trials_per_persona, chance_draws), report_schema (report_temperature,
samples_per_report, max_new_tokens), gates, object.attribute_count,
model_training.seeds (verification, reports), model_hyperparameters.seed,
compute.eval_batch_size.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import data as D
from src.chance import chance_level
from src.config import hyperparameter_columns
from src.estimators.base import Estimator, pooled_pearson
from src.train import end_token_id

METRIC_COLUMNS = [
    "report_schema",
    "stage",
    "introspection_fold",
    "checkpoint_step",
    "block",
    "n_personas",
    "decision_accuracy",
    "decision_gate",
    "recovered_vs_hidden",
    "recovered_vs_hidden_pearson",
    "faithfulness",
    "faithfulness_pearson",
    "chance",
    "chance_pearson",
    "parse_rate",
    "parse_gate",
    "hidden_vs_reported",
    "hidden_vs_reported_pearson",
    "n_personas_reported",
]


def results_columns(cfg):
    return ["run_id", "timestamp"] + list(hyperparameter_columns(cfg).keys()) + METRIC_COLUMNS


# --- model queries -----------------------------------------------------------------


def chat_prompt(tok, user_content):
    """Plunkett's two-message conversation, rendered with the model's chat template, thinking off."""
    messages = [{"role": "system", "content": D.SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _left_pad(tok, texts, device):
    tok.padding_side = "left"
    enc = tok(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids, attention = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    position_ids = (attention.cumsum(-1) - 1).clamp(min=0)
    return input_ids, attention, position_ids


def choice_token_ids(tok):
    ids = []
    for letter in ("A", "B"):
        encoded = tok.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"'{letter}' is not a single token for this tokenizer: {encoded}")
        ids.append(encoded[0])
    return ids


@torch.no_grad()
def choice_logits(model, tok, texts, batch_size, device):
    """Logits of the 'A' and 'B' tokens at the answer position, plus the unconstrained argmax token."""
    model.eval()
    ids = choice_token_ids(tok)
    logits_ab, argmax = [], []
    for i in range(0, len(texts), batch_size):
        input_ids, attention, position_ids = _left_pad(tok, texts[i : i + batch_size], device)
        logits = model(input_ids=input_ids, attention_mask=attention, position_ids=position_ids).logits[:, -1, :].float()
        logits_ab.append(logits[:, ids].cpu().numpy())
        argmax.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(logits_ab), np.concatenate(argmax), ids


@torch.no_grad()
def generate_texts(model, tok, texts, max_new_tokens, temperature, batch_size, device):
    """Greedy (temperature 0) or sampled completions in fresh contexts, stopping at the end-of-turn token.

    `use_cache=True` is passed explicitly and is not optional. The model is loaded
    with `config.use_cache = False`, which is right for training, and Qwen3 ships no
    value in its generation config, so generation would otherwise fall back to that
    False and rebuild attention over the whole sequence for every token. Report
    generation is the slowest part of an evaluation, so the fallback does real
    work for nothing.

    The size of that waste is not established. On CPU at 0.6B the run-to-run
    spread exceeded the effect: an alternating benchmark gave 1.16x, and a whole
    pipeline run was slower with the fix than without it. Both are noise at this
    scale, not evidence against the fix. The fix stays because generation
    without a cache cannot be faster than generation with one, and because the
    decode phase where a cache pays is a much larger share of the work on a GPU
    with 200-token replies than on a CPU with 60-token ones. Measure it there
    before quoting a number.
    """
    model.eval()
    end_ids = sorted({end_token_id(tok), tok.eos_token_id})
    outputs = []
    for i in range(0, len(texts), batch_size):
        input_ids, attention, _ = _left_pad(tok, texts[i : i + batch_size], device)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                      eos_token_id=end_ids, use_cache=True)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=1.0, top_k=0)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        generated = model.generate(input_ids=input_ids, attention_mask=attention, **kwargs)
        new_tokens = generated[:, input_ids.shape[1] :]
        outputs += [tok.decode(row, skip_special_tokens=True).strip() for row in new_tokens]
    return outputs


# --- collection --------------------------------------------------------------------


def collect(model, tok, cfg, comps, data, device, persona_indices, tag, out_dir):
    """Run the model on the verification trials and the report prompts for the given personas.

    Returns the per-persona detail (hidden, recovered, reported vectors by block)
    that `summarize` turns into results rows, and writes Plunkett-style detail
    files to `out_dir`.
    """
    rule, estimator, schemas = comps.rule, comps.estimator, comps.schemas
    me, rs, seeds = cfg["model_estimating"], cfg["report_schema"], cfg["model_training"]["seeds"]
    n = data.attribute_count
    batch_size = cfg["compute"]["eval_batch_size"]
    personas = data.personas
    rule_slices, est_slices = rule.block_slices(n), estimator.block_slices()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. decisions on fresh verification trials; all personas are drawn so the RNG stream matches Plunkett's
    trials = D.fresh_trials(personas, me["estimation_trials_per_persona"], seeds["verification"])
    texts, index = [], []
    for k in persona_indices:
        for t_i, trial in enumerate(trials[k]):
            trial.label = rule.label(data.latents[k], trial)
            texts.append(chat_prompt(tok, D.SELECTION_PROMPT_BASE + trial.generate_choice()))
            index.append((k, t_i))
    logits_ab, argmax, ids = choice_logits(model, tok, texts, batch_size, device)
    temperature, samples = me["decision_temperature"], me["samples_per_trial"]
    rng = np.random.default_rng(cfg["model_hyperparameters"]["seed"])
    decisions = []
    for (k, t_i), pair, top in zip(index, logits_ab, argmax):
        trial = trials[k][t_i]
        p_a = float(1 / (1 + np.exp(-(pair[0] - pair[1]))))
        if temperature > 0:
            p_a_t = float(1 / (1 + np.exp(-(pair[0] - pair[1]) / temperature)))
            choices = ["A" if rng.random() < p_a_t else "B" for _ in range(samples)]
        else:
            choices = ["A" if pair[0] > pair[1] else "B"] * samples
        for s, choice in enumerate(choices):
            row = {
                "model": tag,
                "scenario": trial.scenario.short_name,
                "selection": choice,
                **{f"A_attribute_{i+1}": trial.option_A.attributes[i]["value"] for i in range(n)},
                **{f"B_attribute_{i+1}": trial.option_B.attributes[i]["value"] for i in range(n)},
                "p_A": p_a,
                "label": trial.label,
                "persona": k,
                "trial": t_i,
                "sample": s,
                "argmax_is_choice_token": bool(top in ids),
            }
            decisions.append(row)
    decisions = pd.DataFrame(decisions)
    decisions.to_csv(out_dir / f"selections_{tag}.csv", index=False)
    invalid = 1 - decisions["argmax_is_choice_token"].mean()
    if invalid > 0:
        print(f"  note: unconstrained argmax was neither 'A' nor 'B' on {invalid:.1%} of decision trials")

    # 2. recover the predicted learned latent per persona
    recovered = {}
    for k in persona_indices:
        rows = decisions[decisions["persona"] == k]
        sc = personas[k]
        A = rows[[f"A_attribute_{i+1}" for i in range(n)]].to_numpy(float)
        B = rows[[f"B_attribute_{i+1}" for i in range(n)]].to_numpy(float)
        recovered[k] = estimator.fit(A, B, rows["selection"].tolist(), sc.mins, sc.maxs)
    pd.DataFrame(
        [{"scenario": personas[k].short_name, **{f"b_{c}": v for c, v in zip(estimator.latent_columns(), recovered[k])}} for k in persona_indices]
    ).to_csv(out_dir / f"recovered_{tag}.csv", index=False)
    print(f"  decisions: {len(decisions)} rows, accuracy {(decisions['selection'] == decisions['label']).mean():.3f}, {time.time() - t0:.0f}s")

    # 4. reports in fresh contexts, one freshly drawn option pair per sample
    report_trials = D.fresh_trials(personas, rs["samples_per_report"], seeds["reports"])
    reported, parse, raw = {}, {}, []
    for schema in schemas:
        reported[schema.name], parse[schema.name] = {}, {}
        n_batches = len(schema.batches(personas[persona_indices[0]].names))
        for b_i in range(n_batches):
            # Batch name, block and key count are the same for every persona; only the
            # keys' wording differs. Take them from one persona so nothing downstream
            # depends on the reply loop having run at least once.
            meta = schema.batches(personas[persona_indices[0]].names)[b_i]
            batch_name, block = meta.name, meta.block
            texts, index = [], []
            for k in persona_indices:
                batch = schema.batches(personas[k].names)[b_i]
                for s, trial in enumerate(report_trials[k]):
                    texts.append(chat_prompt(tok, batch.prompt_base + trial.generate_choice()))
                    index.append((k, s))
            replies = generate_texts(model, tok, texts, rs["max_new_tokens"], rs["report_temperature"], batch_size, device)
            per_persona, parsed_rows = {}, []
            for (k, s), reply in zip(index, replies):
                batch = schema.batches(personas[k].names)[b_i]
                values = schema.parse(reply, batch)
                raw.append({"schema": schema.name, "batch": batch.name, "scenario": personas[k].short_name, "sample": s, "reply": reply, "parsed": values is not None})
                if values is None:
                    continue
                per_persona.setdefault(k, []).append(values)
                trial = report_trials[k][s]
                columns = rule.blocks(n).get(batch.block) or [f"{batch.block}_{i+1}" for i in range(len(values))]
                parsed_rows.append(
                    {
                        "explaining_model": tag,
                        "version": batch.name,
                        "scenario": personas[k].short_name,
                        **{f"report_{c}": v for c, v in zip(columns, values)},
                        **{f"A_attribute_{i+1}": trial.option_A.attributes[i]["value"] for i in range(n)},
                        **{f"B_attribute_{i+1}": trial.option_B.attributes[i]["value"] for i in range(n)},
                    }
                )
            parse[schema.name][batch_name] = (len(parsed_rows), len(texts), block)
            reported[schema.name][block] = {k: np.mean(v, axis=0) for k, v in per_persona.items()}
            columns = rule.blocks(n).get(block) or [f"{block}_{i+1}" for i in range(len(meta.keys))]
            header = (
                ["explaining_model", "version", "scenario"]
                + [f"report_{c}" for c in columns]
                + [f"A_attribute_{i+1}" for i in range(n)]
                + [f"B_attribute_{i+1}" for i in range(n)]
            )
            pd.DataFrame(parsed_rows, columns=header).to_csv(
                out_dir / f"weight_reports_{tag}_{schema.name}_{batch_name}.csv", index=False
            )
            print(f"  reports [{schema.name}/{batch_name}]: {len(parsed_rows)}/{len(texts)} parsed, {time.time() - t0:.0f}s")
    with (out_dir / f"reports_raw_{tag}.jsonl").open("w") as f:
        for row in raw:
            f.write(json.dumps(row) + "\n")

    hidden = {k: data.latents[k] for k in persona_indices}
    return {
        "hidden": hidden,
        "recovered": recovered,
        "reported": reported,
        "parse": parse,
        "decisions": decisions,
        "rule_slices": rule_slices,
        "est_slices": est_slices,
    }


# --- metrics -----------------------------------------------------------------------


def _mean_cosine(pairs):
    values = [v for v in (Estimator.distance(a, b) for a, b in pairs) if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def summarize(details, cfg, comps, data, run_id, stage, fold, checkpoint_step):
    """Turn one or more `collect` details (one per fold) into results rows, one per schema and block."""
    rule, estimator, schemas = comps.rule, comps.estimator, comps.schemas
    gates, me = cfg["gates"], cfg["model_estimating"]
    n = data.attribute_count
    rule_slices, est_slices = rule.block_slices(n), estimator.block_slices()
    decisions = pd.concat([d["decisions"] for d in details], ignore_index=True)
    accuracy = float((decisions["selection"] == decisions["label"]).mean())
    decision_gate = accuracy >= gates["min_decision_accuracy"]
    label = f"{stage}{'' if not fold else f' fold {fold}'} step {checkpoint_step}"
    if decision_gate:
        print(f"GATE min_decision_accuracy passed at {label}: {accuracy:.3f} >= {gates['min_decision_accuracy']}")
    else:
        print(f"GATE min_decision_accuracy FIRED at {label}: {accuracy:.3f} < {gates['min_decision_accuracy']}")

    hidden = {k: v for d in details for k, v in d["hidden"].items()}
    recovered = {k: v for d in details for k, v in d["recovered"].items()}
    blocks = list(dict.fromkeys(list(rule_slices) + list(est_slices)))
    for schema in schemas:
        for b in schema.batches([f"attribute_{i}" for i in range(1, n + 1)]):
            if b.block not in blocks:
                blocks.append(b.block)

    base = {"run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"), **hyperparameter_columns(cfg)}
    rows = []
    # Chance depends on the rule, the block and the recovered latents, never on the
    # report schema, so it is computed once per block and reused across schemas.
    chance_by_block = {}
    for schema in schemas:
        reported = {}
        parse = {}
        for d in details:
            for block, per in d["reported"].get(schema.name, {}).items():
                reported.setdefault(block, {}).update(per)
            for batch_name, (parsed, total, block) in d["parse"].get(schema.name, {}).items():
                p, t, _ = parse.get(batch_name, (0, 0, block))
                parse[batch_name] = (p + parsed, t + total, block)
        for block in blocks:
            row = {
                **base,
                "report_schema": schema.name,
                "stage": stage,
                "introspection_fold": fold,
                "checkpoint_step": checkpoint_step,
                "block": block,
                "n_personas": len(recovered),
                "decision_accuracy": accuracy,
                "decision_gate": decision_gate,
            }
            hid = {k: v[rule_slices[block]] for k, v in hidden.items()} if block in rule_slices else {}
            rec = {k: v[est_slices[block]] for k, v in recovered.items()} if block in est_slices else {}
            rep = reported.get(block, {})
            if hid and rec:
                pairs = [(rec[k], hid[k]) for k in rec if k in hid]
                row["recovered_vs_hidden"], row["recovered_vs_hidden_pearson"] = _mean_cosine(pairs), pooled_pearson(pairs)
            if rep and rec:
                pairs = [(rep[k], rec[k]) for k in rep if k in rec]
                row["faithfulness"], row["faithfulness_pearson"] = _mean_cosine(pairs), pooled_pearson(pairs)
            if rec and block in rule_slices:
                if block not in chance_by_block:
                    chance_by_block[block] = chance_level(
                        rule, block, rec, n, me["chance_draws"], cfg["model_hyperparameters"]["seed"]
                    )
                row["chance"], row["chance_pearson"] = chance_by_block[block]
            if rep and hid:
                pairs = [(hid[k], rep[k]) for k in rep if k in hid]
                row["hidden_vs_reported"], row["hidden_vs_reported_pearson"] = _mean_cosine(pairs), pooled_pearson(pairs)
            feeding = [(name, p, t) for name, (p, t, blk) in parse.items() if blk == block]
            if feeding:
                name, p, t = feeding[0]
                rate = p / t if t else float("nan")
                row["parse_rate"], row["parse_gate"] = rate, rate >= gates["min_parse_rate"]
                if row["parse_gate"]:
                    print(f"GATE min_parse_rate passed at {label} [{schema.name}/{name}]: {rate:.3f} >= {gates['min_parse_rate']}")
                else:
                    print(f"GATE min_parse_rate FIRED at {label} [{schema.name}/{name}]: {rate:.3f} < {gates['min_parse_rate']}")
                row["n_personas_reported"] = len(rep)
            rows.append(row)
    return rows


def append_results(rows, cfg, path="results/results.csv"):
    """Append rows to results/results.csv with the same columns every run. Reads all blocks through hyperparameter_columns."""
    columns = results_columns(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).reindex(columns=columns)
    if path.exists():
        existing = list(pd.read_csv(path, nrows=0).columns)
        if existing != columns:
            raise RuntimeError(
                f"{path} has columns {existing} but this pipeline writes {columns}; move the old file aside"
            )
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)
    return frame


def evaluate_model(model, tok, cfg, comps, data, device, run_id, stage, fold, checkpoint_step, persona_indices, results_dir="results"):
    """Steps 1 to 7 for a model already in memory. Returns (rows, detail)."""
    tag = f"{stage}{'' if not fold else f'-fold{fold}'}_step-{checkpoint_step}"
    print(f"evaluating {tag} on {len(persona_indices)} personas")
    detail = collect(model, tok, cfg, comps, data, device, list(persona_indices), tag, Path(results_dir) / run_id)
    rows = summarize([detail], cfg, comps, data, run_id, stage, fold, checkpoint_step)
    append_results(rows, cfg, Path(results_dir) / "results.csv")
    return rows, detail


def evaluate_checkpoint(cfg, checkpoint_dir, run_id=None, stage="decision", fold=0, step=None, results_dir="results"):
    """Steps 1 to 7 for a saved checkpoint: loads the base model, applies the adapter, evaluates all personas."""
    from src.config import resolve
    from src.train import load_model_and_tokenizer

    comps = resolve(cfg)
    comps.backend.setup()
    device = comps.backend.get_device()
    data = D.load_data(cfg, comps.rule)
    model, tok = load_model_and_tokenizer(cfg, device)
    checkpoint_dir = Path(checkpoint_dir)
    if cfg["model_hyperparameters"]["finetune"] == "lora":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    else:
        from src.train import _from_pretrained

        model = _from_pretrained(str(checkpoint_dir), dtype=next(model.parameters()).dtype).to(device)
    run_id = run_id or checkpoint_dir.parents[1].name
    if step is None:
        step = int(checkpoint_dir.name.split("-")[-1]) if checkpoint_dir.name.startswith("step-") else 0
    personas = list(range(cfg["model_training"]["instances"]))
    return evaluate_model(model, tok, cfg, comps, data, device, run_id, stage, fold, step, personas, results_dir)
