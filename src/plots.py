"""Plots from results/train_log.csv and results/results.csv, saved to results/plots/.

  <run_id>_training.png        loss and accuracy over steps, validation points marked: first-answer-token
                               accuracy for decisions, all-answer-token accuracy for introspection
  <run_id>_checkpoints.png     recovered-versus-hidden, and faithfulness against chance, per checkpoint and block,
                               four measures, at the largest estimation trial count
  <run_id>_estimation_trials.png  with a sweep over estimation_trials_per_persona: recovery and faithfulness
                               against the trial count the rule was recovered from, one line per decision checkpoint
  <run_id>_scatter_<block>.png Plunkett's Figure 2 scatter: recovered weight against reported weight, final checkpoint
Reads none of the six settings directly; everything comes from the CSVs.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_training(train_log, run_id, out_dir):
    """Loss and accuracy per step. An introspection stage that names the decision step it branched
    from is drawn starting there, so each branch leaves the decision curve where it ran.

    Accuracy is the first answer token for decisions, where that token is the A or B answer, and
    every answer token for introspection, whose first token is always "{" and is never read.
    """
    log = train_log[train_log["run_id"] == run_id].copy()
    if log.empty:
        return None
    if "introspection_from_step" not in log.columns:
        log["introspection_from_step"] = float("nan")
    log["_from"] = pd.to_numeric(log["introspection_from_step"], errors="coerce").fillna(-1)
    for column in ("answer_token_accuracy", "val_answer_token_accuracy"):
        if column not in log.columns:
            log[column] = float("nan")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (stage, fold, origin), part in log.groupby(["stage", "fold", "_from"]):
        name = stage if not fold else f"{stage} fold {fold}"
        if origin >= 0:
            name += f" from {int(origin)}"
        x = part["step"] + max(origin, 0)
        acc, val_acc = (("first_token_accuracy", "val_first_token_accuracy") if stage == "decision"
                        else ("answer_token_accuracy", "val_answer_token_accuracy"))
        axes[0].plot(x, part["loss"], label=name, linewidth=1)
        axes[1].plot(x, pd.to_numeric(part[acc], errors="coerce"), label=name, linewidth=1)
        val = part[pd.to_numeric(part["val_loss"], errors="coerce").notna()]
        if not val.empty:
            vx = val["step"] + max(origin, 0)
            axes[0].scatter(vx, pd.to_numeric(val["val_loss"]), marker="x", s=40, label=f"{name} val")
            axes[1].scatter(vx, pd.to_numeric(val[val_acc], errors="coerce"), marker="x", s=40, label=f"{name} val")
    axes[0].set(title="training loss", xlabel="step", ylabel="loss")
    axes[1].set(title="accuracy: first answer token (decision), every answer token (introspection)",
                xlabel="step", ylabel="accuracy", ylim=(0, 1.02))
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(run_id)
    fig.tight_layout()
    path = Path(out_dir) / f"{run_id}_training.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# the line each stage is drawn as, and its marker: the base model, untrained, is a star at step 0
SERIES = {"decision": "", "base_model": " base model (untrained)"}
MARKERS = {"": "o", " base model (untrained)": "*", " after introspection": "s"}
# the base model's rows were called chance_level before; results written then are read as base_model
OLD_STAGE_NAMES = {"chance_level": "base_model"}


def _checkpoint_x(res):
    """X position per row, with the introspection folds placed after the last decision checkpoint.

    Stage two is trained on top of the final stage-one checkpoint and counts its own steps from
    zero, so its step number is not on the same axis as the decision checkpoints. Putting it one
    gap past the last of them reads the way the experiment ran: the curve across training, then
    what report training did to it.

    A row that names the decision step its introspection branched from (introspection_from_step)
    goes at that step instead, so a run with several introspection points draws an "after
    introspection" curve alongside the decision curve. The base model, untrained, goes at step 0.
    """
    # distinct steps only: a schema with several questions puts several rows at each step, and
    # the diff of a series with repeats has zeros in it, which would put the introspection point
    # one unit past the last checkpoint, on top of it
    steps = sorted(set(int(v) for v in res[res["stage"] == "decision"]["checkpoint_step"]))
    last = steps[-1] if steps else 0
    gap = max(min(b - a for a, b in zip(steps, steps[1:])) if len(steps) > 1 else last or 1, 1)
    origins = pd.to_numeric(res["introspection_from_step"], errors="coerce") if "introspection_from_step" in res else None
    x, ticks = [], {}
    for i, (_, row) in enumerate(res.iterrows()):
        if row["stage"] == "decision":
            x.append(row["checkpoint_step"])
        elif row["stage"] == "base_model":
            x.append(0)
        elif origins is not None and pd.notna(origins.iloc[i]):
            x.append(int(origins.iloc[i]))
        else:
            x.append(last + gap)
            ticks[last + gap] = "after\nintrospection"
    return x, ticks


def plot_checkpoints(results, run_id, out_dir):
    """Recovery and faithfulness per checkpoint, as cosine and as correlation, through both stages.

    Four panels: recovery on the left and faithfulness on the right, cosine on top and Plunkett's
    pooled correlation below. The two are different measures of the same comparison and can move
    apart, so neither is shown without the other. Chance is drawn on both faithfulness panels
    because a faithfulness number means nothing without it.
    """
    res = results[results["run_id"] == run_id].copy()
    if res.empty:
        return None
    res = _main_count(res)
    if "report_batch" not in res.columns:
        res["report_batch"] = res["block"]
    res["report_batch"] = res["report_batch"].fillna("")
    # stage two is reported per fold and pooled over both; the pooled row is the result
    folds = res["introspection_fold"].astype(str)
    res = res[res["stage"].isin(["decision", "base_model"]) | (folds == "both")]
    if res.empty:
        return None
    res["_x"], ticks = _checkpoint_x(res)
    res["_series"] = [SERIES.get(stage, " after introspection") for stage in res["stage"]]
    res = res.sort_values("_x")
    blocks = list(dict.fromkeys(res["block"]))

    # one row of panels per measure; results written before a measure existed simply leave it empty
    panels = [(f"recovered_vs_hidden{s}", f"faithfulness{s}", f"chance{s}", measure) for s, measure in
              (("", "cosine"), ("_centred", "centred cosine"), ("_pearson", "correlation"), ("_spearman", "rank correlation"))]
    res = res.reindex(columns=list(res.columns) + [c for p in panels for c in p[:3] if c not in res.columns])
    asked = res[res["report_batch"] != ""]
    fig, axes = plt.subplots(len(panels), 2, figsize=(12, 4 * len(panels)), sharex=True)
    for row, (recovery, faith, chance, measure) in enumerate(panels):
        left, right = axes[row][0], axes[row][1]
        for (block, series), part in res.groupby(["block", "_series"], sort=False):
            part = part.drop_duplicates("_x")
            left.plot(part["_x"], part[recovery], marker=MARKERS[series], label=f"{block}{series}")
        # one line per question and block: main_report answers four blocks, each its own line
        for (schema, batch, block, series), part in asked.groupby(["report_schema", "report_batch", "block", "_series"], sort=False):
            name = batch if batch == block else f"{batch} {block}"
            right.plot(part["_x"], part[faith], marker=MARKERS[series], label=f"{schema}/{name}{series}")
        for block in blocks:
            part = res[(res["block"] == block) & (res["_series"] == "")].drop_duplicates("_x")
            right.plot(part["_x"], part[chance], linestyle="--", marker=".", linewidth=1, label=f"chance {block}")
        left.set(title=f"recovered vs hidden ({measure})", ylabel=measure, ylim=(-1.05, 1.05))
        right.set(title=f"faithfulness vs chance ({measure})", ylabel=measure, ylim=(-1.05, 1.05))
        for ax in (left, right):
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("checkpoint step")
        if ticks:
            ax.set_xticks(sorted(set(res["_x"])))
            ax.set_xticklabels([ticks.get(v, str(int(v))) for v in sorted(set(res["_x"]))], fontsize=8)
    fig.suptitle(run_id)
    fig.tight_layout()
    path = Path(out_dir) / f"{run_id}_checkpoints.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _main_count(res):
    """Only the rows recovered from the largest estimation trial count, the main reading of a sweep."""
    if "estimation_trials" not in res.columns or res["estimation_trials"].isna().all():
        return res
    counts = pd.to_numeric(res["estimation_trials"], errors="coerce")
    return res[counts.isna() | (counts == counts.max())]


def plot_estimation_trials(results, run_id, out_dir):
    """How many trials it takes to find the rule: recovery and faithfulness against the trial count.

    Drawn only for a run that swept estimation_trials_per_persona. One line per decision checkpoint
    (and the base model, untrained), on two rows of panels: cosine, and the centred cosine whose
    chance is near 0. Each panel is one block, or for faithfulness one question.
    """
    res = results[results["run_id"] == run_id].copy()
    if res.empty or "estimation_trials" not in res.columns:
        return None
    res["estimation_trials"] = pd.to_numeric(res["estimation_trials"], errors="coerce")
    res = res[res["stage"].isin(["decision", "base_model"]) & res["estimation_trials"].notna()]
    if res["estimation_trials"].nunique() < 2:
        return None
    for column in ("recovered_vs_hidden_centred", "faithfulness_centred"):
        if column not in res.columns:
            res[column] = float("nan")
    res["report_batch"] = res.get("report_batch", pd.Series("", index=res.index)).fillna("")
    measures = (("", "cosine"), ("_centred", "centred cosine"))
    fig, axes = plt.subplots(len(measures), 2, figsize=(12, 4 * len(measures)), sharex=True, squeeze=False)
    for row, (suffix, measure) in enumerate(measures):
        left, right = axes[row]
        for (block, stage, step), part in res.groupby(["block", "stage", "checkpoint_step"], sort=True):
            part = part.drop_duplicates("estimation_trials").sort_values("estimation_trials")
            name = "base model" if stage == "base_model" else f"step {int(step)}"
            left.plot(part["estimation_trials"], part[f"recovered_vs_hidden{suffix}"], marker="o", label=f"{block} {name}")
        asked = res[res["report_batch"] != ""]
        for (batch, stage, step), part in asked.groupby(["report_batch", "stage", "checkpoint_step"], sort=True):
            part = part.sort_values("estimation_trials")
            name = "base model" if stage == "base_model" else f"step {int(step)}"
            right.plot(part["estimation_trials"], part[f"faithfulness{suffix}"], marker="o", label=f"{batch} {name}")
        left.set(title=f"recovered vs hidden ({measure})", ylabel=measure, ylim=(-1.05, 1.05))
        right.set(title=f"faithfulness ({measure})", ylabel=measure, ylim=(-1.05, 1.05))
        for ax in (left, right):
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("estimation trials per persona")
    fig.suptitle(run_id)
    fig.tight_layout()
    path = Path(out_dir) / f"{run_id}_estimation_trials.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_scatter(run_dir, run_id, out_dir):
    """Recovered against reported weights for the last decision checkpoint, one panel per block."""
    run_dir = Path(run_dir)
    # the main fit only: a sweep's smaller counts are saved beside it as recovered_<tag>_trials-<N>.csv
    recovered_files = sorted((p for p in run_dir.glob("recovered_decision_step-*.csv") if "_trials-" not in p.stem),
                             key=lambda p: int(p.stem.split("-")[-1]))
    if not recovered_files:
        return []
    final = recovered_files[-1]
    tag = final.stem[len("recovered_") :]
    recovered = pd.read_csv(final).set_index("scenario")
    paths = []
    for report_file in sorted(run_dir.glob(f"weight_reports_{tag}_*.csv")):
        try:
            reports = pd.read_csv(report_file)
        except pd.errors.EmptyDataError:
            continue
        if reports.empty:
            continue
        columns = [c[len("report_") :] for c in reports.columns if c.startswith("report_")]
        mean_reports = reports.groupby("scenario")[[f"report_{c}" for c in columns]].mean()
        xs, ys = [], []
        for scenario, row in mean_reports.iterrows():
            if scenario not in recovered.index:
                continue
            for c in columns:
                if f"b_{c}" in recovered.columns:
                    xs.append(recovered.loc[scenario, f"b_{c}"])
                    ys.append(row[f"report_{c}"])
        if not xs:
            continue
        name = report_file.stem[len(f"weight_reports_{tag}_") :]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xs, ys, s=12, alpha=0.6)
        ax.set(xlabel="recovered weight", ylabel="reported weight", title=f"{run_id}\n{name}, {tag}")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)
        fig.tight_layout()
        path = Path(out_dir) / f"{run_id}_scatter_{name}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def make_plots(run_id, results_dir="results"):
    """All plots for one run. Reads results/train_log.csv, results/results.csv and the run's detail files."""
    results_dir = Path(results_dir)
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    log_path, res_path = results_dir / "train_log.csv", results_dir / "results.csv"
    if log_path.exists():
        made.append(plot_training(pd.read_csv(log_path), run_id, out_dir))
    if res_path.exists():
        results = pd.read_csv(res_path)
        results["stage"] = results["stage"].replace(OLD_STAGE_NAMES)
        made.append(plot_checkpoints(results, run_id, out_dir))
        made.append(plot_estimation_trials(results, run_id, out_dir))
    made += plot_scatter(results_dir / run_id, run_id, out_dir)
    made = [p for p in made if p]
    print("plots:", ", ".join(str(p) for p in made) if made else "none")
    return made
