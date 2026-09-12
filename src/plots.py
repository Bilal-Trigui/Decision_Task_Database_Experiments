"""Plots from results/train_log.csv and results/results.csv, saved to results/plots/.

  <run_id>_training.png        loss and first-answer-token accuracy over steps, validation points marked
  <run_id>_checkpoints.png     recovered-versus-hidden, and faithfulness against chance, per checkpoint and block
  <run_id>_scatter_<block>.png Plunkett's Figure 2 scatter: recovered weight against reported weight, final checkpoint
Reads none of the six settings directly; everything comes from the CSVs.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_training(train_log, run_id, out_dir):
    log = train_log[train_log["run_id"] == run_id]
    if log.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (stage, fold), part in log.groupby(["stage", "fold"]):
        name = stage if not fold else f"{stage} fold {fold}"
        axes[0].plot(part["step"], part["loss"], label=name, linewidth=1)
        axes[1].plot(part["step"], part["first_token_accuracy"], label=name, linewidth=1)
        val = part[pd.to_numeric(part["val_loss"], errors="coerce").notna()]
        if not val.empty:
            axes[0].scatter(val["step"], pd.to_numeric(val["val_loss"]), marker="x", s=40, label=f"{name} val")
            axes[1].scatter(val["step"], pd.to_numeric(val["val_first_token_accuracy"]), marker="x", s=40, label=f"{name} val")
    axes[0].set(title="training loss", xlabel="step", ylabel="loss")
    axes[1].set(title="first answer token accuracy", xlabel="step", ylabel="accuracy", ylim=(0, 1.02))
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(run_id)
    fig.tight_layout()
    path = Path(out_dir) / f"{run_id}_training.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _checkpoint_x(res):
    """X position per row, with the introspection folds placed after the last decision checkpoint.

    Stage two is trained on top of the final stage-one checkpoint and counts its own steps from
    zero, so its step number is not on the same axis as the decision checkpoints. Putting it one
    gap past the last of them reads the way the experiment ran: the curve across training, then
    what report training did to it.
    """
    decision = res[res["stage"] == "decision"]["checkpoint_step"]
    last = int(decision.max()) if not decision.empty else 0
    gap = max(int(decision.diff().dropna().min()) if len(decision) > 1 else last or 1, 1)
    x, ticks = [], {}
    for _, row in res.iterrows():
        if row["stage"] == "decision":
            x.append(row["checkpoint_step"])
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
    if "report_batch" not in res.columns:
        res["report_batch"] = res["block"]
    res["report_batch"] = res["report_batch"].fillna("")
    # stage two is reported per fold and pooled over both; the pooled row is the result
    folds = res["introspection_fold"].astype(str)
    res = res[(res["stage"] == "decision") | (folds == "both")]
    if res.empty:
        return None
    res["_x"], ticks = _checkpoint_x(res)
    res = res.sort_values("_x")
    blocks = list(dict.fromkeys(res["block"]))
    asked = res[res["report_batch"] != ""]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = (("recovered_vs_hidden", "faithfulness", "chance", "cosine"),
              ("recovered_vs_hidden_pearson", "faithfulness_pearson", "chance_pearson", "correlation"))
    for row, (recovery, faith, chance, measure) in enumerate(panels):
        left, right = axes[row][0], axes[row][1]
        for block in blocks:
            part = res[res["block"] == block].drop_duplicates("_x")
            left.plot(part["_x"], part[recovery], marker="o", label=block)
        for (schema, batch), part in asked.groupby(["report_schema", "report_batch"]):
            right.plot(part["_x"], part[faith], marker="o", label=f"{schema}/{batch}")
        for block in blocks:
            part = res[res["block"] == block].drop_duplicates("_x")
            right.plot(part["_x"], part[chance], linestyle="--", marker=".", linewidth=1, label=f"chance {block}")
        left.set(title=f"recovered vs hidden ({measure})", ylabel=measure, ylim=(-1.05, 1.05))
        right.set(title=f"faithfulness vs chance ({measure})", ylabel=measure, ylim=(-1.05, 1.05))
        for ax in (left, right):
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.legend(fontsize=7)
    for ax in axes[1]:
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


def plot_scatter(run_dir, run_id, out_dir):
    """Recovered against reported weights for the last decision checkpoint, one panel per block."""
    run_dir = Path(run_dir)
    recovered_files = sorted(run_dir.glob("recovered_decision_step-*.csv"), key=lambda p: int(p.stem.split("-")[-1]))
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
        made.append(plot_checkpoints(pd.read_csv(res_path), run_id, out_dir))
    made += plot_scatter(results_dir / run_id, run_id, out_dir)
    made = [p for p in made if p]
    print("plots:", ", ".join(str(p) for p in made) if made else "none")
    return made
