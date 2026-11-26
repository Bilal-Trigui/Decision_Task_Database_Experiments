"""Plots from results/train_log.csv and results/results.csv, saved to results/plots/.

  <run_id>_training.png        loss and accuracy over steps, validation points marked: first-answer-token
                               accuracy for decisions, all-answer-token accuracy for introspection
  <run_id>_checkpoints.png     recovered-versus-hidden, and faithfulness against chance, per checkpoint and block,
                               four measures, at the largest estimation trial count
  <run_id>_estimation_trials.png  with a sweep over estimation_trials_per_persona: recovery and faithfulness
                               against the trial count the rule was recovered from, one line per decision checkpoint
  <run_id>_scatter_<block>.png Plunkett's Figure 2 scatter: recovered weight against reported weight, final checkpoint
Each plot is saved twice: the picture, and <same name>.csv beside it holding exactly the points
drawn, one row per point, so any plot can be redrawn by hand. To redraw a past run's plots from its
results (downloaded from the Hub first, if the run was elsewhere):
    python -m src.plots <run_id> [--results-dir results]
Reads none of the six settings directly; everything comes from the CSVs.
"""
from pathlib import Path

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def save_points(points, png_path, columns):
    """The points a plot drew, as <plot name>.csv beside the picture. `points` is a list of dicts."""
    path = Path(png_path).with_suffix(".csv")
    pd.DataFrame(points, columns=columns).to_csv(path, index=False)
    return path


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
    points = []
    for (stage, fold, origin), part in log.groupby(["stage", "fold", "_from"]):
        name = stage if not fold else f"{stage} fold {fold}"
        if origin >= 0:
            name += f" from {int(origin)}"
        x = part["step"] + max(origin, 0)
        acc, val_acc = (("first_token_accuracy", "val_first_token_accuracy") if stage == "decision"
                        else ("answer_token_accuracy", "val_answer_token_accuracy"))
        loss, accuracy = part["loss"], pd.to_numeric(part[acc], errors="coerce")
        axes[0].plot(x, loss, label=name, linewidth=1)
        axes[1].plot(x, accuracy, label=name, linewidth=1)
        val = part[pd.to_numeric(part["val_loss"], errors="coerce").notna()]
        vx = val["step"] + max(origin, 0)
        val_loss, val_accuracy = pd.to_numeric(val["val_loss"]), pd.to_numeric(val[val_acc], errors="coerce")
        if not val.empty:
            axes[0].scatter(vx, val_loss, marker="x", s=40, label=f"{name} val")
            axes[1].scatter(vx, val_accuracy, marker="x", s=40, label=f"{name} val")
        for panel, line, xs, ys in (("loss", name, x, loss), ("accuracy", name, x, accuracy),
                                    ("loss", f"{name} val", vx, val_loss), ("accuracy", f"{name} val", vx, val_accuracy)):
            points += [{"panel": panel, "line": line, "stage": stage, "fold": fold,
                        "introspection_from_step": origin if origin >= 0 else None, "step": a, "value": b}
                       for a, b in zip(xs, ys)]
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
    save_points(points, path, ["panel", "line", "stage", "fold", "introspection_from_step", "step", "value"])
    return path


# the line each stage is drawn as, and its marker: the base model, untrained, is a star at step 0
SERIES = {"decision": "", "base_model": " base model (untrained)"}
MARKERS = {"": "o", " base model (untrained)": "*", " after introspection": "s"}
LINES = {"": "-", " base model (untrained)": "-", " after introspection": "--"}
# introspection toward a target other than hidden: its own marker, so its line is told from hidden's
TARGET_MARKERS = {"recovered": "D", "shuffled": "x", "random": "v"}
COLOUR_ORDER = ["main", "screen_index", "screen_cut", "screen_weight", "cut",
                "main_report main", "main_report screen_index", "main_report screen_cut", "main_report screen_weight",
                "screen_cut_units screen_cut", "main_report_units main", "main_report_units screen_index",
                "main_report_units screen_cut"]
STAGE_NAMES = {"": "decision", " base model (untrained)": "base model (untrained)",
               " after introspection": "after introspection", "chance": "chance"}
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
    targets = (res["introspection_target"].fillna("").astype(str) if "introspection_target" in res.columns
               else pd.Series("", index=res.index))
    res["_series"] = [SERIES[stage] if stage in SERIES else
                      " after introspection" + ("" if target in ("", "hidden") else f" ({target})")
                      for stage, target in zip(res["stage"], targets)]
    for series in set(res["_series"]) - set(MARKERS):
        target = series.split("(")[-1].rstrip(")")
        MARKERS[series], LINES[series] = TARGET_MARKERS.get(target, "s"), "--"
        STAGE_NAMES[series] = f"after introspection toward {target}"
    res = res.sort_values("_x")
    blocks = list(dict.fromkeys(res["block"]))

    # one row of panels per measure; results written before a measure existed simply leave it empty
    panels = [(f"recovered_vs_hidden{s}", f"faithfulness{s}", f"chance{s}", measure) for s, measure in
              (("", "cosine"), ("_centred", "centred cosine"), ("_pearson", "correlation"), ("_spearman", "rank correlation"))]
    res = res.reindex(columns=list(res.columns) + [c for p in panels for c in p[:3] if c not in res.columns])
    asked = res[res["report_batch"] != ""]
    # the schema name is only needed in a label when a run asks more than one schema
    several = asked["report_schema"].nunique() > 1
    # colour is what is measured (a block, or a question's block), the same in every panel; marker
    # and line style are the stage. One legend for the whole figure, outside the panels
    colours, points = {}, []

    def colour(name):
        if name not in colours:
            # a fixed colour per block and question, so every run's plots match; others take the rest
            short = name.split("/")[-1]
            index = COLOUR_ORDER.index(short) if short in COLOUR_ORDER else len(COLOUR_ORDER) + len(colours)
            colours[name] = plt.get_cmap("tab10")(index % 10)
        return colours[name]

    def draw(ax, panel, measure, name, series, part, column, **style):
        part = part[part[column].notna()]
        if part.empty:
            return
        ax.plot(part["_x"], part[column], color=colour(name), **style)
        points.extend({"panel": panel, "measure": measure, "line": name, "stage": STAGE_NAMES[series],
                       "x": x, "x_label": ticks.get(x, str(int(x))), "value": y}
                      for x, y in zip(part["_x"], part[column]))

    fig, axes = plt.subplots(len(panels), 2, figsize=(13, 4 * len(panels)), sharex=True)
    for row, (recovery, faith, chance, measure) in enumerate(panels):
        left, right = axes[row][0], axes[row][1]
        # recovery is the decision checkpoint's: introspection reuses that recovered rule, so its
        # rows would only repeat the decision points underneath
        for (block, series), part in res[~res["_series"].str.startswith(" after introspection")].groupby(["block", "_series"], sort=False):
            draw(left, "recovered vs hidden", measure, block, series, part.drop_duplicates("_x"), recovery,
                 marker=MARKERS[series], linestyle=LINES[series])
        # one line per question and block: main_report answers four blocks, each its own line
        for (schema, batch, block, series), part in asked.groupby(["report_schema", "report_batch", "block", "_series"], sort=False):
            name = batch if batch == block else f"{batch} {block}"
            draw(right, "faithfulness", measure, f"{schema}/{name}" if several else name, series, part, faith,
                 marker=MARKERS[series], linestyle=LINES[series])
        for block in blocks:
            part = res[(res["block"] == block) & (res["_series"] == "")].drop_duplicates("_x")
            draw(right, "chance", measure, block, "chance", part, chance, linestyle=":", marker=".", linewidth=1)
        left.set(title=f"recovered vs hidden ({measure})", ylabel=measure, ylim=(-1.05, 1.05))
        right.set(title=f"faithfulness vs chance ({measure})", ylabel=measure, ylim=(-1.05, 1.05))
        for ax in (left, right):
            ax.axhline(0, color="grey", linewidth=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("checkpoint step")
        ax.set_xticks(sorted(set(res["_x"])))
        ax.set_xticklabels([ticks.get(v, str(int(v))) for v in sorted(set(res["_x"]))], fontsize=8)
    ordered = sorted(colours, key=lambda n: (COLOUR_ORDER.index(n.split("/")[-1]) if n.split("/")[-1] in COLOUR_ORDER
                                             else len(COLOUR_ORDER), n))
    handles = [Line2D([], [], color=colours[name], linewidth=3, label=name) for name in ordered]
    handles.append(Line2D([], [], color="none", label=""))
    handles += [Line2D([], [], color="black", marker=MARKERS.get(series, "."), linestyle=LINES.get(series, ":"),
                       label=STAGE_NAMES[series]) for series in list(dict.fromkeys(res["_series"])) + ["chance"]]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.80, 0.5), fontsize=9, frameon=False,
               title="colour: what is measured\nmarker: stage", title_fontsize=9)
    fig.suptitle(run_id)
    fig.tight_layout(rect=(0, 0, 0.80, 0.98))
    path = Path(out_dir) / f"{run_id}_checkpoints.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    save_points(points, path, ["panel", "measure", "line", "stage", "x", "x_label", "value"])
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
    points = []

    def keep(panel, measure, line, xs, ys):
        points.extend({"panel": panel, "measure": measure, "line": line, "estimation_trials": a, "value": b}
                      for a, b in zip(xs, ys))

    for row, (suffix, measure) in enumerate(measures):
        left, right = axes[row]
        for (block, stage, step), part in res.groupby(["block", "stage", "checkpoint_step"], sort=True):
            part = part.drop_duplicates("estimation_trials").sort_values("estimation_trials")
            name = "base model" if stage == "base_model" else f"step {int(step)}"
            left.plot(part["estimation_trials"], part[f"recovered_vs_hidden{suffix}"], marker="o", label=f"{block} {name}")
            keep("recovered vs hidden", measure, f"{block} {name}", part["estimation_trials"], part[f"recovered_vs_hidden{suffix}"])
        asked = res[res["report_batch"] != ""]
        for (batch, stage, step), part in asked.groupby(["report_batch", "stage", "checkpoint_step"], sort=True):
            part = part.sort_values("estimation_trials")
            name = "base model" if stage == "base_model" else f"step {int(step)}"
            right.plot(part["estimation_trials"], part[f"faithfulness{suffix}"], marker="o", label=f"{batch} {name}")
            keep("faithfulness", measure, f"{batch} {name}", part["estimation_trials"], part[f"faithfulness{suffix}"])
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
    save_points(points, path, ["panel", "measure", "line", "estimation_trials", "value"])
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
        xs, ys, points = [], [], []
        for scenario, row in mean_reports.iterrows():
            if scenario not in recovered.index:
                continue
            for c in columns:
                if f"b_{c}" in recovered.columns:
                    xs.append(recovered.loc[scenario, f"b_{c}"])
                    ys.append(row[f"report_{c}"])
                    points.append({"scenario": scenario, "attribute": c, "recovered": xs[-1], "reported": ys[-1]})
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
        save_points(points, path, ["scenario", "attribute", "recovered", "reported"])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redraw one run's plots, and their data files, from its results.")
    parser.add_argument("run_id", help="the run's id, the name of its folder under the results folder")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    make_plots(args.run_id, args.results_dir)
