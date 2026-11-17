# configs/minimum_threshold_one_persona/cut_sweep

Cut sweeps on Qwen3-8B: one persona's weights and screen (caffeine >= 35 mg), with the share of training trials
that the weights decide swept from 75% down to 0%. Each run branches from that persona's run at decision step 1500 and
trains on to step 3000 under the new mix, so the sweep asks: as the cut decides more of the answers, what happens to
the weight reports and the cut reports?

`wXX_cYY`: XX% of the 5000 training trials are decided by the weights alone (both options pass the screen). YY% are
decided by the cut: half are C (both options fail it), and in the other half one option fails it, so the other wins.

Those one-fails trials differ between the first A sweep and the matched four:

- **First A sweep** (`cut_sweep_wXX_cYY`, run 2026-11-07/08): `cut_overturns_only` true. Every one-fails trial is one
  where the cut overturns what the weights alone would pick. That means every such trial is answered against the
  weights: a fit of the four unscreened weights to those trials alone has cosine -1.00 with the true ones, and the
  share of A/B training answers the weights alone get right falls from 86% (w75) to 62% (w45) to 0% (w00). So in that
  sweep "the cut decides more" and "the data points at the opposite weights" rise together. The choices did not
  follow: in all five mixes checked (w75, w45, w25, w05, w00, steps 2000 and 3000) the rule recovered from the 1000
  natural estimation trials has cosine 0.96 to 1.00 with the true weights over the four unscreened attributes, and
  the both-pass trials are answered by the true weights 88 to 98% of the time. The model already held the screen at
  step 1500, and the screen explains every one of those trials. What stays open is the reports: a change in the
  weight reports across that sweep can come from either cause. These trials are, on the other hand, the only
  one-fails trials that show the screen at work, since in the others the weights alone give the same answer.
- **Matched four** (`cut_sweep_<persona>_wXX_cYY`, rebuilt 2026-11-16, not run): `cut_overturns_only` false. The
  one-fails trials are taken as drawn, so the weights alone get about half of them right and they say nothing about
  the weights either way (93% of A/B answers at w75, 81% at w45, 50% at w00). Only the share of trials the cut
  decides changes across the sweep. The cost: about half of the one-fails trials no longer show the screen at work,
  and the matched four are not a like-for-like repeat of the first A sweep. Leaving `--as-drawn` off builds the
  first sweep's kind of data for any persona.

| mix | weights decide | C | one option fails |
|---|---|---|---|
| `w75_c25` | 75% | 12.5% | 12.5% |
| `w65_c35` | 65% | 17.5% | 17.5% |
| `w55_c45` | 55% | 22.5% | 22.5% |
| `w45_c55` | 45% | 27.5% | 27.5% |
| `w35_c65` | 35% | 32.5% | 32.5% |
| `w25_c75` | 25% | 37.5% | 37.5% |
| `w15_c85` | 15% | 42.5% | 42.5% |
| `w05_c95` | 5% | 47.5% | 47.5% |
| `w00_c100` | 0% | 50% | 50% |

## The sweeps

| folder | files | persona (weights) | branches from (step 1500) | questions | evaluated every |
|---|---|---|---|---|---|
| `persona_a/` | `cut_sweep_wXX_cYY` | A (0, -83, -38, -100, 31) | 20261106-152354, A's first 10000-step run (adapter and AdamW state) | the original four | 100; **all nine ran** 2026-11-07/08 |
| `persona_a/` | `cut_sweep_a_wXX_cYY` | A | 20261110-211307 (A, 3000 steps, clipping 1.0) | every question (probes_a) | 250; not run |
| `persona_b/` | `cut_sweep_b_wXX_cYY` | B (0, 100, 20, -50, -70) | 20261110-112915 (B, 3000 steps) | every question (probes_b) | 250; not run |
| `persona_a_inverse/` | `cut_sweep_a_inverse_wXX_cYY` | A-inverse (0, 83, 38, 100, -31) | 20261113-170347 (A-inverse) | every question | 250; not run |
| `persona_b_inverse/` | `cut_sweep_b_inverse_wXX_cYY` | B-inverse (0, -100, -20, 50, 70) | 20261112-235923 (B-inverse) | every question | 250; not run |
| `persona_c/` | `cut_sweep_c_wXX_cYY` | C (0, 100, -30, -100, 30), caffeine >= 17.5 mg | 20261116-214255 (C) | every question (C's base) | 250; not run |
| `persona_c_inverse/` | `cut_sweep_c_inverse_wXX_cYY` | C-inverse (0, -100, 30, 100, -30), caffeine >= 17.5 mg | 20261116-220846 (C-inverse) | every question | 250; not run |
| `persona_e_positive/` | `cut_sweep_e_positive_wXX_cYY` | E-positive (0, 100, 40, 70, 25), caffeine >= 17.5 mg | 20261116-221042 (E-positive) | every question | 250; not run |
| `persona_e_negative/` | `cut_sweep_e_negative_wXX_cYY` | E-negative (0, -100, -40, -70, -25), caffeine >= 17.5 mg | 20261116-221120 (E-negative) | every question | 250; not run |

The C and E sweeps (added 2026-11-16) are built the same way as the matched four (`--as-drawn`, every 250, branching
at 1500 from a 3000-step run with save_optimizer last, so AdamW starts fresh) and ask their base files' questions,
which add `cut_probe_caffeine_full`. Their personas cut at 25% of the caffeine range (17.5 mg), not the 35 mg of A and
B, so they are matched with each other but not cut-for-cut with A and B.

The four `cut_sweep_<persona>_*` sweeps are one matched set: their source runs trained identically (3000 steps,
learning rate 2e-4, clipping 1.0, the same seeds; only the persona's weights differ), every branch asks the same
questions, and every branch is evaluated at 1750, 2000, ..., 3000 on the same grid as the base runs. Those source
runs saved AdamW's state only at their last step, so these branches start AdamW fresh at step 1500 (the older A sweep
restored it); all four are alike in this. A's matched data is in `persona_a/screen_as_drawn/`, beside the older A
sweep's folders, which are left as that sweep ran on them.

Every experiment has three files, all complete (nothing inherited) and differing only in batch sizes:
`.json` (16 GB card), `_24gb.json` (L4, A10) and `_40gb.json` (A100 40GB: the batch of 10 in two passes of 5 with gradient
checkpointing off, eval batches 96). The older A sweep has `.json` and `_24gb.json` only.

## Run

Each run needs its source run's step-1500 adapter. If `checkpoints/<source run>/decision/step-1500/` is not on the
machine, it is downloaded from the Hub repo the runs were pushed to (`adamchanna/bdp-plunkett-qwen3`, public, no
token needed; `src/persist.py` `pull_branch_source`), so a fresh A100 box needs nothing but the repo.

One sweep, all nine mixes in order (w00 first), each its own process; restart the same command to carry on after a
stop:

```
python scripts/run_queue.py --name cut_sweep_b_inverse configs/minimum_threshold_one_persona/cut_sweep/persona_b_inverse/*_40gb.json
python scripts/run_queue.py --name cut_sweep_b         configs/minimum_threshold_one_persona/cut_sweep/persona_b/*_40gb.json
python scripts/run_queue.py --name cut_sweep_a_inverse configs/minimum_threshold_one_persona/cut_sweep/persona_a_inverse/*_40gb.json
python scripts/run_queue.py --name cut_sweep_a         configs/minimum_threshold_one_persona/cut_sweep/persona_a/cut_sweep_a_*_40gb.json
python scripts/run_queue.py --name cut_sweep_c          configs/minimum_threshold_one_persona/cut_sweep/persona_c/*_40gb.json
python scripts/run_queue.py --name cut_sweep_c_inverse  configs/minimum_threshold_one_persona/cut_sweep/persona_c_inverse/*_40gb.json
python scripts/run_queue.py --name cut_sweep_e_positive configs/minimum_threshold_one_persona/cut_sweep/persona_e_positive/*_40gb.json
python scripts/run_queue.py --name cut_sweep_e_negative configs/minimum_threshold_one_persona/cut_sweep/persona_e_negative/*_40gb.json
```

One run: `python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/cut_sweep/persona_b/cut_sweep_b_w45_c55_40gb.json`.
Progress: `python scripts/run_queue.py --name cut_sweep_b --status`. Summary: `python scripts/sweep_summary.py cut_sweep_b --plot`.

## Data and rebuilding

`data/minimum_threshold_one_persona/cut_sweep/persona_<b|a_inverse|b_inverse>/wXX_cYY/` and
`.../persona_a/screen_as_drawn/wXX_cYY/` (the matched four), `.../persona_a/wXX_cYY/` (the first A sweep), each built
from that persona's own data folder with only the trial mix changed. `tests/test_cut_sweep.py` checks every folder
holds exactly the shares its config names, under its persona's weights.

```
python scripts/make_cut_sweep.py --as-drawn --persona b_inverse --prefix cut_sweep_b_inverse --every 250 --gpus 16gb 24gb 40gb \
    --source-run 20261112-235923_paper_minimum_threshold_Qwen3-8B_b-inverse \
    --base configs/minimum_threshold_one_persona/inverse/persona_b/one_persona_b_inverse.json
```

(and likewise for the others: C and E take `--base` from `base/persona_c/one_persona_c.json`,
`inverse/persona_c/one_persona_c_inverse.json`, `base/persona_e/one_persona_e_positive.json`,
`inverse/persona_e/one_persona_e_negative.json`; A adds `--data-subfolder screen_as_drawn`. Without `--as-drawn` the script builds
the first A sweep's kind of data, and a bare `python scripts/make_cut_sweep.py` rebuilds that sweep as it ran;
`--no-fetch` writes the files without looking for the source checkpoint). No 14B or 32B sweep: a branch needs a source run on the same model.
