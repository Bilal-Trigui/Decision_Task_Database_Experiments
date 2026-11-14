# configs/minimum_threshold_one_persona/cut_sweep

Persona A's cut sweep on Qwen3-8B. Persona A's weights (0, -83, -38, -100, 31) and screen (caffeine >= 35 mg), with
the share of training trials that the weights decide swept from 75% down to 0%.

`persona_a/cut_sweep_wXX_cYY.json`: XX% of the 5000 training trials are decided by the weights alone (both options
pass the screen). YY% are decided by the cut: half are C (both options fail it), and in the other half one option
fails it and the cut overturns what the weights alone would pick (`cut_overturns_only`). Each run branches from A's
run 20261106-152354 at decision step 1500 (its adapter and optimizer state) and trains steps 1501 to 3000, evaluated
and saved every 100.

| file | weights decide | C | cut overturns | finished run |
|---|---|---|---|---|
| `cut_sweep_w75_c25.json` | 75% | 12.5% | 12.5% | 20261107-114253 |
| `cut_sweep_w65_c35.json` | 65% | 17.5% | 17.5% | 20261107-141812 |
| `cut_sweep_w55_c45.json` | 55% | 22.5% | 22.5% | 20261107-165502 |
| `cut_sweep_w45_c55.json` | 45% | 27.5% | 27.5% | 20261107-193240 |
| `cut_sweep_w35_c65.json` | 35% | 32.5% | 32.5% | 20261107-220820 |
| `cut_sweep_w25_c75.json` | 25% | 37.5% | 37.5% | 20261108-004347 |
| `cut_sweep_w15_c85.json` | 15% | 42.5% | 42.5% | 20261108-031936 |
| `cut_sweep_w05_c95.json` | 5% | 47.5% | 47.5% | 20261108-055435 |
| `cut_sweep_w00_c100.json` | 0% | 50% | 50% | 20261108-082911 |

Each has a `_24gb` twin, which is the same run with micro-batches of 5 and eval batches of 16. Every file is complete:
nothing is inherited, so edit any value in the file itself.

- **Data:** `data/minimum_threshold_one_persona/cut_sweep/persona_a/wXX_cYY/`
- **Run the whole sweep:** `python scripts/run_queue.py --name cut_sweep configs/minimum_threshold_one_persona/cut_sweep/persona_a/cut_sweep_w75_c25.json ...`
  or the queue cell in `prior_experiment_training.ipynb`.
- **Summary:** `python scripts/sweep_summary.py cut_sweep --plot`
- **A new sweep:** `python scripts/make_cut_sweep.py --name <sweep> ...` writes `configs/minimum_threshold_one_persona/<sweep>/persona_a/`
  and builds its data folders.
