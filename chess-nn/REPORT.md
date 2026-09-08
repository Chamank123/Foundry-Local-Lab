# Pilot NN training report

Reply to `README_PILOT_HANDOFF.md` section 10. Everything below was produced in
a Linux container with **4 CPU cores, no GPU** (`torch.cuda.is_available()` is
`False`). Versions: torch 2.14.0, numpy 2.4.6, numba 0.67.0, python-chess 1.11.2,
Python 3.11.

**Headline:** the pipeline is sound end to end — data, training, export and
runtime parity all pass. But the prescribed 30-epoch baseline is **drastically
undertrained**: it barely beats predicting a constant, and its output spans only
±8 cp. A retuned run on the same data, architecture and encoding reaches 48%
below the constant-predictor baseline. Neither net understands material well
enough to ship; that needs the larger dataset, not more tuning.

---

## 1. Accepted row count and splits

| Quantity | Value |
|---|---|
| `manifest.json` `kept` | 100,000 |
| Rows summed independently from the 4 shards | **100,000** (agrees) |
| `scanned` | 101,971 |
| Rejected | `no_usable_eval` 1,344, `invalid_position` 539, `duplicate` 88 |
| `min_depth` / `stop_reason` | 16 / `limit` |
| Train / validation / test | **97,968 / 1,031 / 1,001** |

Splits are disjoint (0 rows shared between any pair) and sum to 100,000. All
100,000 canonical keys are unique. `stop_reason=limit` means the accepted-row
cap was hit, as expected.

## 2. Schema, label signs, and audit inspection — PASSED

Every assertion in handoff section 5 passes. Additional independent checks:

* All 200 `audit_sample.json` rows re-encoded from their FEN with
  `encode_pychess` reproduce the stored `active` indices **exactly** (0/200
  mismatches), the stored `canonical_key` (0/200), and the stored `mover_cp`
  from `selected_white_cp` (0/200). 105/200 are White to move, so both signs
  are exercised.
* Spot-checked signs: White to move `+6000 -> +6000`; Black to move
  `+69.0 -> -69.0`. The single White-relative -> mover-relative conversion is
  correct and was **not** applied twice.
* Structural: every one of the 100,000 rows has exactly 2 king features;
  padding is `-1` everywhere; active indices lie in `[8, 779]`; `cnt` ranges
  3–37 (mean 22.5), all within the 40-slot capacity.
* `wdl == -1` and `group == 0` on every row, as the pilot requires.

**Label distribution** (worth knowing before reading any loss number): median
0 cp, mean +249 cp, 55.4% of rows within ±100 cp, and **18.9% sitting exactly
on the ±6000 mate clamp** (11,481 at +6000, 7,456 at −6000). That mate mass
matters — see section 6.

## 3. Tests

| Test | Result |
|---|---|
| `python train_nnue.py smoke` | **PASS** (synthetic shard -> train -> export -> runtime) |
| `python test_nn.py` | **NOT RUN — missing dependency, not a code defect** |
| `python test_nn_offline.py` (added here) | **PASS** |

`test_nn.py` does `from bb_numba import board_np` at module scope. `bb_numba.py`
is the engine's and ships in neither ZIP, so the import fails with
`ModuleNotFoundError: No module named 'bb_numba'`. Per handoff section 5.6 it was
**not** stubbed. `test_nn_offline.py` runs everything reachable without the
engine, by loading the engine-independent helpers out of the real `test_nn.py`
with only that one import line removed — `board_np` stays undefined, so any
engine-dependent check raises rather than appearing to pass.

Offline results (random weights, 307 positions):

* data contract, canonical split, atomic shards, exact resume — PASS
* weight-file validation (good and malformed) — PASS
* numba `nn_forward` == numpy reference — max diff **4.92e-07 cp**
* numba `nn_forward` == PyTorch — max diff **5.36e-07 cp**
* legal-move selection demo — legal move on all 306 non-terminal positions
* throughput — **157,131 forward passes/sec**, one core

**Still unestablished:** `encode_board_np(bd)` vs `encode_pychess` parity, and
the runtime driven from a real engine board array. Both need the engine.

## 4. Training runs

Device: CPU (4 cores). Architecture, encoding, `--scale 400`, `--lambda 1.0`,
`--loss mse` and `--seed 0` unchanged throughout; only optimisation settings vary.

**Reference point:** a constant predictor that always outputs the training-set
mean scores **0.24377** validation / **0.25115** test expected-score RMSE. Any
model must be read against that, not against zero.

| Run | Settings | Best epoch | Val RMSE | Test RMSE | vs constant (val) |
|---|---|---|---|---|---|
| Prescribed check | 3 ep, bs 8192, lr 1e-3 | 3 | 0.24415 | 0.25223 | −0.2% |
| **Prescribed baseline** | 30 ep, bs 8192, lr 1e-3 | 30/30 | **0.24187** | **0.24990** | **+0.8%** |
| **Retuned** | 120 ep, bs 256, lr 3e-3 | **81**/120 | **0.12082** | **0.13068** | **+50.4%** |

The baseline command runs in **15 seconds** on 4 CPU cores. At batch 8192 there
are only 12 optimiser steps per epoch, so 30 epochs is **360 steps total** — the
net never leaves its initialisation. Its validation loss was still falling
monotonically at epoch 30 (best epoch = last epoch), the classic undertrained
signature.

Selection used **validation only**. `train()` prints a test metric on every run,
so the sweep driver captured and discarded stdout; the test split was read once
per reported row above and never used to choose between configurations.

Sweeps (15 configurations, `sweep_results.json`, `sweep2_results.json`,
`sweep3_results.json`): shrinking the batch to 256 and raising lr to 3e-3 gave
nearly all the gain. Beyond ~200 epochs validation turns back up (overfitting —
208k parameters against 98k training rows). A cosine schedule did **not** help
(best 0.12246 vs 0.12082). The plateau is ~0.121, so this data is exhausted.

## 5. Export / runtime parity on the trained checkpoints — PASSED

`check_export_parity.py` (added here) compares each trained checkpoint's PyTorch
predictions with its exported NumPy and numba predictions on **1,000 held-out
test-split positions**, checking raw float and engine-facing clamped-int output
separately, as handoff section 8 asks.

| | raw float max diff | clamped-int rows differing |
|---|---|---|
| `pilot_net.npz` | 4.65e-06 cp | **0 / 1000** |
| `pilot_net_tuned.npz` | 4.28e-04 cp | **0 / 1000** |

All three implementations agree. Nothing hits the ±6000 clamp.

## 6. Do the nets actually evaluate chess?

Output range on held-out positions is the clearest summary:

| | min cp | max cp | Pearson r | Spearman r | sign agreement (\|label\|≥50) |
|---|---|---|---|---|---|
| `pilot_net.npz` (baseline) | **−7.5** | **+7.8** | 0.485 | 0.636 | 85.7% |
| `pilot_net_tuned.npz` | **−1198.5** | **+1056.7** | **0.833** | **0.757** | **86.8%** |

The baseline returns essentially the same number for every position: start
position +7.8, a queen up +7.8, a rook up +7.8, K+Q vs K +7.8. **It is not an
evaluator.** Search using it would be search with a constant eval.

The tuned net is a real evaluator — a queen up scores +608, K+Q vs K scores
+1057 / −1199, a pawn up +34, start position +32 — and mover-relative mirror
symmetry is exact on both nets.

**But its material scale is badly wrong.** Controlled test: take 379 realistic
positions, remove one non-king piece, measure the eval change.

| Piece removed | True value | Mean eval change (mover loses it) |
|---|---|---|
| Pawn | 100 cp | −33 cp |
| Knight | 320 cp | **+11 cp** (wrong sign) |
| Bishop | 330 cp | −52 cp |
| Rook | 500 cp | −33 cp |
| Queen | 900 cp | **−87 cp** |

Direction is correct only 65–70% of the time, and magnitudes are roughly an
order of magnitude too small. Two causes, both structural rather than fixable by
tuning:

1. **98k rows is far too little** for 208k parameters. Validation overfits past
   ~epoch 200.
2. **The loss saturates.** Targets are `sigmoid(cp/400)`, so everything beyond
   roughly ±1500 cp maps to the same target and carries almost no gradient. The
   18.9% of rows pinned at ±6000 therefore teach little beyond "winning", and
   the net has no reason to ever output more than ~±1200 cp. Mean absolute
   error is 1101 cp but **median** absolute error is only 73 cp — the mean is
   entirely the mate rows.

This is consistent with the handoff's own position: beating the hand-crafted
engine is not required of this pilot.

## 7. Cold start and speed

* numba compile of `nn_forward`: **< 0.1 s** — negligible against the 90-second
  budget. `encode_board_np` compiles against the engine board array, so its
  compile cost must be measured in the engine checkout.
* **156,000 forward passes/sec** on one core, ~6.4 µs each, non-incremental.
  At 120 s + 0.5 s/move that is roughly 150k leaf evals/sec before search
  overhead. Enough for a working engine; incremental updates remain the
  documented later optimisation.

## 8. Can data collection scale? — Yes, and it should

Preparation and training both behaved correctly and are not the bottleneck. The
whole 100k pilot trains in under 3 minutes on 4 CPU cores, so the planned
1M-example set is very comfortable on CPU and trivial on a GPU.

Concrete requests, in priority order:

1. **Scale to ~1M examples**, as planned. This is the single highest-value
   change; the pilot is data-starved, not code-starved.
2. **Reconsider the mate encoding.** 18.9% of rows at exactly ±6000 is a large
   slice of the data delivering almost no gradient through
   `sigmoid(cp/400)`. Either cap mates nearer ±2000 cp, down-weight them, or
   drop rows whose PV is a forced mate and let search find mates.
3. **Raise `--min-depth` above 16 if throughput allows**, and consider sampling
   across the whole export rather than a prefix — the current pilot is the first
   100k accepted rows of a prefix and is explicitly not representative.
4. **Send `bb_numba.py`** (or the NN folder's place in the engine checkout) so
   `test_nn.py` engine parity and a real search integration can be verified.

## 9. What is NOT established

* Engine parity (`encode_board_np` vs `encode_pychess`) — needs `bb_numba.py`.
* Any alpha-beta search integration, clock safety, completed depth, or games.
* CUDA training — this environment has no GPU.
* Competition-time-control gauntlet against the hand-crafted engine.

No claim is made that this package or dataset has passed a complete training and
engine gauntlet.

## Files

| File | What it is |
|---|---|
| `pilot_net.npz` | The handoff's prescribed 30-epoch baseline. Undertrained; kept for reproducibility. |
| `pilot_net_tuned.npz` | **Recommended checkpoint.** 120 ep, bs 256, lr 3e-3, best epoch 81. |
| `pilot_net_check.npz` | The prescribed 3-epoch check run. |
| `test_nn_offline.py` | Engine-independent subset of `test_nn.py`. |
| `check_export_parity.py` | Trained-checkpoint torch/numpy/numba parity on held-out rows. |
| `eval_report.py` | Chess sanity, held-out quality, cold-start and throughput. |
| `sweep.py`, `sweep2.py`, `sweep3.py`, `sweep*_results.json` | Validation-only hyperparameter search. |

`train_nnue.py` gained one optional flag, `--lr-schedule {none,cosine}`,
defaulting to `none` so the handoff's baseline command reproduces exactly. The
export now also records `epochs`, `batch_size`, `learning_rate` and
`lr_schedule` in the checkpoint metadata. Encoder, architecture, label handling
and split logic are untouched.

## Reproducing

```bash
python -m venv .venv && ./.venv/bin/python -m pip install -r requirements_nn.txt
./.venv/bin/python train_nnue.py smoke
./.venv/bin/python test_nn_offline.py
./.venv/bin/python train_nnue.py train --shards pilot_shards --out pilot_net_tuned.npz \
    --epochs 120 --batch-size 256 --learning-rate 0.003 --seed 0 \
    --scale 400 --lambda 1.0 --loss mse
./.venv/bin/python check_export_parity.py pilot_net_tuned.npz --shards pilot_shards
./.venv/bin/python eval_report.py pilot_net_tuned.npz --shards pilot_shards
```
