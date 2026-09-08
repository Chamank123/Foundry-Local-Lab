# Engine integration report

Reply to `NEXT_STEPS_FOR_CLAUDE.md`. Supersedes the affected parts of
`REPORT.md`; everything not corrected below still stands. Same container as
before: 4 CPU cores, no GPU, torch 2.14.0, numpy 2.4.6, numba 0.67.0,
python-chess 1.11.2, Python 3.11.

**Headline:** the engine snapshot closed every missing test — `test_nn.py` is
green with the real `bb_numba`. Integration is done and both evaluators are
selectable. Along the way integration exposed **a competition-fatal bug that
would have cost every NN game on the clock**, described in section 4. Neither
checkpoint is a candidate: both lose 0–8 to the hand-crafted engine.

---

## 1. Corrections to REPORT.md — accepted

| REPORT.md claim | Status |
|---|---|
| The retune isolates batch size and learning rate | **Wrong.** It also raised total updates from 360 to 45,960 (best epoch 81 = 31,023). Update count is the confound and probably the dominant term. |
| The ±6000 rows "deliver almost no gradient" | **Wrong.** `dL/dz = 2(p−t)·p(1−p)/K` depends on `p−t`, not on `t` alone; a mate-labelled row with the model at `p≈0.5` gives a large gradient. |
| Capping mates nearer ±2000 would help | **Unsupported.** `sigmoid(2000/400)=0.9933` vs `sigmoid(6000/400)=0.9999997` — the target barely moves. Withdrawn as a recommendation. |
| 18.9% of rows are forced mates | **Overstated.** `abs(y)==6000` only means the label clamped; an extreme finite cp clamps identically, and the schema carries no `is_mate` flag. Correct statement: 18.9% of labels are *clamped*. |
| The dataset is "exhausted", remaining issues "structural" | **Overstated.** Withdrawn — see section 3, where an optimisation-only change improved test RMSE by 10% on the same 100k rows. |
| Removing a queen implies a −900 change | **Too crude.** The test included tactical and already-decided positions. Rebuilt in `test_material.py` (section 3). |

Your output-bound calculation reproduces exactly from the checkpoints:

| Checkpoint | Lower bound | Upper bound |
|---|---:|---:|
| `pilot_net.npz` | −7.58780600 | +7.83174381 |
| `pilot_net_tuned.npz` | −1198.48968278 | +1056.71854401 |

## 2. Engine tests — the gap is closed

With the NN modules importable (`PYTHONPATH=engine`), the **real `test_nn.py`,
using the real `bb_numba.board_np`, is ALL GREEN**:

* encoding parity, engine == python-chess, on **307 positions**
* numba == numpy, max diff **5.04e-07 cp**
* numba == PyTorch, max diff **5.96e-07 cp**
* move-selection demo legal on all 306 non-terminal positions
* `nn_eval` 151,959 evals/sec, one core

Nothing was mocked; the board format was not adapted to the encoder.

**Added `test_engine_parity.py`,** because `test_nn.py` builds every position
from a FEN and so hands both encoders identically-parsed state. The engine does
not play that way. It walks **35,451 positions reached through the engine's own
`gen_legal`/`make`**, carrying a python-chess board alongside, and compares the
encoders at every ply. **0 mismatches.**

That test was written for a specific worry, and the numbers show the worry was
real even though the encoders agree:

* the engine held an en-passant square in **1,934** positions;
* in **1,800** of those (93%) **no legal en-passant capture existed**;
* a FEN round trip would have **dropped the square in all 1,800**, so a
  FEN-based test cannot see this path at all.

### A train/serve skew this exposed

`make()` sets `bd[EP]` after *every* double push. The pilot data uses the other
convention: **159 of 100,000 rows (0.16%)** carry an en-passant feature, which
matches the "a capture is actually legal" rate measured in the engine walk
(0.38%), not the "any double push" rate (5.5%). The audit sample agrees — its
single ep FEN had a legal capture available.

So at run time the engine turns on an ep feature roughly **34× more often than
training ever did**, and that feature's weights are trained on 159 examples.
Measured impact on `pilot_net_tuned.npz` over 2,527 affected positions:

| | value |
|---|---|
| mean eval change from that one feature | **+45.6 cp** |
| median | +25.5 cp |
| largest | 701.9 cp |
| positions moved more than 50 cp | 35.2% |

**I have not changed the encoder.** It is the frozen 780-feature contract and
the collector is running against it. This needs your decision — the options are
to make the runtime set the ep feature only when a capture is legal (matching
the data), or to re-prep with the engine's convention. Either way both encoders
must change together and `nn_encode.py` stays the single source of truth.

## 3. Output scale — measured, and it is not the material fix

Saturation on 1,000 held-out positions, as requested:

| Checkpoint | accumulator at 0 / at 1 / interior | hidden at 0 / at 1 / interior | Σ\|out_w\| |
|---|---|---|---|
| `pilot_net.npz` | 28.9 / 0.1 / 71.1% | 39.9 / 33.5 / 26.6% | 15.4 cp |
| `pilot_net_tuned.npz` | 85.4 / 6.5 / **8.0%** | 32.6 / 62.0 / **5.4%** | 2255.2 cp |

The tuned net's hidden layer is **94.6% saturated**. It reaches hundreds of
centipawns by pinning clipped units to their bounds, which is what your
inspection of the starting position (23 of 32 units exactly 1) suggested.

I ran the bounded experiment from section 4B: a fixed, non-trainable multiplier
on the output head, folded into `out_w`/`out_b` at export. The function class,
architecture dimensions, export layout and runtime socket are unchanged, and the
model still emits raw centipawns. `--output-scale` defaults to 1.0, so the
existing commands reproduce exactly. One variable changed; same data, splits,
seed, and update budget.

| | val RMSE | test RMSE | best epoch | output range (held-out) |
|---|---|---|---|---|
| `pilot_net_tuned.npz` (scale 1) | 0.12082 | 0.13068 | 81 | −1198 … +1057 |
| `pilot_net_oscale400.npz` (scale 400) | **0.11580** | **0.11760** | 70 | −2441 … +2874 |

Better loss (test −10%), better sign agreement (90.5% vs 86.8%), better rank
correlation (Spearman 0.800 vs 0.757), and export parity still exact.

**But it did not fix material, and this is the important result.** Rebuilt
controlled test (`test_material.py`): quiet positions only — mover not in check,
no capture available to either side — nominal material within ±200 cp, one piece
removed, the result still legal, non-terminal, quiet, and same side to move.
600 such comparisons:

| Checkpoint | correct direction | slope of \|Δeval\| on true piece value (1.0 = correct scaling) |
|---|---|---|
| `pilot_net.npz` | 56% | 0.005 |
| `pilot_net_tuned.npz` | **67%** | **0.032** |
| `pilot_net_oscale400.npz` | 58% | 0.007 |

The checkpoint with the best validation loss has the **worst** material
behaviour of the two trained nets. Validation RMSE and material understanding
are dissociated here, so neither can be selected on loss alone. Corrected
figures for the tuned net: losing a queen moves it −45 cp (mover) / +75 cp
(opponent), not the −87 cp REPORT.md claimed.

## 4. Integration — and the bug it exposed

`evaluate(bd)` already returned mover-relative integer centipawns, the same
contract as `nn_eval`, so the socket matched. The six weight arrays are
**threaded through the search as an argument tuple** — `evaluate`, `quiescence`,
`negamax`, `search_root` — never read from a module global inside njit code.
`load_nn(path)`, `use_handcrafted()` and `nn_active()` switch evaluators and
rebuild per-game state, because the TT, killers and history hold scores from the
previous evaluator. **The hand-crafted evaluation remains the default**; nothing
about `agent.get_move` changed.

### The bug

`export_weights` writes `acc_w` as `.T`, which is **F-ordered**. `np.savez`
records that order and `np.load` returns it F-ordered. Consequences:

1. numba types an F-ordered array as a *different type* from a C-ordered one, so
   the first NN search triggered a **full recompile of the entire search graph:
   11.1 seconds**.
2. `get_move` runs the search in a worker thread joined with a `budget + 0.5s`
   backstop. The compile blew straight through it, the thread was abandoned with
   no move, and `get_move` fell back to **its first legal move — silently, for
   every move of every game.**
3. It also defeats the design: `nn_forward` does `acc_w[idx[k]]` expecting a
   contiguous row, which the docstring calls out explicitly.

This is exactly the failure mode `README_NN.md` warns about, arriving through
memory layout rather than through globals. It was invisible from outside — legal
moves, no crash, no error — and the first smoke match recorded a **time
forfeit** plus moves like `a2a3` before I traced it.

**Fixed** by forcing C order in `export_weights`, in `load_weights` (so existing
checkpoints are safe without re-export), and in `random_weights`; `load_nn` now
rejects any non-contiguous array with a message naming the consequence, and runs
`_warmup_nn()` so compilation against the real weights is paid in the init
budget. Before and after, on the same four positions:

| | before fix | after fix |
|---|---|---|
| nodes searched | **0** | 37,607 – 96,726 |
| move from the start position | `a2a3` (fallback) | `e2e4` |
| time forfeits in an 8-game match | 1 | 0 |

### Timing on the intended machine

| | value |
|---|---|
| import + JIT warmup, NN build, both evaluators compiled | **27.5 s** (90 s budget) |
| `load_nn` including NN-path warmup | ~1 s |
| worker process start to first move accepted | 28.4 – 29.7 s |
| **full-search NPS, hand-crafted** | 1.28 – 1.63 M nodes/s |
| **full-search NPS, NN** | 75 k – 233 k nodes/s |

The NN search is **7–20× slower per node** than the hand-crafted one. The
156,000 forward passes/sec in REPORT.md is not NPS and should not be compared
with these numbers.

### Cost to the hand-crafted engine — please read before submitting

Threading the tuple through every recursive call is not free. At fixed depth 6,
best of three, **node counts are identical** (23,753 / 123,080 / 66,265 — the
search is behaviourally unchanged), but:

| position | pristine NPS | integrated, hand-crafted path | loss |
|---|---|---|---|
| start | 2,159,280 | 1,629,248 | −25% |
| open game | 2,087,014 | 1,536,899 | −26% |
| midgame | 1,604,846 | 1,276,639 | −20% |

**If the hand-crafted engine is the submission, submit the pristine
`search_numba.py`**, preserved unmodified at commit `eb47019`. It is bit-identical
to Aryan's snapshot. If the NN ever becomes the candidate, the zero-overhead
route is to compile the search inside a factory that closes over the loaded
weights, paying one compile at init instead of per-call argument passing.

## 5. Match results

Paired openings, colours reversed, each side in its **own process** so the two
players get independent transposition tables, killers, history and repetition
lists. **5 s + 0.1 s per side — a pipeline smoke test, not the competition
gate.** I have not been given the competition configuration to verify the
120 s + 0.5 s / one core / 90 s init figures against.

| Candidate | Result | Illegal moves | Flags | Crashes |
|---|---|---|---|---|
| `pilot_net_tuned.npz` | **0 – 8** | 0 | 0 | 0 |
| `pilot_net_oscale400.npz` | **0 – 8** | 0 | 0 | 0 |

Games completed cleanly with correct termination and safe clocks. **The pipeline
is correct; the evaluation is not competitive.** Neither checkpoint is ready for
deployment, and 16 games at a short clock would not establish superiority even
if one had scored — it only rules candidates out.

## 6. What I did not do

* No competition-clock gauntlet — the real configuration is unverified here.
* No encoder change for the ep skew — your call, and the collector is running.
* No incremental accumulator or quantisation.
* No CUDA training — no GPU in this container.
* Only one output-scale value (400) was tried, at one update budget.

## 7. Suggested order from here

1. **Decide the en-passant convention** (section 2). It is cheap now and
   expensive after 5M positions are collected against the current one.
2. **Keep collecting.** Nothing found here argues for changing the collector,
   and the mate-clamp change REPORT.md proposed is withdrawn.
3. **Investigate material directly**, not through validation loss — section 3
   shows they move independently. `test_material.py` is a usable regression
   metric; the slope column is the one to watch.
4. **On the 1M set**, run the scale-1 and scale-400 heads at equal update
   budgets and compare on both validation RMSE *and* the material slope.
5. **Re-run `test_engine_parity.py` and `check_export_parity.py`** against any
   new dataset or checkpoint before another match.

## Files

New since REPORT.md:

| File | What it is |
|---|---|
| `engine/` | Aryan's snapshot; pristine at commit `eb47019`, NN-integrated after |
| `test_engine_parity.py` | Encoder parity over engine make/unmake positions |
| `test_material.py` | Controlled quiet-position material probe |
| `match.py`, `match_worker.py` | Paired-opening match, one process per side |
| `bench_search.py` | Real search NPS and time-budget adherence |
| `pilot_net_oscale400.npz` | Output-scale-400 checkpoint (best loss, worse material) |

Changed: `nn_model.py` (C-contiguity, `out_scale` folding), `train_nnue.py`
(`--output-scale`), `engine/search_numba.py` (threading, `load_nn`).
`nn_encode.py` is **untouched**.

## Reproducing

```bash
cd chess-nn
python -m venv .venv && ./.venv/bin/python -m pip install -r requirements_nn.txt
PYTHONPATH=engine ./.venv/bin/python test_nn.py
PYTHONPATH=engine ./.venv/bin/python test_engine_parity.py
./.venv/bin/python train_nnue.py train --shards pilot_shards --out pilot_net_oscale400.npz \
    --epochs 120 --batch-size 256 --learning-rate 0.003 --seed 0 \
    --scale 400 --lambda 1.0 --loss mse --output-scale 400
./.venv/bin/python check_export_parity.py pilot_net_oscale400.npz --shards pilot_shards
./.venv/bin/python test_material.py pilot_net_tuned.npz pilot_net_oscale400.npz
PYTHONPATH=engine ./.venv/bin/python bench_search.py pilot_net_tuned.npz
PYTHONPATH=engine ./.venv/bin/python match.py --nn pilot_net_tuned.npz --games 8
```
