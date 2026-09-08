# Lichess pilot data — handoff to the NN trainer and their LLM

Read this before training. This document accompanies `lichess_pilot_v1.zip`
and the matching `AI_Chessathon_NN_foundation_updated.zip` code package.
It supplements the code package's `README_NN.md`; it does not replace it.

## 1. What this is for

We are building a small neural position evaluator for a from-scratch alpha-beta
chess engine using numba bitboards. The network replaces the hand-written
`evaluate()` function. It receives a position and returns a centipawn score
from the side-to-move's perspective: positive is good for the mover.

The sender's job is to collect and encode training examples. Your job, as the
trainer and assisting LLM, is to verify those examples, train a pilot network,
check exported inference, and integrate it into the existing engine.

**This is a pipeline pilot, not the final training dataset.** Success means
the data loads, training learns, exported predictions agree, and the integrated
engine works. Beating the hand-crafted engine is not required of this pilot.

The eventual shipping gate is real-game strength against the preserved
hand-crafted engine at the team's specified competition conditions:
120 seconds + 0.5 seconds per move, one CPU core. Validation loss alone does
not justify shipping. Keep the known-good engine available throughout.

## 2. Status and provenance: do not assume completion

This README describes the preparation workflow given to the sender. It does
not certify that the sender ran it successfully. Inspect the received files
and `manifest.json` for actual counts and results; do not invent them.

The intended workflow is:

1. Stream the beginning of Lichess's official evaluation export.
2. Save at most 200,000 complete raw records, stopping earlier if approximately
   512 MiB of uncompressed text has been written. The last complete record can
   take the file slightly above that threshold.
3. Save this local source as `pilot_raw.jsonl`.
4. Run the following command with the matching updated code:

```bash
python train_nnue.py prep --db pilot_raw.jsonl --out pilot_shards --limit 100000 --min-depth 16 --shard-size 25000
```

The raw source is a prefix of the export, not a random sample of the entire
database. The prepared pilot keeps the first accepted examples from that prefix.
Do not describe it as representative, balanced, or uniformly sampled.

The expected maximum is 100,000 accepted positions, but filtering can leave
fewer. Read `kept` in the manifest and independently sum shard row counts.
`stop_reason=limit` means the accepted-row limit was reached; `eof` means the
local raw sample ended. Both are normal.

Official source and format: [Lichess evaluation database](https://database.lichess.org/#evals).
The export contains evaluations of varying depth and Stockfish flavours, not
the results of completed games. The local manifest's source size/path/mtime
refer to `pilot_raw.jsonl`, not the complete remote export.

## 3. Files you should receive

| Item | Purpose |
|---|---|
| `lichess_pilot_v1.zip` | Prepared pilot data; normally extracts to `pilot_shards/` |
| `AI_Chessathon_NN_foundation_updated.zip` | Matching encoder, model, runtime, prep/training code, tests, requirements and main README |
| `README_PILOT_HANDOFF.md` | This handoff document |

Inside `pilot_shards/`, expect:

- `shard_00000.npz` and any subsequent numbered shards;
- `manifest.json`, recording preparation settings and actual counts;
- `audit_sample.json`, containing a small sample of source FENs, selected
  White-relative scores, converted mover-relative labels and active indices.

The sender does not need to send their `.venv` or the much larger raw JSONL
file. Ask for the raw sample only if reproducing or debugging preparation
requires it. Keep the received pilot unchanged during experiments.

## 4. Exact data contract

Use the updated code package. Earlier versions expected fewer shard fields.
Do not mix old and new preparation/training scripts.

Each shard contains these arrays:

| Array | Shape | Type | Meaning |
|---|---|---|---|
| `idx` | `(N, 40)` | `int16` | Active feature indices; unused entries are `-1` |
| `cnt` | `(N,)` | `int16` | Number of active indices in each row |
| `y` | `(N,)` | `float32` | Mover-relative centipawns, clamped to ±6000 |
| `wdl` | `(N,)` | `float32` | Actual completed-game result, or `-1` when unknown |
| `key` | `(N,)` | `uint64` | Stable hash of sorted mover-relative active indices |
| `group` | `(N,)` | `uint64` | Reserved game-group key; zero for this pilot |

Every pilot row must have `wdl=-1` and `group=0`.

Only `idx[row, :cnt[row]]` contains features. Never use the `-1` padding as an
index: NumPy/PyTorch would interpret it as the final feature.

The nominal 40-slot buffer is capacity, not the number of active features.
The supplied `_dense_batch()` expands these sparse rows into 780 binary
features on the selected training device.

### Encoding: preserve this exactly

`nn_encode.py` is the single source of truth. Do not write another encoder.

- 768 piece features: two sides × six piece types × 64 squares.
- Piece index: `own_or_opp * 384 + piece_type * 64 + relative_square`.
- `own_or_opp=0` for the mover, `1` for the opponent.
- Piece types: `0=P, 1=N, 2=B, 3=R, 4=Q, 5=K`.
- For Black to move, flip ranks with `square ^ 56` and swap own/opponent roles.
- Indices 768–771: own kingside/queenside, opponent kingside/queenside rights.
- Indices 772–779: en-passant file a–h, all off when absent.

### Labels: no second sign flip or rescaling

Preparation selects the deepest usable first-PV evaluation meeting the depth
threshold. Lichess scores are treated as White-relative and converted once to
mover-relative scores during prep. Mate labels become decisive ±6000 targets.

`y` already uses the network's required perspective and units. Do not negate
Black-to-move examples again. Do not divide `y` by 100 or 1,000. The training
loss applies its own scale.

`wdl` means a completed-game outcome, not the sign of `y` and not a Stockfish
WDL prediction. No actual game outcomes are available in this pilot.

## 5. Setup and initial checks

Run commands from the folder containing the Python modules. The examples use
`python`; substitute your environment's interpreter if necessary. In a Colab
notebook, shell commands generally need a leading `!`.

1. Extract both ZIPs. Place `pilot_shards/` next to `train_nnue.py`.
2. Read this README and the package's `README_NN.md`.
3. Install the training dependencies:

```bash
python -m pip install -r requirements_nn.txt
```

4. Check whether the training environment actually has GPU support:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

The supplied trainer selects CUDA when available and CPU otherwise. A CPU
smoke run is valid, but measure throughput before beginning long training.
Do not assume the trainer automatically uses an Apple GPU.

5. Inspect the manifest and audit sample. Confirm the intended depth, source,
   row counts, signs and encoding. Run the following in a Python cell or script:

```python
import numpy as np
from train_nnue import _load_shards, split_indices

idx, cnt, y, wdl, keys, groups = _load_shards("pilot_shards")
train_rows, val_rows, test_rows = split_indices(keys)

assert len(y) > 0
assert np.isfinite(y).all()
assert np.all(np.abs(y) <= 6000)
assert np.all(wdl == -1)
assert np.all(groups == 0)
assert len(np.unique(keys)) == len(keys)
assert len(train_rows) and len(val_rows) and len(test_rows)

print("Total:", len(y))
print("Train / validation / test:",
      len(train_rows), len(val_rows), len(test_rows))
```

Check the printed total against `manifest.json`. If a split is empty, resolve
that with a larger pilot or explicit split settings; do not silently merge
held-out data into training.

6. In the complete engine checkout, run:

```bash
python test_nn.py
```

This test imports the existing engine's `bb_numba.py`, which is not included in
the NN ZIP. Place the NN folder in the engine checkout so the import resolves.
Do not fabricate or stub `bb_numba.py` and call that an engine-parity pass.

7. Exercise the synthetic training/export loop:

```bash
python train_nnue.py smoke
```

## 6. Train the first pilot network

Preserve the architecture for this baseline:

```text
780 -> Linear(256) -> clipped ReLU [0,1]
    -> Linear(32)  -> clipped ReLU [0,1]
    -> Linear(1)   -> mover-relative centipawns
```

Start with a short run to check that real pilot data trains and exports:

```bash
python train_nnue.py train --shards pilot_shards --out pilot_net_check.npz --epochs 3 --batch-size 8192 --learning-rate 0.001 --seed 0 --scale 400 --lambda 1.0 --loss mse
```

If GPU memory runs out, reduce batch size to 2048 or 1024. Do not change the
encoding or architecture to address a batch-memory problem.

If the short run behaves sensibly, run the planned pilot baseline:

```bash
python train_nnue.py train --shards pilot_shards --out pilot_net.npz --epochs 30 --batch-size 8192 --learning-rate 0.001 --seed 0 --scale 400 --lambda 1.0 --loss mse
```

This command starts a new model; it does not continue training the three-epoch
checkpoint. Thirty epochs is a starting setting, not a guaranteed optimum.

The loss compares `sigmoid(model_output_cp / 400)` with
`sigmoid(y / 400)`. This is a bounded expected-score proxy, not a calibrated
probability of winning. The runtime still outputs centipawns.

The updated script selects the lowest-validation-RMSE epoch and exports that
state, then evaluates its test split. Record the selected epoch, metrics and
runtime. Test metrics are not for choosing between repeated experiments;
using them that way would consume the holdout.

## 7. Split details and limits

For this eval-only pilot, stable canonical keys assign approximately 98% of
rows to training, 1% to validation and 1% to test. Proportions are approximate.
Do not independently random-split the rows again.

For future game-derived data, the supplied trainer uses non-zero `group` keys
instead of position keys for splitting. This keeps a game's rows together,
but does NOT by itself prevent the same position appearing in two different
games or sources assigned to different splits. Before using mixed data, enforce
cross-source/cross-group position isolation as a separate release check.

The current loader expects the new `key` and `group` fields. Old four-array
shards are not directly compatible: re-prepare from source or deliberately
validate and migrate them. Never fill missing keys with random numbers.

## 8. Validate export and engine integration

The shipped tests primarily exercise random-weight parity. Also compare your
actual trained checkpoint's PyTorch predictions with its exported NumPy/numba
predictions on held-out pilot positions. Check raw floating-point outputs and
the engine-facing integer/clamped output separately.

Do not treat the legal-move demo in `test_nn.py` as a complete alpha-beta
integration test. It selects among legal moves using the evaluator; it does
not establish that the full search has been wired correctly.

After wiring the net into the actual search:

- Confirm legal moves, completed games and correct score perspective.
- Preserve terminal/draw handling in search.
- Check cold-start/JIT time against the team's 90-second initialisation budget.
- Check clock safety, search throughput and completed depth.
- Pass weights into compiled code or load fixed weights before first compile;
  do not assume changing a Python global replaces numba's compiled weights.
- Keep PyTorch out of per-leaf runtime inference.

The current runtime rebuilds the accumulator each call. Incremental evaluation
and quantisation are later tasks, not requirements for accepting this pilot.
With mover-relative inputs, incremental evaluation needs both fixed White and
Black perspectives or another explicitly correct design.

## 9. Verification status: what has and has not been established

The package author reported syntax checks and isolated checks of data handling,
weight loading, numeric formulas and clamping. Some checks substituted missing
dependencies. The actual complete engine, compiled numba/PyTorch parity, CUDA
training and competition-time-control gauntlet were not validated in that
environment.

Therefore run the real dependency-backed tests locally. Do not repeat a claim
that this package or this user's dataset has passed a complete training and
engine gauntlet until you have evidence.

The sender's preparation instructions perform basic shard checks. They do not
include independent Stockfish relabelling, a full distribution audit or proof
that the trained model will be strong.

## 10. What to report back to the sender

Send a concise report with:

1. Exact accepted row count and train/validation/test counts.
2. Whether schema, label signs and audit examples passed inspection.
3. Whether the real tests and synthetic smoke run passed; include exact errors
   for any failure and distinguish missing dependencies from code defects.
4. Device used, training settings, selected epoch and validation/test metrics.
5. Whether the trained checkpoint passed export/runtime parity.
6. Whether actual engine games completed without illegal moves, startup failures
   or time losses.
7. Whether data collection can now scale, and any concrete requested change.

Return `pilot_net.npz` and relevant logs when available. Do not call a pilot
network submission-ready solely because training loss decreases.

Once the pilot works, build a separately versioned, broader dataset: initially
about one million examples, later 5–10 million, with wider source coverage and
eventually positions reached by our engine labelled using a fixed Stockfish
setup. Preserve the pilot for reproducibility.
