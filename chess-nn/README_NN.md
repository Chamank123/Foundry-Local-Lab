# NN foundation — AI Chessathon

> Brief for the trainer, the two data owners, and any AI assistants helping them.
> Read this from top to bottom before changing the model, encoding, data pipeline,
> or engine integration.

## 1. What we are building

The engine is a from-scratch alpha-beta chess engine using numba-JIT bitboards.
Its position judgement is currently a hand-written `evaluate()` that returns a
centipawn score from the side-to-move's perspective.

We are training a small NNUE-style value network to replace that function and
nothing else:

```text
position -> NN evaluation -> mover-relative centipawn score
```

The decision rule is simple:

> **The network ships only if it beats the locked hand-crafted engine in real
> games at the competition time control: 120s + 0.5s/move, one CPU core.**

Validation loss is useful for selecting checkpoints and detecting bugs. It is
not the final measure of engine strength. If the network does not win the game
gauntlet convincingly, submit the hand-crafted engine.

Before any NN work, preserve a known-good, tested copy of the hand-crafted
engine as the submission safety net.

## 2. Files and ownership

| File | Role |
|---|---|
| `nn_encode.py` | **The encoding contract.** Converts either the engine board or a `chess.Board` into the same 780 features. Never create a second encoder. |
| `nn_model.py` | PyTorch model, architecture constants, weight export and load. |
| `nn_runtime.py` | numba forward pass returning centipawns. The first version rebuilds the accumulator each call. |
| `train_nnue.py` | `smoke`, `prep`, and `train` commands. Streams evaluation data, writes audited/resumable shards, assigns stable splits, and exports the best checkpoint. |
| `test_nn.py` | Encoding parity, PyTorch/NumPy/numba forward parity, engine use, and speed checks. Keep green. |
| `requirements_nn.txt` | Python packages required for preparation, training, and the full test suite. |

Install the dependencies in the trainer/data environment with:

```bash
python3 -m pip install -r requirements_nn.txt
```

`test_nn.py` also imports the engine's existing `bb_numba.py`, so run it from the
NN folder in the complete engine checkout rather than from this ZIP in isolation.

The trainer owns `nn_model.py`, `nn_runtime.py`, engine wiring, checkpoint
selection, and the gauntlet. The data owners own acquisition, preparation,
auditing, splits, manifests, and targeted Stockfish-labelled data.

## 3. Inputs: 780 frozen features

The network receives 768 piece-placement features and 12 state features, all
relative to the side to move.

```text
piece index = own_or_opp * 384 + piece_type * 64 + relative_square
```

- `own_or_opp`: 0 for the mover's pieces, 1 for the opponent's pieces.
- `piece_type`: 0=P, 1=N, 2=B, 3=R, 4=Q, 5=K.
- If White moves, use the board unchanged.
- If Black moves, map each square with `square ^ 56` and swap the colours.
- Files are unchanged by this rank flip.

The final 12 features start at index 768:

| Index | Meaning |
|---:|---|
| 768 | Mover has kingside castling rights |
| 769 | Mover has queenside castling rights |
| 770 | Opponent has kingside castling rights |
| 771 | Opponent has queenside castling rights |
| 772–779 | En-passant file a–h; all zero if unavailable |

The output is also mover-relative: positive is good for the player whose turn it
is. It remains in centipawns so it can replace `evaluate()` without changing the
search's score convention.

Repetition and the fifty-move rule belong in the search. They cannot be inferred
reliably from these 780 features.

### Encoding rule

`nn_encode.py` is the only authority. Data preparation, tests, training, and
runtime integration must not reimplement its logic. If the two encoder paths in
`test_nn.py` disagree, stop and fix encoding before doing any training.

## 4. Architecture: frozen first baseline

```text
780 -> Linear(256) -> clipped ReLU [0,1]
    -> Linear(32)  -> clipped ReLU [0,1]
    -> Linear(1)   -> centipawns
```

Keep the accumulator width at 256 for the first serious training and gauntlet.
Only try 128 if profiling on the competition machine shows that the 256-wide
network loses more strength through reduced search depth than it gains through
evaluation quality.

The shallow, clipped architecture is deliberate: sparse inputs make the first
layer cheap, and bounded activations make later integer quantisation safer.

## 5. Training target

Do not regress raw centipawns directly. Train through a bounded logistic
transformation:

```text
pred_score = sigmoid(model_output_cp / K)
eval_score = sigmoid(target_cp / K)

if an actual completed-game result is known:
    target_score = lambda * eval_score + (1 - lambda) * game_result
else:
    target_score = eval_score

loss = MSE(pred_score, target_score)
```

where:

- `K` defaults to 400 and is a tunable scale, not a law of chess.
- `target_cp` is clamped to ±6000 after conversion to mover perspective.
- Certain mates map to +6000 or -6000 according to the mating side.
- `game_result` is 1 for a mover win, 0.5 for a draw, and 0 for a mover loss.
- `lambda=1.0` means pure evaluation training.
- `lambda=0.7` is a sensible later experiment when genuine game results exist.

`sigmoid(cp/K)` is a **bounded expected-score proxy**, not a literally calibrated
probability of winning. Its purpose is to focus learning on chess-relevant score
differences and prevent extreme evaluations from dominating the loss.

The `wdl` field means the result of an actual completed game from the mover's
perspective. A Stockfish WDL prediction is not an actual result and must not be
silently stored under the same meaning. When no completed-game result is known,
write `WDL_UNKNOWN = -1` and train that row as pure evaluation data.

## 6. Data-source strategy

### Stage A: Lichess evaluations — best starting source

Start with `lichess_db_eval.jsonl.zst`. It provides hundreds of millions of
FENs already evaluated by Stockfish-family engines, so it avoids spending weeks
generating the first set of labels locally. It is the quickest route to a useful
baseline.

It is not a perfect final dataset:

- evaluations come from different analysis depths and Stockfish flavours;
- positions reflect what users chose to analyse, not necessarily what this
  engine reaches at search leaves;
- the export contains no completed-game result for WDL blending;
- reported depth is a useful quality signal, not a guarantee of uniform search.

For each JSON record:

1. Parse the FEN with `python-chess`.
2. Reject malformed, unsupported, or terminal positions that the static
   evaluator will never be asked to score.
3. Select the evaluation with the highest reported depth.
4. Take the first PV's score; do not average candidate-move scores.
5. Convert the Lichess/White-relative score to mover-relative score exactly once.
6. Map mates to ±6000, then clamp all targets to ±6000.
7. Encode only through `nn_encode.py`.
8. Store `wdl=-1` because the evaluation export has no game outcome.

Before scaling, unit-test score perspective using several obvious winning and
losing FENs, including both sides to move and both mate signs. A sign reversal
can produce apparently normal training loss while destroying playing strength.

### Stage B: positions from our own engine — best targeted improvement

After the Lichess pipeline works:

1. Run games and representative searches from varied and competition-relevant
   openings.
2. Sample positions where the current static evaluator is actually called,
   especially suitable quiescence leaves.
3. Cap samples per game/search and deduplicate so adjacent positions do not
   dominate.
4. Label them with one fixed local Stockfish version and one fixed node budget.
5. Record the exact Stockfish binary/version, threads, hash, node limit, and
   options in the dataset manifest.
6. Encode them with the same `nn_encode.py` and shard schema.
7. If the position belongs to a completed game, add the mover-relative result;
   otherwise keep `wdl=-1`.

An initial experiment can mix roughly 80% Lichess data and 20% targeted data.
That ratio is only a starting hypothesis; compare it against Lichess-only in the
gauntlet.

### Optional Stage C: game PGNs

Lichess game PGNs provide real outcomes, but human results are noisy labels and
the positions do not automatically have centipawn evaluations. Use PGN data
only if positions are matched to reliable evaluations or labelled with the
fixed local Stockfish setup. Treat WDL blending as an experiment, not an
automatic improvement.

## 7. Shard format

Never store dense 780-element arrays on disk. Store active feature indices:

```text
shard_NNNNN.npz
    idx  (N, 40) int16     active feature indices, -1 padded
    cnt  (N,)    int16     number of active features
    y    (N,)    float32   mover-relative centipawns, clamped ±6000
    wdl  (N,)    float32   actual mover-relative result or -1
    key  (N,)    uint64    canonical mover-relative position key
    group (N,)   uint64    completed-game group key, or 0 if unknown
```

There are at most 32 occupied-square features plus castling and en-passant
state, so 40 slots are sufficient. Sparse storage is roughly 110 bytes per
position; ten million rows remain around 1.1 GB before filesystem
and archive overhead. Expand each batch to a dense tensor only on the training
device.

Every dataset release must also contain a manifest recording:

- dataset version and creation date;
- source file name and checksum;
- source snapshot/date where known;
- code commit and encoder version;
- filtering, depth, sampling, clamping, and split settings;
- retained/rejected counts and rejection reasons;
- train/validation/test row counts;
- shard names and checksums;
- Stockfish settings for locally labelled rows.

## 8. Pilot-first preparation workflow

### Step 1: lock and test the baseline

```bash
python3 test_nn.py
python3 train_nnue.py smoke
```

Preserve the known-good hand-crafted engine before making integration changes.

### Step 2: create a 100,000-position pilot

```bash
python3 train_nnue.py prep \
  --db lichess_db_eval.jsonl.zst \
  --out shards_pilot/ \
  --limit 100000 \
  --min-depth 16
```

Depth 16 is an initial quality/yield experiment. Compare it with the current
depth-12 default if collection becomes too slow or rejects too much data. Do not
choose a threshold by assumption: record retained counts and compare downstream
results.

Streaming means the database is decompressed as it is read, with no permanent
uncompressed copy. If `--db` expects a local file, the compressed export still
has to be downloaded once.

For an integration-only pilot, taking the first accepted rows is adequate. For
the serious dataset, do not assume the beginning of the export is representative.
Use a documented reproducible sampling method across a broad scan and inspect
the resulting distribution.

### Step 3: audit the pilot

Before creating millions of rows, verify:

| Audit | Requirement |
|---|---|
| Perspective | Positive labels always favour the mover |
| Units | `y` is centipawns, not pawns, probability, or cp/1000 |
| Encoding | Runtime and data-prep encoders match exactly |
| Sparse indices | Unique, in 0–779, and consistent with `cnt` |
| State | Castling and en-passant survive perspective conversion |
| WDL | Eval-only rows contain `-1` |
| Legality | Malformed/unsupported positions are rejected and counted |
| Coverage | Both movers, game phases, material counts, and score bands appear |
| Extremes | Mate and tablebase-like scores are signed and clamped correctly |

`prep` automatically writes `audit_sample.json` containing original FEN,
selected source evaluation, converted target, canonical key, and active feature
indices. Spot-check 100–1,000 rows with a
fixed local Stockfish build. Exact scores can differ with version and search
budget; reversed signs or systematic unit errors cannot.

### Step 4: freeze leakage-safe splits

Use a deterministic split such as:

- 98% training;
- 1% validation for checkpoint/configuration selection;
- 1% untouched test data.

Assign splits using a stable hash of the canonical mover-relative position and
state. Do not use Python's process-randomised built-in `hash()`. Canonically
identical positions must always enter the same split, even if their original
FEN strings differ.

For game-derived data, give every position from one game the same non-zero
`group` key; training uses that key instead of the position key for split
assignment. Also enforce exact-position deduplication. Maintain a second, competition-
relevant test set built from held-out openings/searches. Neither numerical test
set replaces the real-game gauntlet.

`train_nnue.py` implements this split from each shard's canonical `key` array.
The default validation and test allocations are each 10 per thousand (1%).

### Step 5: train and integrate the pilot

```bash
python3 train_nnue.py train \
  --shards shards_pilot/ \
  --out pilot_net.npz \
  --epochs 30 \
  --batch-size 8192 \
  --learning-rate 0.001 \
  --seed 0 \
  --scale 400 \
  --lambda 1.0 \
  --loss mse
```

Confirm that:

- loss falls and validation behaviour is sensible;
- obvious material advantages receive the correct sign;
- exported weights reproduce PyTorch predictions in NumPy and numba;
- the wired engine initialises, returns legal moves, respects its clock, and
  completes games.

The pilot is an end-to-end systems test, not a strength verdict.

### Step 6: scale in stages

1. Prepare approximately one million audited positions.
2. Train a first serious network while preparation continues.
3. If the pipeline remains clean, scale to roughly 5–10 million positions.
4. Compare score/game-phase distributions rather than assuming more rows are
   automatically better.
5. Version every dataset and never overwrite a dataset used for a reported run.

To extend a deliberately stopped run using the identical source and prep
configuration, increase `--limit` and add `--resume`:

```bash
python3 train_nnue.py prep \
  --db lichess_db_eval.jsonl.zst \
  --out shards_pilot/ \
  --limit 1000000 \
  --min-depth 16 \
  --resume
```

The manifest guards the source identity and preparation settings, validates
existing shard row counts, reloads canonical keys for deduplication, and resumes
after the exact recorded source line. Because `.zst` streaming restarts at the
beginning, resuming is correct but may spend time replaying/skipping earlier
records before producing new rows.

For a reproducible hash sample, use `--sample-modulus M --sample-remainder R`.
For example, modulus 40 and remainder 0 retains approximately one fortieth of a
full scan. Do not combine this with a small early `--limit` and then describe it
as a uniform sample of the entire export: the reader must reach EOF for that
claim.

The Lichess export is already described as FEN-unique, but still deduplicate
after canonical perspective conversion and whenever sources are combined.

### Parallel preparation safety

Do not let two processes invent `shard_00000.npz` in the same folder. Skipping
an existing name is not sufficient protection against simultaneous writes.
`prep` creates an exclusive `.prep.lock` and refuses a second writer, but workers
should still use separate output directories.

Safe options are:

- separate worker directories followed by a deterministic merge, deduplication,
  split assignment, and manifest generation; or
- explicitly disjoint worker/range prefixes implemented and tested by the prep
  script, with atomic temporary-file rename on completed shards.

Shard and manifest writes use same-directory temporary files followed by atomic
replacement. Resume is explicit and checked by `test_nn.py`. If a process is
killed so abruptly that `.prep.lock` remains, verify that no prep worker is still
running before removing that one lock file. A monolithic
`.zst` file still does not provide cheap arbitrary record-range starts, so use
separate worker directories and verified sampling remainders rather than
inventing byte ranges.

## 9. Runtime and engine wiring

The working non-incremental path is:

```text
nn_runtime.nn_forward(
    idx, cnt,
    acc_w, acc_b,
    l1_w, l1_b,
    out_w, out_b
) -> centipawns
```

Weights are arguments deliberately. numba can freeze the contents of module-
global arrays at first compilation, silently retaining stale weights.

Two integration choices remain:

1. **Weights passed:** thread arrays from `get_move` through root search,
   `negamax`, and `quiescence`. Cleanest for comparing checkpoints.
2. **Fixed import swap:** load the final weights before warm-up/first compile
   and restart the process whenever weights change. Fewer shipped-code edits,
   but only safe for a fixed network.

Never create a PyTorch tensor per search leaf. Search inference is CPU, low
latency, and unbatched. PyTorch is for training and offline checks.

Re-profile on the actual competition machine. The current non-incremental
measurement of roughly 130k evaluations/s versus roughly 450k hand-crafted NPS
is development evidence, not a guaranteed competition-machine result. Record
initialisation time, evaluations/s, search NPS, average completed depth, and
clock margin.

## 10. Incremental accumulator: later, with two perspectives

Incremental evaluation is the main possible speed recovery, but implement it
only after a correct non-incremental network has been trained, integrated, and
tested in games.

One important consequence of the frozen encoding is that side-to-move changes
every ply. The active mover-relative representation therefore appears to swap
almost every piece feature after each move. A single mover-relative accumulator
cannot be updated using only the moved piece's few feature changes.

The intended solution is to maintain **two fixed-perspective accumulators**:

- one representing the position from White's perspective;
- one representing the position from Black's perspective.

Update both accumulators on make/unmake using only the affected piece, castling,
promotion, capture, and en-passant features. At evaluation time, select the
accumulator corresponding to the side to move. Refresh an accumulator from
scratch when its validity is uncertain, and test every special move.

Incremental parity tests must cover:

- quiet moves and captures;
- promotions and promotion captures;
- castling and loss of castling rights;
- en-passant creation, expiry, and capture;
- make/unmake restoration;
- random legal sequences against full recomputation.

Do not ship incremental evaluation merely because it is faster. It must be
prediction-identical, or within the explicitly accepted quantisation tolerance,
to full recomputation.

## 11. Quantisation

Quantisation is a separate, gated speed experiment:

1. Save a float checkpoint and its held-out predictions.
2. Export quantised weights with explicit scales and accumulator widths.
3. Compare float PyTorch, float NumPy/numba, and quantised numba on a large
   held-out set.
4. Check saturation/overflow and worst-case rather than only mean error.
5. Re-run engine correctness, clock, speed, and game tests.

Clipped ReLU helps bound activations, but it does not make scale selection or
integer overflow automatically correct.

## 12. Gauntlet

The gauntlet owns the final decision.

- Use the competition time control and one CPU core per engine.
- Use the same opening with colours reversed.
- Include varied, suitably unbalanced openings; starting-position-only matches
  can produce too many draws.
- Keep opening suites and random seeds fixed across candidate comparisons.
- Log wins, draws, losses, crashes, illegal moves, initialisation failures,
  time losses, NPS, and completed depth.
- Treat any startup or clock regression as a blocker.
- Use short screening matches to eliminate poor configurations, then a much
  larger confirmation match for the apparent winner.
- Report uncertainty; a narrow lead over a few dozen games is not reliable.

Compare at minimum:

1. locked hand-crafted baseline;
2. Lichess-only network;
3. Lichess plus targeted engine-position network;
4. optional WDL-blended network;
5. any quantised/incremental build proposed for submission.

Ship the best **reliable real-game engine**, not the checkpoint with the lowest
validation loss.

## 13. Team workflow

### Trainer

- Runs smoke training immediately on the 100k pilot.
- Owns model configuration, training, checkpoint export, runtime wiring, and
  the gauntlet.
- Can race a small number of purposeful configurations across Colab/Kaggle.
- Does not change encoding or dataset semantics locally without coordinating
  and versioning the change.

### Data owner A: Lichess pipeline

- Acquires/verifies the compressed evaluation export.
- Produces the pilot, then 1M and 5–10M versioned releases.
- Owns reproducible sampling, depth filters, sparse shards, and manifests.
- Writes into a dedicated worker directory, never a shared shard namespace.

### Data owner B: QA and targeted data

- Independently audits signs, units, encodings, filters, distributions, and
  train/validation/test isolation.
- Owns fixed Stockfish spot checks and the targeted leaf-position pipeline.
- Builds the held-out competition-relevant evaluation set.
- Adds completed-game WDL only when provenance and mover perspective are known.

### Explicit owners

- **Gauntlet and engine wiring:** trainer.
- **Locked safety-net submission:** nominate one named team member before NN
  integration begins.
- **Dataset release approval:** both data owners sign off on manifest and audit.

This structure allows all three people to work from day one: the trainer proves
the loop on pilot data, one data owner scales Lichess, and the other audits and
builds the targeted stream.

## 14. Stop conditions and gotchas

- **Encoding drift:** stop immediately if encoder parity fails.
- **Wrong perspective:** unit-test both colours and mate signs before scaling.
- **Unit drift:** shards store centipawns; logistic conversion happens in loss.
- **Data leakage:** canonical duplicates and positions from one game stay in one
  split.
- **Ambiguous WDL:** unknown is `-1`; predicted WDL is not a completed result.
- **Concurrent shard collisions:** use isolated worker directories or proven
  atomic disjoint prefixes.
- **Unrepresentative limit:** the first N accepted records are not automatically
  a sound serious training sample.
- **Resume cost:** correctness is preserved, but the compressed stream is replayed
  from the beginning to reach the recorded line.
- **Tensor per leaf:** forbidden in search.
- **Stale numba globals:** pass weights, or load fixed weights before compile and
  restart to change them.
- **Incremental perspective bug:** maintain/test both fixed perspectives.
- **Quantisation scale bug:** compare against float predictions before games.
- **Over-fitting validation:** the gauntlet decides.
- **Initialisation budget:** measure a clean-unzip cold start after every runtime
  change and stay comfortably inside 90 seconds.
- **Clock safety:** a stronger evaluator that flags games does not ship.

## 15. Immediate checklist

- [ ] Lock and archive the hand-crafted baseline.
- [ ] Assign the safety-net owner.
- [ ] Run `test_nn.py` and `train_nnue.py smoke`.
- [ ] Confirm the competition allows externally labelled training data.
- [ ] Create the 100k Lichess pilot.
- [ ] Audit signs, units, state features, sparsity, legality, and coverage.
- [ ] Implement/verify stable canonical train/validation/test splitting.
- [ ] Train, export, wire, and game-test the pilot.
- [ ] Release the 1M dataset with a manifest.
- [ ] Begin targeted leaf collection and benchmark Stockfish labelling rate.
- [ ] Scale to 5–10M only after the pipeline passes audit.
- [ ] Compare Lichess-only and mixed-data nets in paired-opening games.
- [ ] Attempt incremental evaluation and quantisation only behind parity tests.
- [ ] Run the final real-time-control gauntlet and submit its reliable winner.

## 16. References

- Lichess open evaluation database and schema:
  <https://database.lichess.org/#evals>
- Stockfish explanation of evaluation and WDL interpretation:
  <https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html#interpretation-of-the-stockfish-evaluation>
- Stockfish NNUE principles, sparse inputs, accumulators, quantisation, and
  multiple perspectives:
  <https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html>
