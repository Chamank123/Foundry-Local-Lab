# Hybrid evaluator experiment — handoff

This folder tests whether the supplied 8-million-position NN can improve the
original handcrafted (HC) evaluator without replacing it. The original engine
is preserved in `engine_pristine/`; the experiment engine is in `engine/` and
still defaults to HC unless `load_nn(...)` is called explicitly.

## Bottom line

The standalone 256x32 NN is not shippable: it loses badly to HC and makes the
search roughly nine times slower by node count at the competition clock. A
small 64x16 residual network was therefore trained to predict only a correction
to HC. At 25% correction strength it was the strongest fixed-depth screen
candidate. Final competition-clock result and recommendation are recorded below.

## What changed

- `engine/search_numba.py`: the original evaluator body is preserved as
  `evaluate_handcrafted`; modes support HC, full NN, blends, and HC plus a
  residual correction. HC remains the default.
- `engine/nn_adapter.py`: adds `legal-ep-v1`, which exposes the en-passant input
  only when the mover actually has a legal en-passant capture. This changes only
  the NN input and never the board or legal move generator.
- `train_residual.py` and `residual_data.py`: train a compact correction network
  against the same 8M teacher positions while retaining HC as the baseline.
- `nn_runtime.py`: refuses to load a residual checkpoint as a standalone value
  net, preventing a silent integration error.
- `verify_snapshot.py`: snapshot-count and split failures now fail the command
  instead of merely printing a warning.
- `test_hybrid.py`, `assess_hybrid.py`, `screen_hybrids.py`,
  `experiment_matches.py`, and `hybrid_worker.py`: reproducible verification and
  game-test tools.

## Training result

The residual was trained for three CPU epochs with batch size 256, Adam at
0.003, cosine decay, seed 7, and architecture 780→64→16→1. Its output is added
to the original HC centipawn score inside the squashed probability loss.

| Metric | HC alone | HC + full residual |
|---|---:|---:|
| Validation RMSE | 0.18670 | 0.14310 |
| Held-out test RMSE | 0.18833 | 0.14510 |

This is a bounded experiment, not evidence that the three-epoch model converged.
All 13,378 training rows carrying an EP feature were audited and had a legal EP
capture. The supplied data therefore used legal-capture EP semantics, while the
original runtime encoded any stored EP target square.

## Verification

- All 8,000,000 rows and 32 shards passed snapshot checks; split counts were
  7,840,256 train, 79,698 validation, and 80,046 held-out test.
- All 200 source audit FENs re-encoded with the expected feature set, key, and
  mover-relative sign.
- On 2,000 validation positions, PyTorch, NumPy, and numba residual outputs
  differed by at most 0.000641 cp. The final combined integer score had zero
  mismatches.
- The actual original HC evaluator and the refactored HC function produced
  identical scores in the hybrid test suite.
- Legal, irrelevant, and pinned en-passant cases passed against python-chess;
  the board was unchanged by encoding.
- The full foundation suite passed: 307-position engine/Python encoding parity,
  NumPy/numba/PyTorch forward parity, preparation/resume checks, and legal move
  selection on 306 non-terminal positions.

## Equal-depth screen

Each candidate played eight games at depth 4 against HC over four paired
openings, with colours reversed. These are selection diagnostics, not
competition-clock evidence.

| Candidate | Score / 8 | Issues |
|---|---:|---:|
| Standalone full NN, legacy EP | 0.0 | 0 |
| Standalone full NN, legal EP | 1.0 | 0 |
| 10% full-NN blend | 6.0 | 0 |
| 25% full-NN blend | 6.5 | 0 |
| HC + 100% residual | 0.5 | 0 |
| HC + 25% residual | 7.0 | 0 |

The result shows that small corrections can improve choices at equal depth, but
does not account for the cost of reaching that depth.

## Competition-clock check

Fresh processes, one pinned CPU, 120 seconds + 0.5 seconds/move, automatic
threefold/50-move draws, and a 600-ply cap were used. These ran on the available
local machine, not the judge's AMD EPYC, so startup and speed must be rechecked
in the official environment.

| Candidate | Paired score | Candidate nodes | HC nodes | Operational issues |
|---|---:|---:|---:|---:|
| 25% full-NN blend | 0.0 / 2 | 52,563,434 | 437,174,450 | 0 |
| HC + 25% residual | 2.5 / 4 | 466,844,257 | 1,143,709,557 | 0 |

The full-NN blend was stopped after the completed paired opening because it lost
with both colours. It searched about one ninth as many nodes as HC. Both games
were valid; the stopped extra games are not counted.

The residual candidate scored one win and three draws across two paired
openings, unbeaten over 699 plies. It searched 40.8% as many nodes as HC (about
2.45x slower by node rate). Candidate startup was 27.6–30.3 seconds and peak RSS
was 498.3 MB locally. The draws were one 50-move, one threefold, and one
insufficient-material result; the win was checkmate. No game reached the ply
cap.

The clean-extracted candidate ZIP, with the checkpoint automatically loaded by
a ZIP-relative path, imported in 31.02 seconds locally, reported evaluator mode
3250 (HC plus 25% residual), and returned a legal move from the starting position
in 37.46 seconds total. This is inside the 90-second initialisation budget
locally but still requires an official-image check.

## Recommendation

Promote the 25% residual candidate to a larger official-machine gauntlet; do not
declare it the submission winner from four games. A 2.5/4 score is encouraging
but statistically weak, and the candidate searches roughly 2.45x fewer nodes.
Until it wins a substantially larger paired-opening gauntlet on the actual
competition machine, submit `engine_pristine/`. Keep the residual candidate as a
separate challenger so reverting requires no code surgery.

## Reproduction

From this directory with Python 3.12, numpy, numba, torch, and python-chess:

```bash
python3 verify_snapshot.py ../data/larger_shards
python3 test_hybrid.py
python3 assess_hybrid.py ../data/larger_shards residual_64x16.npz
```

Retraining the supplied residual experiment from scratch:

```bash
python3 train_residual.py \
  --shards ../data/larger_shards \
  --out residual_64x16_new.npz \
  --epochs 3 --batch 256 --lr 0.003 \
  --h0 64 --h1 16 --threads 2 --seed 7
```

Do not pass `residual_64x16.npz` to `nn_runtime.load`; it is a correction, not a
complete evaluator. To reproduce the tested 25% residual mode in the experiment
engine, call this before the first clocked move:

```python
import search_numba
search_numba.load_nn("residual_64x16.npz", weight=0.25, ep_policy="legal")
```

Calling `load_nn` changes only the experiment engine process. With no call, or
when using the files in `engine_pristine/`, behavior stays on the locked HC
baseline.
