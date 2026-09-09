# Final report — pilot through 8M dataset

Answers the seven items in `NEXT_STEPS_FOR_CLAUDE.md` section 7. Full detail
lives in `REPORT.md` (pilot), `REPORT2.md` (engine integration) and `REPORT3.md`
(8M dataset, sections 9–10 for the later rounds). Where those disagree with this
document, this document is current.

Environment for every number below: Linux container, **4 CPU cores, no GPU**,
torch 2.14.0, numpy 2.4.6, numba 0.67.0, python-chess 1.11.2, Python 3.11.

---

## 1. Engine snapshot and training-code changes

**Engine snapshot:** `agent_1.zip` as supplied (`agent.py`, `search_numba.py`,
`bb_numba.py`, `bb_core.py`). No commit id was included. Preserved **unmodified**
at git commit `eb47019` so the integration diff is reviewable against it.

**Changes to the NN package** (`nn_encode.py` is **untouched** throughout):

| File | Change | Why |
|---|---|---|
| `nn_model.py` | Force `acc_w` C-contiguous in `export_weights`, `load_weights`, `random_weights` | Fixes the on-clock recompile bug, section 2 |
| `nn_model.py` | `build_model(out_scale, h0, h1)`; `out_scale` folded into `out_w`/`out_b` at export | Output-scale experiment; configurable width |
| `train_nnue.py` | `--lr-schedule {none,cosine}`, `--output-scale`, `--h0`, `--h1`; richer checkpoint metadata | All default to previous behaviour |
| `engine/search_numba.py` | Six weight arrays threaded through `evaluate`/`quiescence`/`negamax`/`search_root`; `load_nn`/`use_handcrafted`/`nn_active`; `_warmup_nn` | NN integration, hand-crafted stays the default |
| `check_export_parity.py` | Infer `h0`/`h1` from the checkpoint | Latent bug exposed by `--h0/--h1` |

All defaults preserve the original behaviour: the handoff's baseline training
command and `agent.get_move` are unchanged.

## 2. Real engine encoding and parity tests

**Commands and actual results** (all run, none mocked):

| Test | Command | Result |
|---|---|---|
| Shipped engine test | `PYTHONPATH=engine python test_nn.py` | **ALL GREEN** — encoding parity engine == python-chess on **307** positions; numba == numpy **5.04e-07 cp**; numba == PyTorch **5.96e-07 cp**; 306 legal move selections |
| make/unmake parity | `PYTHONPATH=engine python test_engine_parity.py` | **0 mismatches** over **35,451** positions reached through the engine's own `gen_legal`/`make` |
| Trained-checkpoint parity | `python check_export_parity.py <ckpt> --shards overnight_shards_8m` | **PASS** for every checkpoint; raw float ≤3.2e-03 cp; clamped int 0–1 of 1000 rows differing by 1 cp (truncation, not a defect) |
| Snapshot schema | `python verify_snapshot.py overnight_shards_8m` | **PASSED** — see section 4 |
| Offline subset | `python test_nn_offline.py` | PASS (kept for checkouts without the engine) |

**Gaps:** none remaining in encoder/runtime parity. `test_engine_parity.py` was
added because `test_nn.py` builds every position from a FEN, so both encoders
receive identically-parsed state; the engine instead reaches positions via
`make()`, which sets `bd[EP]` after every double push while `Board.fen()` drops
the square when no capture is legal. 1,800 of 1,934 ep-holding positions took
that path, invisible to any FEN-based test. The encoders agree on all of them.

**Open issue — en-passant train/serve skew.** The engine sets an ep feature on
~5.5% of positions; the data carries one on **0.167%** (13,378 of 8,000,000),
matching the "capture actually legal" convention. That feature, trained on few
examples, moves the eval a median 25.5 cp and up to 702 cp. **The encoder was
not changed** — it is the frozen contract and the collector ran against it.
This needs your decision.

## 3. Output-scale investigation

**Measured:** the pilot's hidden layer was **94.6% saturated** (32.6% at 0,
62.0% at 1, 5.4% interior) with the accumulator 85.4% at zero — it reached
magnitude by pinning clipped units. Output bounds reproduce your figures exactly
(`pilot_net.npz` −7.58780600 … +7.83174381).

**Changed:** a fixed, non-trainable multiplier on the output head, folded into
`out_w`/`out_b` at export. Function class, export layout and runtime socket
unchanged; the model still emits raw centipawns; `--output-scale` defaults to 1.0.

**Evidence:** at pilot scale it improved loss but *worsened* material (58% vs
67% correct direction) — which is why no candidate was chosen on loss alone. At
8M that contradiction vanished and it wins on both axes. Net verdict: a small,
real win, and evidence the pilot was too small to rank configurations.

**Also tested and rejected:** rescaling the trained output by ×1.69 to undo the
0.59 compression. It fixes the diagnostics (material slope 0.513 → 0.867) but in
play is non-transitive — loses 5–27 to the net it came from. Not adopted. The
underlying point stands: this search's pruning margins are cp-denominated
(`static − 85·depth ≥ beta`, null-move, aspiration), so they interact with eval
scale and should be treated as tunable if a network eval is ever adopted.

## 4. Dataset, label distribution, update budget, validation curve

**Version:** `overnight_shards_8m`, 32 shards, `schema_version` 2,
`min_depth` 16, `stop_reason` limit, source `pilot_raw.jsonl` (3.94 GB).

| | value |
|---|---|
| Rows summed independently from shards | **8,000,000** (agrees with manifest `kept`) |
| Scanned / rejected | 8,274,453 / 191,574 no-usable-eval, 53,625 invalid-position, 29,254 duplicate |
| Splits | train **7,840,256**, validation **79,698**, test **80,046** |
| Canonical keys unique | 8,000,000, **0 collisions** |
| Audit re-encode | **200/200** exact on indices, keys and signs |
| Labels on the ±6000 clamp | 1,204,140 (**15.05%**), mean +139.83 cp |
| Rows with an ep feature | 13,378 (0.167%) |

**Contamination check:** the 8M set contains all 100,000 pilot positions, but
`split_indices` is a pure function of the key, so shared positions land in the
same split in both — **0 pilot-test rows in overnight training**. A random
re-split would have silently contaminated every pilot-vs-8M comparison.

**Update budget:** 30,626 updates/epoch at batch 256, so one epoch on 8M ≈ the
whole pilot tuned run's 31,023 updates. Load 11 s, peak RSS **1.76 GB**,
91 s/epoch at 256×32 and 142 s/epoch at 512×64.

**Validation curves:** `net_8m_scale1` best epoch 12/12 (still improving —
undertrained). `net_8m_scale400` best 8/12. `net_8m_scale400_cos24` best **17/24**,
plateau at ~15 then drift up. `net_8m_512x64` best **19/30**, rising after —
both later runs are converged, not cut short.

## 5. Trained weights, source, reproduction

All in the bundle and on branch `claude/neural-network-training-14g3jq`.

| Checkpoint | What it is |
|---|---|
| **`net_8m_scale400_cos24.npz`** | **RECOMMENDED.** 8M, 256×32, scale-400 head, cosine, best epoch 17 |
| `net_8m_512x64.npz` | 8M, 512×64. Converged; not better, 3.6× slower |
| `net_8m_scale400.npz`, `net_8m_scale1.npz` | 8M, 12-epoch runs |
| `net_8m_cos24_x17.npz` | ×1.69 rescale; tested, not recommended |
| `pilot_net.npz` | The handoff's prescribed 30-epoch baseline, kept for reproducibility |
| `pilot_net_tuned.npz`, `pilot_net_oscale400.npz`, `pilot_net_check.npz` | Pilot-era checkpoints |

```bash
cd chess-nn
python -m venv .venv && ./.venv/bin/python -m pip install -r requirements_nn.txt
# put the received data in place: pilot_shards/ and overnight_shards_8m/

./.venv/bin/python verify_snapshot.py overnight_shards_8m
PYTHONPATH=engine ./.venv/bin/python test_nn.py
PYTHONPATH=engine ./.venv/bin/python test_engine_parity.py

# the recommended checkpoint
./.venv/bin/python train_nnue.py train --shards overnight_shards_8m \
    --out net_8m_scale400_cos24.npz --epochs 24 --batch-size 256 \
    --learning-rate 0.003 --seed 0 --scale 400 --lambda 1.0 --loss mse \
    --output-scale 400 --lr-schedule cosine

./.venv/bin/python check_export_parity.py net_8m_scale400_cos24.npz --shards overnight_shards_8m
./.venv/bin/python compare_checkpoints.py --shards overnight_shards_8m \
    net_8m_scale400_cos24.npz net_8m_512x64.npz
./.venv/bin/python test_material.py net_8m_scale400_cos24.npz
PYTHONPATH=engine ./.venv/bin/python test_blunder.py net_8m_scale400_cos24.npz
PYTHONPATH=engine ./.venv/bin/python bench_search.py net_8m_scale400_cos24.npz
PYTHONPATH=engine ./.venv/bin/python match.py --nn net_8m_scale400_cos24.npz \
    --games 48 --depth 5 --base 100000 --inc 0
```

## 6. Engine timing and match results

**Timing — these are real search numbers, not forward-pass throughput.**

| | value |
|---|---|
| Import + JIT warmup, both evaluators compiled | **27.5 s** (90 s budget) |
| `load_nn` incl. NN-path warmup | ~1 s |
| Worker start to first move accepted | 27–30 s |
| **Full-search NPS, hand-crafted** | **1.28–1.63 M nodes/s** |
| **Full-search NPS, NN 256×32** | **75 k–233 k nodes/s** (7–20× slower) |
| Forward pass only, 256×32 / 512×64 | 157,811 / 43,954 evals/sec |

The 156k evals/sec figure in `REPORT.md` is **not** NPS and must not be compared
with the rows above.

**Matches** — paired openings, colours reversed, each side in its own process.
Fixed depth removes the NN's speed handicap and measures evaluation quality only.

| Match | Games | Result | Score rate |
|---|---|---|---|
| `cos24` vs hand-crafted, 5 s + 0.1 s | 8 | 0 – 8 | 0% |
| `cos24` vs hand-crafted, **depth 5** | 48 | **1 – 47** | **2.1% ± 2.1%** |
| `cos24` vs `pilot_net_tuned`, depth 5 | 16 | **14.5 – 1.5** | **91%** |
| `cos24` vs `net_8m_scale400`, depth 5 | 48 | **31.5 – 16.5** | **65.6% ± 6.9%** |
| `net_8m_512x64` vs `cos24`, depth 5 | 48 | 19.5 – 28.5 | 40.6% ± 7.1% |
| `cos24 ×1.69` vs `cos24`, depth 5 | 32 | 5.0 – 27.0 | 15.6% ± 6.4% |

**No illegal moves, no flags, no crashes in any match** after the fix in
section 2 — before it, the NN forfeited on time and played fallback moves.

Caveat: these matches are **deterministic** (fixed depth, deterministic search,
fixed openings), so re-running reproduces them exactly. The quoted standard
errors describe variation across *openings*, not repeat sampling.

**Not done:** the competition-clock gauntlet. The competition configuration was
not supplied with the engine, so the 120 s + 0.5 s / one core / 90 s init figures
remain unverified here.

## 7. Is the candidate ready?

**No. Not ready for deployment. Keep the hand-crafted engine as the submission.**

`net_8m_scale400_cos24.npz` is ready for **further testing**. It is a genuine,
large improvement over the pilot (91% head to head) and over every other
checkpoint here, but it scores **2.1%** against the hand-crafted engine even
with its speed handicap entirely removed.

What the four tuning levers showed:

| Lever | Effect |
|---|---|
| More data, 100k → 8M | **Large win** |
| Longer training + cosine | **Real win**, CI above 50% |
| Output-scale head | Small win |
| cp rescale ×1.69 | No win, non-transitive |
| More capacity, 2.1× params | **No win**, at 3.6× cost |

Two candidate explanations for the gap were tested and **rejected**: the eval
does not hang material more than the hand-crafted one (19.3% vs 24.0% at 1 ply),
and it is not simple cp miscalibration. Loss keeps improving in small increments
while the standing against the reference does not move.

**Conclusion: the ceiling is the training distribution.** The net learns from
Lichess *analysis* positions; the search evaluates leaves that are unbalanced and
mid-tactic. Further width, schedule or output tuning on this dataset is not worth
the compute. The next experiment worth running is the one the original handoff
anticipated: label **engine-reached positions** with a fixed Stockfish setup, mix
them in, and re-run this same gate. That needs a Stockfish binary, which this
container does not have.

Open decision for the team: the **en-passant convention** (section 2).
