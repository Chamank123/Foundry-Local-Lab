# 8M dataset report

Third report, covering the 8,000,000-position overnight snapshot. Supersedes the
affected parts of `REPORT.md` and `REPORT2.md`. Same container: 4 CPU cores, no
GPU, torch 2.14.0, numpy 2.4.6, numba 0.67.0, python-chess 1.11.2, Python 3.11.

**Headline:** the data worked. The 8M net beats the pilot net **14.5–1.5 (91%)**
head to head, its material understanding improved more than tenfold, and the
"material blindness" I reported for the pilot is largely gone. It is still well
behind the hand-crafted engine — **2–14 at equal depth** — so it is not a
candidate, but the gap is now an ordinary evaluation-quality gap rather than a
broken evaluator.

---

## 1. Snapshot verification — passed

`verify_snapshot.py` checks each shard on its own before anything is
concatenated, so a bad shard is named rather than surfacing 8M rows later.

| Check | Result |
|---|---|
| Shards received | 32, contiguous `00000`–`00031`, all byte-identical in size |
| Rows summed independently from shards | **8,000,000** — agrees with manifest `kept` |
| `min_depth` / `schema_version` / `stop_reason` | 16 / 2 / `limit` |
| Rejected during prep | 191,574 no-usable-eval, 53,625 invalid-position, 29,254 duplicate (of 8,274,453 scanned) |
| Exactly 2 king features per row | **all 8,000,000** |
| Active-feature counts | 3–37, within the 40-slot capacity |
| `wdl == -1`, `group == 0`, `\|y\| <= 6000` | hold throughout |
| Canonical keys unique | **8,000,000 unique, 0 collisions** |
| Splits | train 7,840,256 / validation 79,698 / test 80,046, disjoint, summing to total |
| Audit sample | **200/200** rows re-encoded from their FENs reproduce the stored active indices, canonical keys and mover-relative signs exactly |

The audit result is also the answer to "compare your encoding with the
collector's frozen 780-feature contract": they agree exactly.

Labels: **15.05% sit on the ±6000 clamp** (pilot 18.9%), mean +139.83 cp.

### Two findings from verification

**The 8M set contains all 100,000 pilot positions.** Because `split_indices` is
a pure function of the key (`key % 1000`), a shared position lands in the *same*
split in both datasets. **Zero pilot test rows appear in overnight training.**
The deterministic split is what makes the two datasets safely comparable — a
random re-split would have silently contaminated every pilot-vs-8M comparison
below.

**The en-passant convention is unchanged.** 13,378 of 8,000,000 rows (**0.167%**)
carry an ep feature, matching the pilot's 0.16% and the measured "an ep capture
is actually legal" rate (0.38%), not the engine's "any double push" rate (5.5%).
The skew in `REPORT2.md` section 2 therefore carries into this dataset. Still
your decision; the encoder remains untouched.

## 2. Training

Carried the pilot's best settings (batch 256, lr 3e-3) and changed one thing at
a time. **30,626 updates per epoch**, so a single epoch here is roughly the
whole pilot tuned run's 31,023-update budget. 91 s per epoch; load 11 s; peak
RSS **1.76 GB**, comfortable on 15 GB.

| Run | epochs | best epoch | val RMSE | test RMSE |
|---|---|---|---|---|
| `net_8m_scale1.npz` | 12 | **12 of 12** | 0.14238 | 0.14469 |
| `net_8m_scale400.npz` | 12 | 8 of 12 | **0.14062** | **0.14316** |

Scale-1 peaked on its final epoch, so it is still improving and undertrained;
scale-400 converged inside the budget. A longer cosine run is under way.

## 3. Comparison on one fixed validation distribution

`train()` reports RMSE on its own dataset's split, and the pilot's 1,031-row
validation set is a **subset** of the 8M set's 79,698 — those printed numbers
are not comparable. `compare_checkpoints.py` scores every checkpoint on the same
20,000 rows of the 8M validation split. Trivial baseline (predict the mean
target): **0.24513**.

| Checkpoint | RMSE | Pearson | Spearman | sign | median AE | output range |
|---|---|---|---|---|---|---|
| `pilot_net_tuned.npz` | 0.18311 | 0.694 | 0.554 | 75.3% | 141.3 | −1199 … +1057 |
| `pilot_net_oscale400.npz` | 0.19107 | 0.704 | 0.533 | 74.8% | 141.0 | −2449 … +2874 |
| `net_8m_scale1.npz` | 0.13835 | 0.815 | 0.719 | 84.8% | 103.8 | −1359 … +1560 |
| **`net_8m_scale400.npz`** | **0.13692** | 0.815 | **0.725** | 84.7% | **101.6** | −1624 … +1960 |

Note `pilot_net_tuned` scores 0.12082 on its own validation set but **0.18311**
here. `REPORT.md`'s pilot numbers were optimistic because that 1,031-row split
was small and easy, not because the net was good. Judge checkpoints on a common
distribution.

## 4. Material — the pilot's biggest defect is largely fixed

Same controlled probe as `REPORT2.md`: 600 quiet, roughly balanced, legal,
same-side-to-move comparisons.

| Checkpoint | correct direction | material slope | mean Δ losing a queen |
|---|---|---|---|
| `pilot_net_tuned.npz` | 67% | 0.032 | −45 / +75 cp |
| `net_8m_scale1.npz` | 70% | 0.361 | −232 / +442 cp |
| **`net_8m_scale400.npz`** | **78%** | **0.377** | −232 / +467 cp |

**A tenfold improvement in the material slope**, from the data alone.

### Correcting how I described this

A slope below 1.0 is not by itself a defect. Scaling every evaluation by a
positive constant does not change move ordering, so a *uniformly* compressed net
is one a search barely notices. What matters is whether material is compressed
**relative to the net's other terms**. Comparing the material slope with the
net's overall response slope on non-saturated labels (|y| ≤ 800, 81% of rows):

| Checkpoint | in-band slope | material slope | material ÷ in-band |
|---|---|---|---|
| `pilot_net_tuned.npz` | 0.361 | 0.032 | **0.09** |
| `net_8m_scale1.npz` | 0.533 | 0.361 | 0.68 |
| `net_8m_scale400.npz` | 0.551 | 0.377 | **0.68** |

So the pilot net really was material-blind — material was 11× more compressed
than everything else it had learned. The 8M nets track material at 68% of their
overall response: mildly under-weighted, not blind. The remaining compression is
mostly uniform, and therefore mostly harmless to search. **"The net doesn't
understand material" is no longer the right description of the 8M nets**, and it
is no longer the leading explanation for the playing gap in section 6.

## 5. Output scale — the pilot's contradiction did not reproduce

At pilot scale, `REPORT2.md` found the scale-400 head won on loss but **lost** on
material (58% vs 67%), which is why I refused to pick a candidate on loss alone.
At 8M that contradiction is gone: scale-400 is better on **both** axes (RMSE
0.13692 vs 0.13835; material 0.377 vs 0.361, 78% vs 70%). It also converges
sooner. That is now a settled, if modest, win, and it reinforces the earlier
point — the pilot was too small to rank configurations reliably.

Export parity holds for both, on 1,000 held-out positions: raw float agreement
to ≤4.8e-03 cp, and clamped-int agreement 0/1000 rows differing for scale-1 and
1/1000 differing by 1 cp for scale-400 (truncation either side of an integer
boundary, not an export defect).

## 6. Play

Paired openings, colours reversed, each side in its own process.

**Time control 5 s + 0.1 s** (a smoke test, not the competition gate):

| Candidate | Result | Illegal | Flags | Crashes |
|---|---|---|---|---|
| `net_8m_scale400.npz` vs hand-crafted | 0 – 8 | 0 | 0 | 0 |

**Fixed depth 5, no clock** — added in `match.py --depth` because a timed match
cannot separate evaluation *quality* from evaluation *cost*: the NN is 7–20×
slower per node, so at equal time it simply searches shallower.

| Match | Result | Score rate |
|---|---|---|
| `net_8m_scale400.npz` vs hand-crafted | **2 – 14** | 12.5% |
| **`net_8m_scale400.npz` vs `pilot_net_tuned.npz`** | **14.5 – 1.5** | **91%** |

Read those together:

* **The data delivered.** 91% against the pilot net at equal depth is a large,
  unambiguous strength gain, and it is a *playing* result, not a loss metric.
* **The hand-crafted engine is still much better.** Removing the entire speed
  handicap moved the result only from 0–8 to 2–14. So the gap is not mainly
  speed, and — per section 4 — not mainly material either. It is general
  evaluation quality: the tapered PST with bishop-pair and passed-pawn terms is
  simply a strong evaluator for this search.
* 16 games is a small sample. These separate "clearly worse" from "clearly
  better"; they cannot resolve anything finer, and none of this is the
  competition gate.

## 7. Status

**Not a deployment candidate.** Keep the hand-crafted engine as the submission.
`net_8m_scale400.npz` is the best checkpoint so far and is ready for further
testing, not deployment.

## 8. Suggested next steps

1. **Decide the en-passant convention** (section 1). Unchanged from `REPORT2.md`
   and now affects 8M rows.
2. **Train longer.** Both 12-epoch runs are short for this data; scale-1 was
   still improving at its last epoch. A 24-epoch cosine run is in progress.
3. **Close the quality gap, not the material gap.** Section 4 says material is
   mostly fixed. The remaining ideas are more capacity (H0/H1 are 256/32),
   longer training, and the 15% clamped labels — but note the clamp
   recommendation from `REPORT.md` stays withdrawn.
4. **Consider the speed budget honestly.** Even a net that matched the PST in
   quality would give up 7–20× in nodes. Incremental accumulator updates and
   quantisation are the documented answers, and they stop being optional once
   the evaluation is competitive.
5. **Re-run `verify_snapshot.py`, `check_export_parity.py` and
   `test_engine_parity.py`** on any new dataset or checkpoint before matches.

## Files

| File | What it is |
|---|---|
| `verify_snapshot.py` | Per-shard schema gate plus cross-shard invariants |
| `compare_checkpoints.py` | Scores checkpoints on one fixed validation distribution |
| `net_8m_scale1.npz` | 8M, scale-1 head, 12 epochs |
| `net_8m_scale400.npz` | **Best so far.** 8M, scale-400 head, best epoch 8 |
| `match.py --depth`, `--nn2` | Fixed-depth matches and network-vs-network |

`overnight_shards_8m/` is gitignored and left unchanged, as received.
`nn_encode.py` remains untouched.

## Reproducing

```bash
cd chess-nn
./.venv/bin/python verify_snapshot.py overnight_shards_8m
./.venv/bin/python train_nnue.py train --shards overnight_shards_8m \
    --out net_8m_scale400.npz --epochs 12 --batch-size 256 \
    --learning-rate 0.003 --seed 0 --scale 400 --lambda 1.0 --loss mse \
    --output-scale 400
./.venv/bin/python compare_checkpoints.py --shards overnight_shards_8m \
    pilot_net_tuned.npz net_8m_scale1.npz net_8m_scale400.npz
./.venv/bin/python test_material.py net_8m_scale400.npz
PYTHONPATH=engine ./.venv/bin/python match.py --nn net_8m_scale400.npz \
    --nn2 pilot_net_tuned.npz --games 16 --depth 5 --base 100000 --inc 0
```
