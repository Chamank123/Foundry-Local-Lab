# Next steps after the pilot report — trainer and Claude handoff

This supplements REPORT.md, README_PILOT_HANDOFF.md and the overnight kit.
For next training runs, the instructions here supersede the earlier generic
batch-8192 / 30-epoch recommendation. Keep the existing dataset encoding,
trained weights and current working trainer. Do not restart the project.

## 1. The team's immediate plan

Work in parallel:

- Aryan supplies a complete snapshot of the actual engine and runs the existing
  overnight collector. First milestone: at least 1M verified new positions;
  continue towards 5M if the existing time/download budget allows.
- Trainer + Claude use the engine snapshot to close the missing engine tests,
  check the output scaling issue below and integrate the tuned pilot as a test
  candidate. Do not wait for the full larger dataset to start integration.
- ChatGPT has reviewed REPORT.md and independently inspected both uploaded
  weight files. It has not received the actual engine, the full pilot shards,
  or the updated trainer/test scripts referenced in the report, so it cannot
  independently certify those missing pieces yet.

The current choice is `pilot_net_tuned.npz` for integration experiments.
Preserve `pilot_net.npz` as a reproducibility artifact. Preserve the working
hand-crafted engine as the default submission until a candidate wins the
competition-time-control comparison with adequate evidence.

## 2. What Aryan should supply

Supply the COMPLETE current engine source snapshot with its directory structure:

- Actual `bb_numba.py` and every module it imports, e.g. board/move-generation
  helpers if present. A fabricated substitute is not acceptable.
- Actual search implementation, e.g. `search_numba.py` if that is its name.
- Actual entrypoint/submission agent, evaluation code and runtime assets.
- Dependency/install files, run instructions and competition configuration.
- Existing engine tests, match runner, opening set and the preserved baseline.
- Commit identifier if available; include current uncommitted changes if those
  are the version in use. A GitHub ZIP only includes the selected committed ref.

Do not require those illustrative filenames if the real repository uses different
ones; inspect and use its actual structure. No virtual environment, credentials
or bulk raw databases are needed in the source handoff.

The trainer already has the pilot work. Keep all of it, including the modified
`train_nnue.py`, `test_nn_offline.py`, `check_export_parity.py`, `eval_report.py`,
and sweep scripts/results. Those files were listed in REPORT.md but were not
among the three files supplied to ChatGPT. Export them together if a second
review is wanted; the two NPZs and report are not a complete reproducible project.

## 3. Corrections to the conclusions in REPORT.md

### The original optimisation settings were inadequate for this pilot

With 97,968 training rows and batch8192, there are ceil(97968/8192)=12 optimiser
updates per epoch, or 360 in 30 epochs. With batch256 there are 383 per epoch:
120 epochs perform 45,960 updates, with the reported best at epoch81 after
31,023 updates. Thus the retune also increased epochs and total update count;
its gains are not isolated evidence for batch size or learning rate alone.

Record sample count, batch size, updates, epochs, learning rate, elapsed time,
validation curve and output range. Epoch counts alone are not comparable across
dataset sizes. A 3-epoch run is an execution/learning check, not proof of strength.

### Independent checkpoint inspection found exact output-range bounds

The last hidden layer is clipped to [0,1]. For raw output `b + sum(w_i*h_i)`,
the output must lie between `b + sum(min(w_i,0))` and `b + sum(max(w_i,0))`.
These bounds follow directly from the attached weights:

| Checkpoint | Lower bound (cp) | Upper bound (cp) |
|---|---:|---:|
| pilot_net.npz | -7.58780584 | +7.83174363 |
| pilot_net_tuned.npz | -1198.48960114 | +1056.71852875 |

The tuned limits match the reported observed min/max. The baseline could not
possibly output a sensible hundreds-of-centipawns score with its saved weights.
This is not an explicit output clamp of +/-8 or +/-1200 in the runtime; it is
the current weights combined with the clipped hidden layer. Training can change
these bounds. The baseline is not literally constant over all inputs, but its
extremely narrow cp range makes it unsuitable as the intended evaluation.

Independent NumPy inference reproduced:

| Position | Baseline cp | Tuned cp |
|---|---:|---:|
| Starting position | +7.8317 | +31.8749 |
| Starting position with Black queen removed, White to move | +7.8317 | +608.1367 |
| Starting position with White queen removed, White to move | -5.2621 | -13.2227 |

On the tuned starting-position input, 23 of the 32 last-hidden activations are
exactly1 and eight are exactly0. This is evidence to investigate activation
saturation and output parameterisation. It is not proof that saturation alone
causes the poor material responses: clipped networks legitimately use boundary
activations, and bounding outputs is inherent to this architecture.

Reproduce bounds using the actual checkpoint (no PyTorch required):

```python
import numpy as np
for filename in ("pilot_net.npz", "pilot_net_tuned.npz"):
    with np.load(filename, allow_pickle=False) as z:
        w = z["out_w"].astype(np.float64).ravel()
        b = float(z["out_b"].ravel()[0])
        print(filename, b + np.minimum(w, 0).sum(), b + np.maximum(w, 0).sum())
```

### More data is a useful experiment, not a proven complete fix

An overfitting validation curve supports increasing data/diversity, but the ratio
of rows to parameters does not establish that 100k positions cannot teach material.
Output scaling, gradient flow, distribution and teacher coverage remain plausible
contributors. Do not declare the dataset exhausted or all remaining issues
structural without additional evidence.

Removing a queen does not universally imply a -900 teacher-evaluation change:
tactics, side to move, already-winning/losing positions and newly invalid boards
can affect that test. Keep only legal comparisons, control side-to-move, separate
quiet roughly balanced positions, and ideally compare the same modifications
with the teacher. Bad responses in sensible examples remain useful diagnostics.

### Saturated target labels do not automatically mean zero training gradient

For `L=(p-t)^2`, `p=sigmoid(z/K)`, `t=sigmoid(cp/K)`,
`dL/dz = 2*(p-t)*p*(1-p)/K`. The target is a constant during differentiation.
A mate-labelled target near1 can produce a substantial gradient when the model
predicts near0.5. Extreme labels lose score-magnitude distinction, but they still
teach winning/losing. At K400, sigmoid(1500/400) is about0.977, not exactly1.
Lowering a mate label from6000 to2000 changes its target only from about0.9999997
to0.9933; this is not a demonstrated remedy for the poor material behaviour.

The same sigmoid-space MSE construction is documented in the official
[Stockfish NNUE loss discussion](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#mean-squared-error-mse).

Also, `abs(y)==6000` means a clamped label. It does not by itself prove that the
original source PV had a mate score: an extreme finite cp value may also clamp.
The current six-array training schema does not carry an original `is_mate` flag.
Do not describe all such rows as verified forced mates without checking source
records. Search is not guaranteed to find every long mate at its available depth.

## 4. Trainer: do these tasks now, in this order

### A. Preserve and inspect the real engine

Create an isolated working copy/branch and retain an executable hand-crafted
baseline. Read the engine's own instructions. Identify actual board layout,
score perspective, mate/draw conventions, evaluation call sites, make/unmake,
quiescence, transposition-table handling, init budget and clock controls.

Put the NN package where the real engine imports resolve. `test_nn.py` in the
earlier package looks in its own directory and its parent; this does not prove
the actual project uses either layout. Adapt import wiring to the real project,
not the board format to an assumed test fixture.

Run `python test_nn.py` with the real `bb_numba.board_np` and real dependencies.
Resolve actual failures. Do not mock engine imports or count offline-only tests
as passing this gate. Exercise both movers, castling, EP (including the engine's
representation when no legal EP capture exists), promotions, and board states
reached through actual make/unmake as well as FEN construction.

### B. Review the optimiser/output scale before another large sweep

Reproduce the tuned pilot and the output-range calculation. Log hidden-layer
saturation, gradient magnitudes and output bounds on a representative batch.
Check that cp-scale output from a small randomly initialised linear head is not
making training spend most updates merely growing output magnitude.

A bounded experiment, if warranted, is a normalised trainable output head with
a fixed conversion to cp, folding that conversion into exported output weights
and bias. This preserves the function class and runtime socket if done correctly.
It is an experiment, not a validated fix supplied by this handoff. Keep the
existing architecture dimensions and make the public model output raw cp;
re-run trained PyTorch/NumPy/numba parity after any such change. Do not fix this
by multiplying only runtime predictions or modifying labels ad hoc.

Compare with the working retuned baseline using fixed data/splits and comparable
update budgets. Make changes one at a time; do not simultaneously change output
scale, target scale, data filters and architecture. Avoid another large sweep
unless a specific remaining issue warrants it.

### C. Integrate the existing tuned pilot while data collection runs

Thread the six weight arrays through the actual search where necessary. Keep
the runtime non-incremental initially. Preserve mover-relative cp output, search
mate/draw handling and existing pruning assumptions. Make it possible to choose
hand-crafted vs NN evaluation and clear/recreate cached search state when switching.
Do not use PyTorch tensors at search leaves.

Run trained-checkpoint parity through actual engine board arrays, then the real
engine correctness tests and a small match smoke test. Check legal moves, no
crashes, game termination and safe clocks. The legal-move demo in `test_nn.py`
is not a full alpha-beta integration test.

Measure full clean-start/import/JIT time and full-search speed on the intended
machine. The reported 156k forward passes/sec is not full search NPS and excludes
some engine costs. The sub-0.1s reported forward compile is not a measurement of
the complete engine startup.

The README supplied by Aryan specifies 120s+0.5s/move, one CPU core and90s init.
Verify those against the actual current competition configuration provided with
the engine; distinguish any quick local test clock from the real selection gate.

## 5. Aryan: data collection can continue unchanged

Use `Chess_Overnight_Data_Kit.zip` and its `START_HERE.md`. Do not edit or replace
an actively running collector or its `foundation/` files: code hashes are part
of its resume contract. No new mate filter, label clamp or depth change is
required tonight. This preserves comparability and avoids discarding information
on an unverified explanation.

Keep the shared folder accessible. With default full parts, four completed
250k ZIPs give1M positions. Use the verifier's actual count if there are partial
parts. The trainer can receive those while the collector continues towards5M.
Only completed `.zip` files are ready, and cloud upload must finish first.

The existing source sample is sequential after skipping a raw prefix; more of
it is not automatically representative. Treat tonight's data as the next
development dataset. After collection, have the trainer compare piece counts,
side-to-move, label bands and clamped-label fraction with the pilot. If mate
filtering/balancing is later tested, produce a versioned alternative and evaluate
on a common fixed validation distribution. Keep the original data.

## 6. Trainer: when the first million positions arrive

1. Copy/download a fixed set of completed ZIPs locally.
2. Use the kit's receiver to validate and unpack into a NEW fixed snapshot:

```bash
python receive_overnight.py --from "/path/to/downloaded_parts" --out "/path/to/overnight_shards_1m"
```

3. Confirm successful `RECEIVED.json`, actual row counts and split counts. Keep
   that snapshot unchanged during training. Do not blindly append the old pilot.
4. Compare your current encoding with the collector's frozen 780-feature contract.
5. With the existing tuned trainer, one concrete FIRST 1M experiment is:

```bash
python train_nnue.py train --shards "/path/to/overnight_shards_1m" --out net_1m_trial.npz --epochs 12 --batch-size 256 --learning-rate 0.003 --seed 0 --scale 400 --lambda 1.0 --loss mse
```

The paths are placeholders for actual local folders. About980k training rows
produce about3,829 updates per epoch, so12 epochs give about46k updates, similar
to the tuned pilot's total120-epoch update count. This is a starting comparison,
not a promise that12 epochs is optimal or that equal updates isolate data effects.
If using an output-parameterisation experiment, record it separately.

Time the first few epochs and extrapolate on the actual hardware. Do not promise
that1M is trivial on a GPU or insist on GPU availability: the report already
demonstrates useful pilot training on a four-core CPU. Larger runs need measured
time and RAM budgets. The existing loader concatenates shards in RAM.

Select checkpoints using validation, save to distinct filenames, recheck trained
export parity and chess sanity. If validation is still improving at the end,
plan a longer run or proper checkpoint continuation. In the earlier trainer a
new command starts from scratch; it does not resume a previous training run.
Do not repeatedly choose configurations by their printed held-out test scores.

## 7. Engine comparison and final report

Once integration is correct, compare pilot vs hand-crafted briefly to establish
that the pipeline runs. Prioritise a serious comparison of the improved candidate
at the verified competition clock and CPU limit. Use paired opening positions
with colours reversed, record wins/draws/losses and uncertainty, plus flags,
crashes, startup, NPS and achieved search depth. A few wins or lower validation
loss alone do not establish superiority. Keep the baseline when inconclusive.

Return a concise report with:

- Exact engine commit/snapshot and training code changes.
- Real engine encoding/parity tests: commands, actual results and gaps.
- Output-scale investigation: what was measured, what changed, evidence.
- Dataset version/count, label distribution, update budget and validation curve.
- Trained weights, complete modified source and reproduction commands.
- Actual engine timing and match results; do not call forward throughput NPS.
- Whether the candidate is ready for further testing or has earned deployment.

Aryan can send that report/source/weights back to ChatGPT for an independent
review. Until the actual engine is supplied, ChatGPT cannot provide its missing
board constructor or certify search integration.
