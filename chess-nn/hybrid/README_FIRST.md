# Read this first

There are three ZIP files in this handoff. Keep them separate.

1. `SAFE_BASELINE.zip` is the original handcrafted engine, unchanged. This is
   still the safe submission today.
2. `RESIDUAL25_CANDIDATE.zip` is the experimental engine that automatically
   loads the new small residual network at 25% strength. It is formatted as a
   submission ZIP, but should not replace the baseline until a larger gauntlet
   confirms the result.
3. `HYBRID_EXPERIMENT_HANDOFF.zip` contains source, both relevant weight files,
   training code, tests, JSON results, PGNs, logs, and the detailed report.

## What happened in plain English

The big NN understands the teacher data better than the original evaluator, but
it is too slow inside search and loses games. Mixing 25% of that big NN into the
old evaluator still lost both competition-clock games.

A new, smaller NN was trained to learn only the old evaluator's mistakes. Using
one quarter of that correction scored 7/8 in the equal-depth screen and 2.5/4
at 120 seconds + 0.5 seconds/move: one win and three draws, with no crashes,
illegal moves, or flags. This is promising, not conclusive. Four games can be
luck.

## What to do next

Give all three ZIPs to the teammate running the gauntlet and send this prompt:

> Read `README_FIRST.md` and then `HYBRID_RESULTS.md` in full. Do not modify or
> replace `SAFE_BASELINE.zip`. Validate both ZIPs under the official Python 3.12
> image and actual AMD EPYC one-core restriction. Then run at least 50 paired
> games (100 games total) between `RESIDUAL25_CANDIDATE.zip` and
> `SAFE_BASELINE.zip` at exactly 120+0.5, using the competition opening set with
> colours reversed for each opening, fresh processes per game, automatic
> threefold and 50-move draws, and a 600-ply cap. Record W/D/L, score, confidence
> interval, flags, crashes, illegal moves, startup time, peak RAM, nodes, and
> PGNs. Do not tune on those match results. Recommend the residual candidate
> only if it wins convincingly and has zero operational failures; otherwise keep
> the safe baseline. Treat the old standalone/full NN as rejected.

Do not combine the residual checkpoint with another evaluator again: the file
already represents a correction to the original handcrafted evaluator, and the
candidate applies exactly the tested 25% amount.

## Evidence already collected

- All 8,000,000 supplied positions and 32 shards passed validation.
- All 200 source audit FENs and both evaluator signs passed.
- The original evaluator remained byte-identical in the safe baseline.
- The foundation suite and hybrid-specific suite passed.
- A clean extraction of the candidate ZIP imported in 31.02 seconds locally,
  confirmed evaluator mode 3250 (HC + 25% residual), and returned legal move
  `e2e4` from the starting position in 37.46 seconds total.
- Local peak memory in clock games was about 498 MB, below the 2 GB limit.

The local machine is not the judge's AMD EPYC, so the teammate must repeat the
startup, memory, and full gauntlet checks in the official environment.
