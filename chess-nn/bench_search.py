"""Full-search speed and time-budget adherence, hand-crafted vs NN.

Section 7 of NEXT_STEPS is explicit that forward-pass throughput is NOT NPS.
This measures the real search: nodes per second through negamax/quiescence, the
depth actually completed, and -- because the smoke match produced a time
forfeit -- how long get_move() really takes against the clock it was given.

    PYTHONPATH=engine python bench_search.py [weights.npz]
"""
import sys
import time

import chess

POSITIONS = [
    ("start",     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("open game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("midgame",   "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("endgame",   "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]


def bench(label, weights, clock_ms=5000):
    import agent
    import search_numba as engine
    if weights:
        engine.load_nn(weights)
    engine.new_game()

    print(f"\n{label}")
    print(f"  {'position':10} {'move':6} {'elapsed':>9} {'budget':>9} "
          f"{'over?':>6} {'nodes':>10} {'NPS':>10}")
    worst_overshoot = 0.0
    for name, fen in POSITIONS:
        engine.new_game()
        budget = engine._budget_seconds(clock_ms) if hasattr(engine, "_budget_seconds") else None
        started = time.perf_counter()
        move = agent.get_move(fen, clock_ms)
        elapsed = time.perf_counter() - started
        nodes = int(engine._nodes[0])
        assert chess.Move.from_uci(move) in chess.Board(fen).legal_moves, "ILLEGAL"
        budget_txt = f"{budget:9.3f}" if budget else f"{'?':>9}"
        over = elapsed - budget if budget else 0.0
        worst_overshoot = max(worst_overshoot, over)
        print(f"  {name:10} {move:6} {elapsed:9.3f} {budget_txt} "
              f"{over:+6.3f} {nodes:10,} {nodes / elapsed:10,.0f}")
    return worst_overshoot


if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else None
    label = f"NN ({weights})" if weights else "hand-crafted"
    over = bench(label, weights)
    print(f"\nworst overshoot beyond the engine's own budget: {over:+.3f}s")
