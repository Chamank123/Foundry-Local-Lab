"""One engine process for match.py.

Each side runs in its OWN process so the two players get independent
transposition tables, killers, history and repetition lists. Switching the
global evaluator inside a single process would force a new_game() reset before
every move and hand both sides a crippled, non-representative search.

    python match_worker.py [weights.npz]      # no argument = hand-crafted
"""
import sys


def fixed_depth_move(engine, fen, depth):
    """Iterative deepening to a FIXED depth, with no clock.

    A time-limited match cannot separate evaluation quality from evaluation
    cost: the NN is 7-20x slower per node than the tapered PST, so at equal
    time it simply searches shallower. Giving both sides the same depth removes
    that handicap and asks only which evaluation steers better.
    """
    bd = engine.board_np(fen)
    # Mirror get_move's per-move bookkeeping so repetition and ordering behave.
    key = engine.zkey(bd)
    if engine._nghist < engine._ghist.shape[0]:
        engine._ghist[engine._nghist] = key
        engine._nghist += 1
    engine._killers.fill(-1)
    engine._history.fill(0)
    engine._stop[0] = 0
    engine._nodes[0] = 0

    best, prev = -1, -1
    for d in range(1, depth + 1):
        score, move, completed = engine.search_root(
            bd, d, prev if prev != -1 else 0, -engine.INF, engine.INF,
            engine._tt_key, engine._tt_data, engine._killers, engine._history,
            engine._stop, engine._nodes, engine._ghist, engine._nghist,
            engine._movebuf, engine._scorebuf, engine._mbbuf,
            engine._NNW, engine._USE_NN[0])
        if completed and move != -1:
            best, prev = move, move
    if best == -1:
        return "0000"
    return engine._uci(int(best))


def main():
    args = [a for a in sys.argv[1:]]
    depth = 0
    if "--depth" in args:
        i = args.index("--depth")
        depth = int(args[i + 1])
        del args[i:i + 2]
    weights = args[0] if args else None
    import agent
    import search_numba as engine
    if weights:
        engine.load_nn(weights)
    engine.new_game()
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            return
        if line == "NEWGAME":
            engine.new_game()
            sys.stdout.write("OK\n")
            sys.stdout.flush()
            continue
        fen, _, remaining = line.rpartition("|")
        if depth:
            sys.stdout.write(fixed_depth_move(engine, fen, depth) + "\n")
        else:
            sys.stdout.write(agent.get_move(fen, int(remaining)) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
