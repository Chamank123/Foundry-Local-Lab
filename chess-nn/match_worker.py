"""One engine process for match.py.

Each side runs in its OWN process so the two players get independent
transposition tables, killers, history and repetition lists. Switching the
global evaluator inside a single process would force a new_game() reset before
every move and hand both sides a crippled, non-representative search.

    python match_worker.py [weights.npz]      # no argument = hand-crafted
"""
import sys


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else None
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
        sys.stdout.write(agent.get_move(fen, int(remaining)) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
