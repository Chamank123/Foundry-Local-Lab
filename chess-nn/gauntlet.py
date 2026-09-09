"""Independent paired-opening gauntlet between two engine DIRECTORIES.

Each side is a directory containing its own agent.py/search_numba.py, run in its
own process, so a candidate that auto-loads a network at import is exercised
exactly as the judge would run it. Neither side can see the other's
transposition table, history or repetition list.

    python gauntlet.py --a candidate --b hybrid/engine_pristine --games 40 \
        --base 10 --inc 0.1

Deviation from a strict competition harness: processes are REUSED across games
with new_game() between them, which clears the TT, killers, history and
repetition list. Fresh processes per game would be more faithful but cost ~33 s
of JIT warmup each, which dominates the wall clock at this sample size.
"""
import argparse
import subprocess
import sys
import time

import chess

OPENINGS = [
    "e2e4 e7e5", "d2d4 d7d5", "e2e4 c7c5", "d2d4 g8f6",
    "c2c4 e7e5", "g1f3 d7d5", "e2e4 e7e6", "d2d4 d7d5 c2c4 c7c6",
    "e2e4 c7c6", "d2d4 g8f6 c2c4 e7e6", "e2e4 d7d5", "d2d4 e7e6",
    "c2c4 g8f6", "g1f3 g8f6", "e2e4 d7d6", "d2d4 d7d5 g1f3 g8f6",
    "e2e4 g8f6", "d2d4 f7f5", "c2c4 c7c5", "e2e4 b8c6",
    "d2d4 g8f6 c2c4 g7g6", "e2e4 e7e5 g1f3 b8c6", "d2d4 d7d5 c2c4 e7e6",
    "e2e4 c7c5 g1f3 d7d6",
]

WORKER = r'''
import sys, os
sys.path.insert(0, os.getcwd())
import agent, search_numba as engine
sys.stdout.write("READY\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    if line == "QUIT": break
    if line == "NEWGAME":
        engine.new_game()
        sys.stdout.write("OK\n"); sys.stdout.flush(); continue
    if line == "NODES":
        sys.stdout.write(str(int(engine._nodes[0])) + "\n"); sys.stdout.flush(); continue
    fen, _, ms = line.rpartition("|")
    sys.stdout.write(agent.get_move(fen, int(ms)) + "\n"); sys.stdout.flush()
'''


class Engine:
    def __init__(self, directory, label):
        self.label = label
        started = time.perf_counter()
        self.proc = subprocess.Popen([sys.executable, "-c", WORKER], cwd=directory,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        if self.proc.stdout.readline().strip() != "READY":
            raise RuntimeError(f"{label} failed to start")
        self.startup = time.perf_counter() - started
        self.nodes = 0
        print(f"  {label:28} started in {self.startup:5.1f}s  ({directory})")

    def send(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n"); self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def play(white, black, opening, base, inc, max_plies=600):
    board = chess.Board()
    for uci in opening.split():
        board.push(chess.Move.from_uci(uci))
    white.send("NEWGAME"); black.send("NEWGAME")
    clock = {chess.WHITE: base, chess.BLACK: base}
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        side = board.turn
        engine = white if side == chess.WHITE else black
        started = time.perf_counter()
        try:
            uci = engine.send(f"{board.fen()}|{int(clock[side] * 1000)}")
        except Exception as exc:
            return ("0-1" if side == chess.WHITE else "1-0"), f"crash {exc}"
        spent = time.perf_counter() - started
        clock[side] -= spent
        if clock[side] < 0:
            who = white.label if side == chess.WHITE else black.label
            return ("0-1" if side == chess.WHITE else "1-0"), f"FLAG({who})"
        clock[side] += inc
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return ("0-1" if side == chess.WHITE else "1-0"), f"UNPARSEABLE {uci}"
        if move not in board.legal_moves:
            return ("0-1" if side == chess.WHITE else "1-0"), f"ILLEGAL {uci}"
        board.push(move)
    try:
        engine.send("NODES")
    except Exception:
        pass
    if board.is_game_over(claim_draw=True):
        return board.result(claim_draw=True), board.outcome(claim_draw=True).termination.name
    return "1/2-1/2", f"ply cap {board.ply()}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="candidate directory")
    parser.add_argument("--b", required=True, help="reference directory")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--base", type=float, default=10.0)
    parser.add_argument("--inc", type=float, default=0.1)
    args = parser.parse_args()

    print(f"clock {args.base}s + {args.inc}s/move, paired openings, colours reversed")
    a = Engine(args.a, "A: " + args.a)
    b = Engine(args.b, "B: " + args.b)

    score = {"a": 0.0, "b": 0.0}
    issues, terminations = [], {}
    started = time.perf_counter()
    game = 0
    while game < args.games:
        opening = OPENINGS[(game // 2) % len(OPENINGS)]
        a_white = (game % 2 == 0)
        white, black = (a, b) if a_white else (b, a)
        result, note = play(white, black, opening, args.base, args.inc)
        terminations[note] = terminations.get(note, 0) + 1
        if result == "1-0":
            winner = "a" if a_white else "b"
        elif result == "0-1":
            winner = "b" if a_white else "a"
        else:
            winner = None
        if winner:
            score[winner] += 1.0
        else:
            score["a"] += 0.5; score["b"] += 0.5
        if note.startswith(("FLAG", "ILLEGAL", "UNPARSEABLE", "crash")):
            issues.append(f"game {game + 1}: {note}")
        game += 1
        print(f"  game {game:3d}/{args.games}  A as {'W' if a_white else 'B'}  "
              f"{result:7}  {note:22}  running A {score['a']:.1f} - {score['b']:.1f} B",
              flush=True)

    a.close(); b.close()
    n = score["a"] + score["b"]
    rate = score["a"] / n
    stderr = (rate * (1 - rate) / n) ** 0.5
    print(f"\nA {score['a']} - {score['b']} B over {n:.0f} games "
          f"({time.perf_counter() - started:.0f}s)")
    print(f"A score rate {rate * 100:.1f}% +- {stderr * 100:.1f}% (1 s.e.); "
          f"95% CI roughly [{max(0, rate - 2 * stderr) * 100:.0f}%, "
          f"{min(1, rate + 2 * stderr) * 100:.0f}%]")
    print(f"terminations: {terminations}")
    print(f"operational issues: {len(issues)}" + (f" -> {issues}" if issues else ""))
