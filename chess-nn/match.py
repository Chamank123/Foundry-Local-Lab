"""Paired-opening match between the hand-crafted evaluation and a NN checkpoint.

Section 7 of NEXT_STEPS asks for paired openings with colours reversed, and for
flags/crashes/startup to be recorded rather than just a win count. Each opening
is played twice so both engines get each colour from the same position.

This is a PIPELINE smoke test at a short clock unless --clock says otherwise;
it is NOT the competition selection gate.

    PYTHONPATH=engine python match.py --nn pilot_net_tuned.npz --games 8
"""
import argparse
import subprocess
import sys
import time

import chess

# Eight quiet, standard openings. Paired play (each twice, colours reversed)
# removes most of the colour bias from a short match.
OPENINGS = [
    "e2e4 e7e5", "d2d4 d7d5", "e2e4 c7c5", "d2d4 g8f6",
    "c2c4 e7e5", "g1f3 d7d5", "e2e4 e7e6", "d2d4 d7d5 c2c4 c7c6",
    "e2e4 c7c6", "d2d4 g8f6 c2c4 e7e6", "e2e4 d7d5", "d2d4 e7e6",
    "c2c4 g8f6", "g1f3 g8f6", "e2e4 d7d6", "d2d4 d7d5 g1f3 g8f6",
    "e2e4 g8f6", "d2d4 f7f5", "c2c4 c7c5", "e2e4 b8c6",
    "d2d4 g8f6 c2c4 g7g6", "e2e4 e7e5 g1f3 b8c6", "d2d4 d7d5 c2c4 e7e6",
    "e2e4 c7c5 g1f3 d7d6",
]


class Worker:
    def __init__(self, weights, label, depth=0):
        self.label = label
        cmd = [sys.executable, "match_worker.py"] + ([weights] if weights else [])
        if depth:
            cmd += ["--depth", str(depth)]
        started = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = self.proc.stdout.readline().strip()
        self.startup = time.perf_counter() - started
        if ready != "READY":
            raise RuntimeError(f"{label} failed to start: {ready!r}")
        print(f"  {label}: started in {self.startup:.1f}s")

    def new_game(self):
        self.proc.stdin.write("NEWGAME\n")
        self.proc.stdin.flush()
        self.proc.stdout.readline()

    def move(self, fen, remaining_ms):
        self.proc.stdin.write(f"{fen}|{int(remaining_ms)}\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.label} died")
        return line.strip()

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def play(white, black, opening, base_s, inc_s, max_plies=200):
    board = chess.Board()
    for uci in opening.split():
        board.push(chess.Move.from_uci(uci))
    white.new_game()
    black.new_game()
    clock = {chess.WHITE: base_s, chess.BLACK: base_s}
    notes = []
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        side = board.turn
        worker = white if side == chess.WHITE else black
        started = time.perf_counter()
        try:
            uci = worker.move(board.fen(), clock[side] * 1000)
        except RuntimeError as exc:
            return ("0-1" if side == chess.WHITE else "1-0"), [f"crash: {exc}"]
        spent = time.perf_counter() - started
        clock[side] -= spent
        if inc_s >= 0 and base_s > 0 and clock[side] < 0:
            return ("0-1" if side == chess.WHITE else "1-0"), ["FLAG (time forfeit)"]
        clock[side] += inc_s
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return ("0-1" if side == chess.WHITE else "1-0"), [f"unparseable move {uci!r}"]
        if move not in board.legal_moves:
            return ("0-1" if side == chess.WHITE else "1-0"), [f"ILLEGAL move {uci}"]
        board.push(move)
    if board.is_game_over(claim_draw=True):
        return board.result(claim_draw=True), notes
    return "1/2-1/2", notes + [f"adjudicated draw at ply {board.ply()}"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn", required=True)
    parser.add_argument("--nn2", default=None,
                        help="oppose two networks instead of NN vs hand-crafted")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--base", type=float, default=5.0, help="seconds per side")
    parser.add_argument("--inc", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=0,
                        help="fixed-depth match: ignore the clock, give both "
                             "sides this depth, isolating eval quality from speed")
    args = parser.parse_args()

    if args.depth:
        print(f"FIXED DEPTH {args.depth} for both sides: no clock, so this measures "
              f"evaluation quality with the speed handicap removed. It is a "
              f"diagnostic, not a competition result.")
    else:
        print(f"clock: {args.base}s + {args.inc}s/move  (NOT the competition control "
              f"unless it matches the verified configuration)")
    print("starting engines ...")
    nn = Worker(args.nn, f"NN({args.nn})", args.depth)
    hc = Worker(args.nn2, f"NN({args.nn2})" if args.nn2 else "hand-crafted", args.depth)

    score = {"nn": 0.0, "hc": 0.0}
    issues = []
    pairs = (args.games + 1) // 2
    for pair in range(pairs):
        opening = OPENINGS[pair % len(OPENINGS)]
        for nn_is_white in (True, False):
            if score["nn"] + score["hc"] >= args.games:
                break
            white, black = (nn, hc) if nn_is_white else (hc, nn)
            result, notes = play(white, black, opening, args.base, args.inc)
            if result == "1-0":
                won = "nn" if nn_is_white else "hc"
            elif result == "0-1":
                won = "hc" if nn_is_white else "nn"
            else:
                won = None
            if won:
                score[won] += 1.0
            else:
                score["nn"] += 0.5
                score["hc"] += 0.5
            colour = "W" if nn_is_white else "B"
            print(f"  [{opening:24}] NN as {colour}: {result:7} "
                  f"{'; '.join(notes) if notes else ''}")
            issues.extend(notes)

    nn.close()
    hc.close()
    played = score["nn"] + score["hc"]
    rate = score["nn"] / played
    # Binomial standard error on the score rate. With a few dozen games this is
    # wide, and it is the reason a small match can only separate "clearly worse"
    # from "clearly better".
    stderr = (rate * (1.0 - rate) / played) ** 0.5
    opponent = args.nn2 if args.nn2 else "hand-crafted"
    print(f"\n{args.nn} {score['nn']} - {score['hc']} {opponent}  ({played:.0f} games)")
    print(f"score rate: {rate * 100:.1f}% +- {stderr * 100:.1f}% (1 s.e.), "
          f"95% CI roughly [{max(0.0, rate - 2 * stderr) * 100:.0f}%, "
          f"{min(1.0, rate + 2 * stderr) * 100:.0f}%]")
    flags = [i for i in issues if "FLAG" in i or "ILLEGAL" in i or "crash" in i]
    print(f"illegal moves / flags / crashes: {len(flags)}"
          + (f" -> {flags}" if flags else ""))
