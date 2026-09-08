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
]


class Worker:
    def __init__(self, weights, label):
        self.label = label
        cmd = [sys.executable, "match_worker.py"] + ([weights] if weights else [])
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
        if clock[side] < 0:
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
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--base", type=float, default=5.0, help="seconds per side")
    parser.add_argument("--inc", type=float, default=0.1)
    args = parser.parse_args()

    print(f"clock: {args.base}s + {args.inc}s/move  (NOT the competition control "
          f"unless it matches the verified configuration)")
    print("starting engines ...")
    nn = Worker(args.nn, f"NN({args.nn})")
    hc = Worker(None, "hand-crafted")

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
    print(f"\nNN {score['nn']} - {score['hc']} hand-crafted  ({played:.0f} games)")
    print(f"NN score rate: {score['nn'] / played * 100:.0f}%")
    flags = [i for i in issues if "FLAG" in i or "ILLEGAL" in i or "crash" in i]
    print(f"illegal moves / flags / crashes: {len(flags)}"
          + (f" -> {flags}" if flags else ""))
