"""Paired-opening comparisons with real response deadlines and replayable PGNs.

Clock matches default to fresh processes each game, 600 plies and one pinned CPU.
Fixed-depth screens may reuse warmed workers; they are diagnostics only.
"""
import argparse
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import chess
import chess.pgn
from match import OPENINGS


class Worker:
    def __init__(self, spec, label, log):
        self.label = label
        self.stderr = log.open("a")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",NUMBA_NUM_THREADS="1")
        self.p = subprocess.Popen([sys.executable,"-u","hybrid_worker.py",json.dumps(spec)],
                                  stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
                                  text=True,bufsize=1,env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.p.stdout, selectors.EVENT_READ)
        try:
            self.ready = self.read(90)
            if not self.ready.get("ready"):
                raise RuntimeError("Bad startup message")
        except BaseException:
            self.close()
            raise

    def read(self, seconds):
        if not self.selector.select(max(.01,seconds)):
            raise TimeoutError(f"{self.label} response deadline")
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.label} exited: {self.p.poll()}")
        return json.loads(line)

    def call(self, request, seconds):
        self.p.stdin.write(json.dumps(request)+"\n")
        self.p.stdin.flush()
        return self.read(seconds)

    def close(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.selector.close()
        self.stderr.close()


def play(workers, candidate_white, opening, a):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    for uci in opening.split():
        move = chess.Move.from_uci(uci)
        board.push(move)
        node = node.add_variation(move)
    game.headers["White"] = "candidate" if candidate_white else "baseline"
    game.headers["Black"] = "baseline" if candidate_white else "candidate"
    game.headers["TimeControl"] = f"{a.base}+{a.inc}" if not a.depth else f"depth{a.depth}"
    for worker in workers:
        worker.call({"newgame":True},5)
    clocks = [a.base,a.base]
    moves, nodes, elapsed = [0,0], [0,0], [0.,0.]
    max_rss = [w.ready["rss_mb"] for w in workers]
    issue, result = None, None
    while not board.is_game_over(claim_draw=True) and board.ply() < a.max_plies:
        side = 0 if board.turn == candidate_white else 1
        start = time.perf_counter()
        try:
            reply = workers[side].call({"fen":board.fen(),"time_left_ms":int(clocks[side]*1000),
                                        "depth":a.depth}, 90 if a.depth else clocks[side]+.05)
        except (TimeoutError,RuntimeError,BrokenPipeError,ValueError) as e:
            issue = type(e).__name__+": "+str(e)
            result = "0-1" if board.turn else "1-0"
            break
        spent = time.perf_counter()-start
        clocks[side] -= spent
        elapsed[side] += spent
        nodes[side] += reply["nodes"]
        moves[side] += 1
        max_rss[side] = max(max_rss[side],reply["rss_mb"])
        if not a.depth and clocks[side] < 0:
            issue="flag"
            result="0-1" if board.turn else "1-0"
            break
        clocks[side] += a.inc
        try:
            move=chess.Move.from_uci(reply["move"])
            if move not in board.legal_moves:
                raise ValueError("illegal move")
        except ValueError:
            issue="illegal: "+reply["move"]
            result="0-1" if board.turn else "1-0"
            break
        board.push(move)
        node=node.add_variation(move)
    if result is None:
        result=board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    game.headers["Result"]=result
    game.headers["Termination"]=issue or ("ply_cap" if board.ply()>=a.max_plies else str(board.outcome(claim_draw=True).termination))
    score = .5 if result == "1/2-1/2" else float((result == "1-0") == candidate_white)
    row={"score":score,"result":result,"candidate_white":candidate_white,"opening":opening,
         "issue":issue,"plies":board.ply(),"moves":moves,"nodes":nodes,"thinking_s":elapsed,
         "startup_s":[w.ready["startup_s"] for w in workers],"peak_rss_mb":max_rss,
         "remaining_s":clocks}
    return row, game


def main(a):
    target=Path(a.out)
    if target.exists():
        raise ValueError("Use a fresh --out path")
    target.parent.mkdir(parents=True,exist_ok=True)
    candidate={"weights":str(Path(a.nn).resolve()),"weight":a.weight,"ep":a.ep,"cpu":a.cpu}
    baseline={"pristine":bool(a.pristine),"cpu":a.cpu}
    if a.pristine and a.depth:
        raise ValueError("Use integrated mode0 for fixed-depth screens; pristine for clock matches")
    rows=[]
    workers=[]
    try:
        for i in range(a.games):
            if not workers:
                workers=[Worker(candidate,"candidate",target.with_suffix(".candidate.log"))]
                workers.append(Worker(baseline,"baseline",target.with_suffix(".baseline.log")))
            row,game=play(workers,i%2==0,OPENINGS[(a.opening_offset+i//2)%len(OPENINGS)],a)
            rows.append(row)
            with target.with_suffix(".pgn").open("a") as f:
                f.write(str(game)+"\n\n")
            summary={"arguments":vars(a),"games":rows,"score":sum(r["score"] for r in rows),
                     "played":len(rows),"issues":sum(r["issue"] is not None for r in rows)}
            target.write_text(json.dumps(summary,indent=2))
            print(f"Game {i+1}/{a.games}: {row['result']}, candidate {'W' if i%2==0 else 'B'}, "
                  f"score {summary['score']}/{i+1}; issue={row['issue']}; plies={row['plies']}",flush=True)
            if not a.reuse or row["issue"]:
                for worker in workers:
                    worker.close()
                workers=[]
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nn",required=True)
    p.add_argument("--weight",type=float,default=1)
    p.add_argument("--ep",choices=("legal","raw"),default="legal")
    p.add_argument("--games",type=int,default=8)
    p.add_argument("--depth",type=int,default=0)
    p.add_argument("--base",type=float,default=120)
    p.add_argument("--inc",type=float,default=.5)
    p.add_argument("--max-plies",type=int,default=600)
    p.add_argument("--cpu",type=int,default=0)
    p.add_argument("--opening-offset",type=int,default=0)
    p.add_argument("--reuse",action="store_true",help="Reuse workers for screening only")
    p.add_argument("--pristine",action="store_true")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    if a.games<=0 or a.games%2 or a.base<=0 or a.inc<0 or a.depth<0:
        p.error("Require positive even games, positive base, nonnegative increment/depth")
    main(a)
