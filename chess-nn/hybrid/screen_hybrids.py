"""Small deterministic screen; reuses two workers to avoid repeated compilation.

These games are development diagnostics, not the fresh-process competition gate.
"""
import argparse
import json
from pathlib import Path
from experiment_matches import Worker, play
from match import OPENINGS


def main(a):
    out=Path(a.out)
    if out.exists():
        raise ValueError("Use a new output directory")
    out.mkdir(parents=True)
    full=str(Path("net_8m_scale400_cos24.npz").resolve())
    profiles=[("legacy_nn",full,1,"raw"),("legal_ep_nn",full,1,"legal"),
              ("blend10",full,.1,"legal"),("blend25",full,.25,"legal")]
    if a.residual:
        residual=str(Path(a.residual).resolve())
        profiles += [("residual_full",residual,1,"legal"),("residual25",residual,.25,"legal")]
    workers=[]
    try:
        workers.append(Worker({"cpu":a.cpu},"candidate",out/"candidate.log"))
        print("Candidate started",workers[0].ready,flush=True)
        workers.append(Worker({"cpu":a.cpu},"baseline",out/"baseline.log"))
        print("Baseline started",workers[1].ready,flush=True)
        for name,weights,weight,ep in profiles:
            workers[0].call({"load":{"weights":weights,"weight":weight,"ep":ep}},90)
            rows=[]
            for i in range(a.games):
                row,game=play(workers,i%2==0,OPENINGS[(a.opening_offset+i//2)%len(OPENINGS)],a)
                rows.append(row)
                with (out/f"{name}.pgn").open("a") as f:
                    f.write(str(game)+"\n\n")
                result={"name":name,"weights":weights,"weight":weight,"ep":ep,
                        "arguments":vars(a),"score":sum(x["score"] for x in rows),
                        "played":len(rows),"issues":sum(x["issue"] is not None for x in rows),"games":rows}
                (out/f"{name}.json").write_text(json.dumps(result,indent=2))
                print(f"{name} game {i+1}/{a.games}: {row['result']}, score={result['score']}, issue={row['issue']}",flush=True)
                if row["issue"]:
                    raise RuntimeError("Screen encountered an engine failure; inspect logs")
    finally:
        for w in workers:
            w.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    p.add_argument("--residual")
    p.add_argument("--cpu",type=int,default=0)
    p.add_argument("--games",type=int,default=8)
    p.add_argument("--opening-offset",type=int,default=0)
    p.add_argument("--depth",type=int,default=4)
    p.add_argument("--base",type=float,default=120)
    p.add_argument("--inc",type=float,default=.5)
    p.add_argument("--max-plies",type=int,default=600)
    a=p.parse_args()
    main(a)
