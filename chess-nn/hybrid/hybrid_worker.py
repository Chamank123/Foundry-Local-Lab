"""Private test worker. stdout is protocol; source stays readable for review."""
import json
import os
from pathlib import Path
import resource
import sys
import time
import faulthandler

root = Path(__file__).resolve().parent
spec = json.loads(sys.argv[1])
if spec.get("cpu") is not None and spec["cpu"] >= 0 and hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, {int(spec["cpu"])})
sys.path.insert(0, str(root / ("engine_pristine" if spec.get("pristine") else "engine")))
started = time.perf_counter()
print("worker: importing engine",file=sys.stderr,flush=True)
faulthandler.dump_traceback_later(60, repeat=False)
import agent
import search_numba as engine
faulthandler.cancel_dump_traceback_later()
print(f"worker: import finished {time.perf_counter()-started:.2f}s",file=sys.stderr,flush=True)
if spec.get("weights"):
    engine.load_nn(spec["weights"], weight=spec.get("weight",1), ep_policy=spec.get("ep","legal"))
print(f"worker: load/warmup finished {time.perf_counter()-started:.2f}s",file=sys.stderr,flush=True)
from match_worker import fixed_depth_move
print(json.dumps({"ready":True,"startup_s":time.perf_counter()-started,
                  "rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}),flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("quit"):
        break
    if request.get("newgame"):
        engine.new_game()
        print(json.dumps({"ok":True}),flush=True)
        continue
    if request.get("load"):
        options=request["load"]
        engine.load_nn(options["weights"],weight=options.get("weight",1),ep_policy=options.get("ep","legal"))
        print(json.dumps({"ok":True}),flush=True)
        continue
    if request.get("depth"):
        # Fixed-depth helper expects the integrated search signature. The
        # pristine engine is used only in clock games and baseline parity.
        move = fixed_depth_move(engine, request["fen"], request["depth"])
    else:
        move = agent.get_move(request["fen"], request["time_left_ms"])
    print(json.dumps({"move":move,"nodes":int(engine._nodes[0]),
                      "rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}),flush=True)
