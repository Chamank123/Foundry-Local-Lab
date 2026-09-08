"""Third sweep: sweep2 showed the plateau (best epochs 81-248 of 400-600), so
stop lengthening and test cosine decay around the best configuration."""
import json, sys
from sweep import run, constant_baseline

grid = [
    (120, 256, 3e-3, "cosine"),
    (150, 256, 3e-3, "cosine"),
    (100, 512, 3e-3, "cosine"),
    (150, 512, 5e-3, "cosine"),
]
reference = constant_baseline()
print(f"constant-predictor validation RMSE: {reference:.5f}", flush=True)
print(f"{'epochs':>7} {'batch':>6} {'lr':>7} {'sched':>7} | {'best val RMSE':>13} "
      f"{'@epoch':>7} {'vs constant':>12} {'secs':>6}", flush=True)
results = []
for epochs, bs, lr, sched in grid:
    best, at, secs = run(epochs, bs, lr, lr_schedule=sched)
    results.append({"epochs": epochs, "batch_size": bs, "learning_rate": lr,
                    "lr_schedule": sched, "best_val_rmse": best,
                    "best_epoch": at, "seconds": secs})
    print(f"{epochs:7d} {bs:6d} {lr:7.4f} {sched:>7} | {best:13.5f} {at:7d} "
          f"{(reference-best)/reference*100:11.1f}% {secs:6.0f}", flush=True)
results.sort(key=lambda r: r["best_val_rmse"])
print("\nbest on validation:", json.dumps(results[0], indent=2))
json.dump({"constant_baseline_val_rmse": reference, "results": results},
          open("sweep3_results.json", "w"), indent=2)
