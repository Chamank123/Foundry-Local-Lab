"""Second sweep: every configuration in sweep.py was still improving at its
final epoch, so extend training length. Validation-only selection, as before."""
import json, sys
from sweep import run, constant_baseline

grid = [
    (600, 1024, 3e-3),
    (400,  256, 1e-3),
    (400,  256, 3e-3),
    (600,  512, 3e-3),
]
reference = constant_baseline()
print(f"constant-predictor validation RMSE: {reference:.5f}")
print(f"{'epochs':>7} {'batch':>6} {'lr':>7} | {'best val RMSE':>13} {'@epoch':>7} "
      f"{'vs constant':>12} {'secs':>6}")
results = []
for epochs, bs, lr in grid:
    best, at, secs = run(epochs, bs, lr)
    results.append({"epochs": epochs, "batch_size": bs, "learning_rate": lr,
                    "best_val_rmse": best, "best_epoch": at, "seconds": secs})
    print(f"{epochs:7d} {bs:6d} {lr:7.4f} | {best:13.5f} {at:7d} "
          f"{(reference-best)/reference*100:11.1f}% {secs:6.0f}")
    sys.stdout.flush()
results.sort(key=lambda r: r["best_val_rmse"])
print("\nbest on validation:", json.dumps(results[0], indent=2))
json.dump({"constant_baseline_val_rmse": reference, "results": results},
          open("sweep2_results.json", "w"), indent=2)
