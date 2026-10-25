"""Time one PointWorld forward pass (DINOv3 + PTv3 + heads) per eval sample on the GPU.

Takes the eval.py arguments. Batches are moved to the GPU before the clock starts, so
data loading is excluded. The first $WARMUP batches (default 5) are dropped: they include
torch.compile and kernel selection.

    python tests/forward_latency.py --model_path ... --domains=droid --data_dirs=... \
        --batch_size=1 --eval_num_batches=60
"""

import numba  # noqa: F401  (must load before torch on ROCm, see eval.py)

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from arguments import parse_args  # noqa: E402
from evaluation.tester import Tester  # noqa: E402

if __name__ == "__main__":
    args = parse_args()
    warmup = int(os.environ.get("WARMUP", "5"))
    tester = Tester(args)
    loader, _ = tester._build_eval_loader("test", enable_mask=False)
    times, points = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if 0 < args.eval_num_batches <= i:
                break
            batch = {k: v.to(tester.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            torch.cuda.synchronize()
            start = time.perf_counter()
            tester.model(batch, training=False)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - start)
                points.append(int(batch["scene_exists"][:, 0].sum()) + int(batch["robot_exists"].sum()))
    times.sort()
    result = {
        "model_path": args.model_path,
        "device": torch.cuda.get_device_name(0),
        "samples": len(times),
        "median_ms": 1000 * statistics.median(times),
        "p90_ms": 1000 * times[int(0.9 * (len(times) - 1))],
        "mean_points_per_sample": statistics.mean(points),
    }
    if os.environ.get("LATENCY_OUT"):
        with open(os.environ["LATENCY_OUT"], "w") as f:
            json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
