"""Split one PointWorld forward pass into the DINOv3 scene encoder, PTv3, and the three
PTv3 pieces that stand in for CUDA-only libraries (SubMConv3d, varlen attention,
segment_csr). Wall clock with a device sync around each call, so the parts add a little
overhead; nested parts (SubMConv3d inside PTv3) are counted in both.

Takes the eval.py arguments; the first $WARMUP batches (default 5) are not timed.
$PROFILE_OUT, if set, receives the JSON.

    python tests/profile_forward.py --model_path ... --domains=droid --data_dirs=... \
        --batch_size=1 --eval_num_batches=25
"""

import numba  # noqa: F401  (must load before torch on ROCm, see eval.py)

import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import ptv3.ptv3 as ptv3_module  # noqa: E402
import scene_featurizer  # noqa: E402
from arguments import parse_args  # noqa: E402
from evaluation.tester import Tester  # noqa: E402
from ptv3 import segment, sparse_conv  # noqa: E402

totals = collections.defaultdict(float)
calls = collections.Counter()
timing = [False]


def timed(name, fn):
    def wrapper(*args, **kwargs):
        if not timing[0]:
            return fn(*args, **kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        totals[name] += time.perf_counter() - start
        calls[name] += 1
        return out
    return wrapper


scene_featurizer.SceneEncoder2D.forward = timed("scene encoder (DINOv3)", scene_featurizer.SceneEncoder2D.forward)
ptv3_module.PointTransformerV3.forward = timed("PTv3", ptv3_module.PointTransformerV3.forward)
sparse_conv.SubMConv3d.forward = timed("PTv3/SubMConv3d", sparse_conv.SubMConv3d.forward)
ptv3_module.flash_attn_varlen_qkvpacked_func = timed(
    "PTv3/varlen attention", ptv3_module.flash_attn_varlen_qkvpacked_func
)
segment.segment_csr = timed("PTv3/segment_csr", segment.segment_csr)

if __name__ == "__main__":
    args = parse_args()
    warmup = int(os.environ.get("WARMUP", "5"))
    tester = Tester(args)
    loader, _ = tester._build_eval_loader("test", enable_mask=False)
    samples = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if 0 < args.eval_num_batches <= i:
                break
            batch = {k: v.to(tester.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            timing[0] = i >= warmup
            torch.cuda.synchronize()
            start = time.perf_counter()
            tester.model(batch, training=False)
            torch.cuda.synchronize()
            if timing[0]:
                totals["forward"] += time.perf_counter() - start
                samples += 1
    result = {
        "model_path": args.model_path,
        "samples": samples,
        "ms_per_sample": {k: 1000 * v / samples for k, v in sorted(totals.items())},
        "calls_per_sample": {k: v / samples for k, v in sorted(calls.items())},
        "share_of_forward": {k: v / totals["forward"] for k, v in sorted(totals.items())},
    }
    if os.environ.get("PROFILE_OUT"):
        with open(os.environ["PROFILE_OUT"], "w") as f:
            json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
