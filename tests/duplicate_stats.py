"""Count PTv3 stage-0 points that share a voxel with another point, on real eval batches.

Takes the eval.py arguments and runs the evaluation forward pass with a pre-hook on the
first PTv3 block, where the SubMConv3d positional encoding sees the raw point cloud.
The JSON goes to $DUPSTATS_OUT (default duplicate_stats.json).

    DUPSTATS_OUT=duplicate_stats.json python tests/duplicate_stats.py --model_path ... \
        --domains=droid --data_dirs=... --batch_size=1 --eval_num_batches=-1
"""

import numba  # noqa: F401  (must load before torch on ROCm, see eval.py)

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from arguments import parse_args  # noqa: E402
from evaluation.tester import Tester  # noqa: E402

rows = []


def count(module, inputs):
    point = inputs[0]
    indices = point.sparse_conv_feat.indices.long()
    is_robot = point.is_robot.bool().view(-1)
    _, voxel, per_voxel = torch.unique(indices, dim=0, return_inverse=True, return_counts=True)
    shared = per_voxel[voxel] > 1
    robot_in_voxel = torch.zeros(per_voxel.numel(), dtype=torch.long, device=indices.device)
    robot_in_voxel.index_add_(0, voxel, is_robot.long())
    mixed = (robot_in_voxel > 0) & (robot_in_voxel < per_voxel)
    rows.append({
        "points": indices.shape[0],
        "voxels": per_voxel.numel(),
        "shared_points": int(shared.sum()),
        "scene_points": int((~is_robot).sum()),
        "scene_shared": int((shared & ~is_robot).sum()),
        "robot_points": int(is_robot.sum()),
        "robot_shared": int((shared & is_robot).sum()),
        "mixed_voxels": int(mixed.sum()),
        "max_per_voxel": int(per_voxel.max()),
    })


if __name__ == "__main__":
    args = parse_args()
    tester = Tester(args)
    tester.model.dynamics_predictor.predictor_model.enc.enc0.block0.register_forward_pre_hook(count)
    loader, _ = tester._build_eval_loader("test", enable_mask=False)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if 0 < args.eval_num_batches <= i:
                break
            batch = {k: v.to(tester.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            tester.model(batch, training=False)

    total = {k: sum(r[k] for r in rows) for k in rows[0] if k != "max_per_voxel"}
    summary = {
        "batches": len(rows),
        "points": total["points"],
        "shared_fraction": total["shared_points"] / total["points"],
        "non_representative_fraction": (total["points"] - total["voxels"]) / total["points"],
        "scene_shared_fraction": total["scene_shared"] / total["scene_points"],
        "robot_shared_fraction": total["robot_shared"] / total["robot_points"],
        "mixed_voxels_per_batch": total["mixed_voxels"] / len(rows),
        "max_points_in_one_voxel": max(r["max_per_voxel"] for r in rows),
    }
    with open(os.environ.get("DUPSTATS_OUT", "duplicate_stats.json"), "w") as f:
        json.dump({"summary": summary, "per_batch": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))
