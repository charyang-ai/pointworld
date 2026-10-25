"""Restrict a released paper-split manifest to the clips of a locally restored subset.

convert_wds.py (data branch) requires every manifest test clip to exist locally, so the
full release manifest fails on a subset. This keeps the paper's test assignment for the
clips that are present. DROID files are named <uuid>_flows.h5, so local clip IDs carry a
"_flows" suffix that the release manifest does not; main's confidence lookup strips it.

    python scripts/make_paper_subset_manifest.py --domain droid \
        --integrity_check_file /data/droid/flows-fs-optimized/integrity_check.json \
        --paper_manifest /path/to/data-branch/manifests/droid_paper_split_manifest.json \
        --output_manifest /data/droid_subset_manifest.json
"""

import argparse
import json
import os
import re

MANIFEST_SCHEMA_VERSION = "wds_manifest.v1"


def local_uuid(domain, h5_path):
    if domain == "behavior":
        match = re.search(r"(task-\d+)/(episode_\d+)\.(h5|hdf5)$", h5_path.replace("\\", "/"))
        if not match:
            raise ValueError(f"Unexpected BEHAVIOR path: {h5_path}")
        return f"{match.group(1)}_{match.group(2)}"
    return os.path.splitext(os.path.basename(h5_path))[0]


def release_uuid(uuid):
    return uuid[: -len("_flows")] if uuid.endswith("_flows") else uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--domain", required=True, choices=["droid", "behavior"])
    parser.add_argument("--integrity_check_file", required=True)
    parser.add_argument("--paper_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    args = parser.parse_args()

    with open(args.integrity_check_file, encoding="utf-8") as f:
        valid_clips = json.load(f)["valid_clips"]
    with open(args.paper_manifest, encoding="utf-8") as f:
        paper = json.load(f)
    if paper.get("domain") != args.domain:
        raise ValueError(f"paper manifest domain is {paper.get('domain')!r}, expected {args.domain!r}")
    paper_test = set(paper["test_clip_keys"])

    local_ids = sorted({f"{local_uuid(args.domain, p)}-{clip}" for p, clip in valid_clips})
    test_ids = []
    for clip_id in local_ids:
        uuid, _, clip = clip_id.rpartition("-")
        if f"{release_uuid(uuid)}-{clip}" in paper_test:
            test_ids.append(clip_id)
    if not test_ids:
        raise SystemExit("No local clip is in the paper test split.")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "domain": args.domain,
        "seed": paper.get("seed"),
        "test_percentage": paper.get("test_percentage"),
        "filters": {"tasks": [], "only_uuid_keywords": [], "max_clips": -1},
        "stats": {"num_selected_total": len(local_ids), "num_selected_test": len(test_ids)},
        "source_manifest": os.path.basename(args.paper_manifest),
        "test_clip_keys": test_ids,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_manifest)), exist_ok=True)
    with open(args.output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"local_clips={len(local_ids)} paper_test_clips_present={len(test_ids)} -> {args.output_manifest}")


if __name__ == "__main__":
    main()
