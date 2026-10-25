"""Rank DROID flow shards by how many paper-split test clips they contain.

_shards_manifest.json comes from nvidia/PointWorld-DROID (droid/flows-fs-optimized/),
the paper manifest from the upstream data branch (manifests/).

python scripts/rank_droid_shards.py <_shards_manifest.json> <droid_paper_split_manifest.json> [top_k]
"""
import collections
import json
import sys

shards_path, paper_path = sys.argv[1], sys.argv[2]
top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 15

shards = json.load(open(shards_path, encoding="utf-8"))["shards"]
paper = json.load(open(paper_path, encoding="utf-8"))

test_by_uuid = collections.Counter()
for clip_id in paper["test_clip_keys"]:
    uuid, _, _clip = clip_id.rpartition("-")
    test_by_uuid[uuid] += 1

rows = []
for shard in shards:
    uuids = [
        m.rsplit("/", 1)[-1][: -len("_flows.h5")] for m in shard["members"] if m.endswith("_flows.h5")
    ]
    n_test = sum(test_by_uuid.get(u, 0) for u in uuids)
    n_files = sum(1 for u in uuids if test_by_uuid.get(u, 0) > 0)
    size = sum(p["size_bytes"] for p in shard["parts"])
    rows.append((n_test, n_files, len(uuids), shard["shard_id"], size))

print(f"paper test clips: {sum(test_by_uuid.values())} in {len(test_by_uuid)} episodes")
for n_test, n_files, n_members, shard_id, size in sorted(rows, reverse=True)[:top_k]:
    print(f"{shard_id} test_clips={n_test:4d} episodes_with_test={n_files:3d}/{n_members:3d} {size / 1e9:.2f} GB")
