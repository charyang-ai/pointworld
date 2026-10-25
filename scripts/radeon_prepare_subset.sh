#!/bin/bash
# Container side. Turns a restored subset of the released H5 data into the WDS layout
# eval.py reads, keeping the paper's test assignment for the clips that are present.
#
#   bash scripts/radeon_prepare_subset.sh droid \
#     /workspace/data/droid_restored/droid/flows-fs-optimized /workspace/data/droid/wds \
#     /workspace/data/droid_restored/droid/confidence/expert_confidence-seed=42.h5
#   bash scripts/radeon_prepare_subset.sh behavior \
#     /workspace/data/behavior_restored/behavior/flows /workspace/data/behavior/wds
#
# DATA_SRC is a checkout of the upstream `data` branch, used unchanged.
set -euo pipefail
DOMAIN=${1:?usage: radeon_prepare_subset.sh droid|behavior FLOWS_DIR WDS_DIR [CONFIDENCE_H5]}
FLOWS=${2:?restored flows dir}
WDS=${3:?output WDS dir}
CONFIDENCE=${4:-}
PY=${PY:-/workspace/venv/bin/python}
DATA_SRC=${DATA_SRC:-/workspace/data_src}
HERE=$(cd "$(dirname "$0")" && pwd)

cd "$DATA_SRC"
"$PY" data_integrity_check.py --input_dir "$FLOWS" --domain "$DOMAIN" \
  --num_mp_workers 16 --output_file "$FLOWS/integrity_check.json"
"$PY" "$HERE/make_paper_subset_manifest.py" --domain "$DOMAIN" \
  --integrity_check_file "$FLOWS/integrity_check.json" \
  --paper_manifest "manifests/${DOMAIN}_paper_split_manifest.json" \
  --output_manifest "$FLOWS/paper_subset_manifest.json"
# Evaluation reads only the test split; keep a sliver of train so convert_wds.py accepts the split.
"$PY" convert_wds.py --input_dir "$FLOWS" --output_dir "$WDS" --domain "$DOMAIN" \
  --manifest "$FLOWS/paper_subset_manifest.json" --train_frac 0.02
if [ -n "$CONFIDENCE" ]; then
  # Filtered DROID metrics read the released confidence file from the test split.
  ln -f "$CONFIDENCE" "$WDS/test/expert_confidence-seed=42.h5"
fi
ls -la "$WDS/train" "$WDS/test"
