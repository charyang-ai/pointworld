#!/bin/bash
# Container side. Runs the upstream eval.py on the one visible GPU and keeps log and metrics.
#
#   bash scripts/radeon_eval.sh TAG CHECKPOINT DOMAIN WDS_DIR [extra eval.py args]
#   bash scripts/radeon_eval.sh small-droid /workspace/weights/small-droid/model-best.pt \
#       droid /workspace/data/droid/wds --confidence_thres=0.8
set -euo pipefail
TAG=${1:?tag}
CKPT=${2:?checkpoint}
DOMAIN=${3:?domain}
WDS=${4:?wds dir}
shift 4
PY=${PY:-/workspace/venv/bin/python}
OUT=${RESULTS:-/workspace/results}/$TAG
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

{
  echo "tag=$TAG checkpoint=$CKPT domain=$DOMAIN data=$WDS"
  echo "POINTWORLD_SUBM_MODE=${POINTWORLD_SUBM_MODE:-cuda} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"
  "$PY" -c 'import torch; print("torch", torch.__version__, "hip", torch.version.hip, "device", torch.cuda.get_device_name(0), "count", torch.cuda.device_count())'
  echo "start $(date -Is)"
} | tee "$OUT/eval.log"

start=$(date +%s)
set +e
"$PY" eval.py --model_path "$CKPT" --domains="$DOMAIN" --data_dirs="$WDS" \
  --batch_size=1 --eval_num_batches=-1 "$@" 2>&1 | tee -a "$OUT/eval.log"
status=${PIPESTATUS[0]}
set -e
echo "end $(date -Is) seconds=$(( $(date +%s) - start )) exit=$status" | tee -a "$OUT/eval.log"

metrics=$(grep -a -o 'eval_logs/[^ ]*/metrics.json' "$OUT/eval.log" | tail -n 1 || true)
if [ -n "$metrics" ] && [ -f "$metrics" ]; then
  cp "$metrics" "$OUT/metrics.json"
fi
exit "$status"
