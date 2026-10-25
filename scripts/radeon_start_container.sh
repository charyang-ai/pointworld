#!/bin/bash
# Host side. Creates the single-GPU container used for the validation run.
#
#   bash scripts/radeon_start_container.sh /path/to/workdir [/dev/dri/renderD129]
#
# The workdir is mounted at /workspace; clone this repo to <workdir>/src first.
# Only one render node is passed through, so the container sees exactly one GPU.
# Find the node for a given GPU with `ls -l /dev/dri/by-path` and `rocm-smi --showbus`.
# Groups are added by number: the image's /etc/group has no "render" entry.
# Core dumps are off: a crashed process with HIP initialised dumps tens of GB to the host.
set -euo pipefail
HOST_DIR=${1:?usage: radeon_start_container.sh HOST_DIR [RENDER_NODE]}
RENDER_NODE=${2:-/dev/dri/renderD128}
NAME=${NAME:-radeon-pointworld}
IMAGE=${IMAGE:-rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0}
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)
mkdir -p "$HOST_DIR/weights" "$HOST_DIR/data" "$HOST_DIR/results"

docker run -d --name "$NAME" \
  --entrypoint bash \
  --device=/dev/kfd \
  --device="$RENDER_NODE" \
  --group-add "$VIDEO_GID" \
  --group-add "$RENDER_GID" \
  --ipc=host \
  --shm-size=32g \
  --ulimit core=0 \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -e ROCR_VISIBLE_DEVICES=0 \
  -v "$HOST_DIR":/workspace \
  -w /workspace/src \
  "$IMAGE" \
  -lc 'sleep infinity'

docker exec "$NAME" /opt/venv/bin/python -c \
  'import torch; print("torch", torch.__version__, "hip", torch.version.hip, "devices", torch.cuda.device_count(), torch.cuda.get_device_name(0))'
