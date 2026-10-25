#!/bin/bash
# Container side. Installs everything except PyTorch into /workspace/venv.
#
# The image keeps its ROCm PyTorch in /opt/venv. A venv created from it does not
# inherit /opt/venv's site-packages, so a .pth file adds them. pip then counts the
# ROCm torch as installed and does not pull a CUDA build from PyPI.
#
#   docker exec radeon-pointworld bash /workspace/src/scripts/radeon_install_deps.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
INDEX=${PIP_INDEX_URL:-https://pypi.org/simple}
VENV=${VENV:-/workspace/venv}
SRC=${SRC:-/workspace/src}

apt-get update
apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 python3-venv

if [ ! -x "$VENV/bin/python" ]; then
  /opt/venv/bin/python -m venv "$VENV"
fi
SITE=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
OPT_SITE=$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$OPT_SITE" > "$SITE/zz_rocm_torch_from_image.pth"

PIP=("$VENV/bin/python" -m pip --no-cache-dir)
# torch 2.11 in the image declares setuptools<82.
"${PIP[@]}" install -i "$INDEX" -U pip "setuptools<82" wheel
"${PIP[@]}" install -i "$INDEX" -r "$SRC/environments/requirements-rocm.txt"
"${PIP[@]}" install -i "$INDEX" --no-deps urdfpy==0.0.22 timm==1.0.19

"$VENV/bin/python" - << 'PY'
import numpy, torch
assert torch.version.hip, f"expected ROCm torch, got {torch.__version__} from {torch.__file__}"
t = torch.from_numpy(numpy.arange(6, dtype=numpy.float32)).cuda() * 2
assert t.cpu().numpy().tolist() == [0, 2, 4, 6, 8, 10]
import cv2, h5py, numba, pytorch_kinematics, scipy, timm, torchmetrics, trimesh, urdfpy, wandb, webdataset
print("torch", torch.__version__, "hip", torch.version.hip, "numpy", numpy.__version__)
print("device", torch.cuda.get_device_name(0), "count", torch.cuda.device_count())
print("DEPS_OK")
PY
