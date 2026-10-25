"""
Plain-PyTorch stand-in for the parts of spconv that PTv3 uses: SparseConvTensor,
SubMConv3d (kernel_size=3) and modules.is_spconv_module.

spconv ships CUDA-only kernels. This module runs on ROCm, CUDA and CPU, and keeps
spconv's KRSC weight layout (out_channels, k0, k1, k2, in_channels) so released
checkpoints load unchanged. Kernel tap k = k0 * 9 + k1 * 3 + k2 reads the input at
indices[:, 1:] + (k0 - 1, k1 - 1, k2 - 1).
"""

import os
from types import SimpleNamespace

import torch
import torch.nn as nn

# PointWorld does not voxel-deduplicate its points, so several points can share a
# voxel. spconv's output then depends on the algorithm (both keep the point itself
# for the centre tap):
#   cuda       MaskImplicitGemm, spconv's default on CUDA and what the released
#              checkpoints were trained with. One point per voxel is the hash
#              representative; the others only receive taps 0..12.
#   cuda_first same, but the lowest index is the representative (CUDA picks
#              whichever duplicate wins a write race; "cuda" uses the highest).
#   native     the CPU build. Representatives sum taps 0..12 over all duplicates
#              of a neighbouring voxel; every point receives taps 14..26.
#   full       every point receives all 27 taps from representatives.
SUBM_MODE = os.environ.get("POINTWORLD_SUBM_MODE", "cuda")
_MODES = ("cuda", "cuda_first", "native", "full")

_CENTER = 13
_OFFSETS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, indice_dict=None):
        self.features = features
        self.indices = indices
        self.spatial_shape = [int(s) for s in spatial_shape]
        self.batch_size = int(batch_size)
        # Neighbour tables keyed by indice_key, shared by tensors with the same indices.
        self.indice_dict = {} if indice_dict is None else indice_dict

    def replace_feature(self, features):
        return SparseConvTensor(
            features, self.indices, self.spatial_shape, self.batch_size, self.indice_dict
        )


@torch.no_grad()
def build_neighbor_table(indices, spatial_shape, mode=None):
    """(27, N) table of input rows per output row and tap, -1 where there is none."""
    mode = mode or SUBM_MODE
    if mode not in _MODES:
        raise ValueError(f"POINTWORLD_SUBM_MODE must be one of {_MODES}, got {mode!r}")
    device = indices.device
    n = indices.shape[0]
    batch = indices[:, 0].long()
    coord = indices[:, 1:].long()
    shape = torch.tensor(spatial_shape, device=device, dtype=torch.long)

    def encode(c):
        return ((batch * shape[0] + c[:, 0]) * shape[1] + c[:, 1]) * shape[2] + c[:, 2]

    sorted_keys, perm = torch.sort(encode(coord), stable=True)
    voxel_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    ends = torch.cumsum(counts, dim=0)
    first_wins = mode in ("cuda_first", "native")
    representative = perm[ends - counts] if first_wins else perm[ends - 1]

    table = torch.full((27, n), -1, dtype=torch.long, device=device)
    for k, offset in enumerate(_OFFSETS):
        if k == _CENTER:
            continue
        nb = coord + torch.tensor(offset, device=device)
        inside = ((nb >= 0) & (nb < shape)).all(dim=1)
        nb_keys = encode(nb)
        pos = torch.searchsorted(voxel_keys, nb_keys).clamp(max=voxel_keys.numel() - 1)
        found = inside & (voxel_keys[pos] == nb_keys)
        table[k] = torch.where(found, representative[pos], table[k])
    table[_CENTER] = torch.arange(n, device=device)

    if mode in ("cuda", "cuda_first"):
        voxel_of_point = torch.empty(n, dtype=torch.long, device=device)
        voxel_of_point[perm] = torch.repeat_interleave(
            torch.arange(voxel_keys.numel(), device=device), counts
        )
        is_rep = representative[voxel_of_point] == torch.arange(n, device=device)
        table[_CENTER + 1:, ~is_rep] = -1
    return table


class SubMConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, indice_key=None, mode=None):
        super().__init__()
        if kernel_size != 3:
            raise ValueError("SubMConv3d here only implements kernel_size=3")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.mode = mode
        self.weight = nn.Parameter(torch.empty(out_channels, 3, 3, 3, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.trunc_normal_(self.weight, std=0.02)

    def _taps(self, x):
        """Per tap: (k, rows, idx, scatter). Gather taps add feat[idx] into out[rows];
        scatter taps (native mode, k < 13) add feat[rows] into out[idx]."""
        mode = self.mode or SUBM_MODE
        key = (self.indice_key, mode)
        if self.indice_key is not None and key in x.indice_dict:
            return x.indice_dict[key]
        table = build_neighbor_table(x.indices, x.spatial_shape, mode)
        taps = [k for k in range(27) if k != _CENTER]
        scatter = [mode == "native" and k < _CENTER for k in taps]
        lookup = torch.stack([table[26 - k] if s else table[k] for k, s in zip(taps, scatter)])
        valid = lookup >= 0
        # Two host syncs per table instead of one per tap and call.
        counts = valid.sum(dim=1).tolist()
        tap, rows = valid.nonzero(as_tuple=True)
        idx = lookup[tap, rows]
        entry = list(zip(taps, rows.split(counts), idx.split(counts), scatter))
        if self.indice_key is not None:
            x.indice_dict[key] = entry
        return entry

    def forward(self, x):
        feat = x.features
        weight = self.weight.reshape(self.out_channels, 27, self.in_channels).to(feat.dtype)
        out = feat @ weight[:, _CENTER].t()
        for k, rows, idx, scatter in self._taps(x):
            if scatter:
                out.index_add_(0, idx, feat[rows] @ weight[:, k].t())
            else:
                out[rows] += feat[idx] @ weight[:, k].t()
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return x.replace_feature(out)


def is_spconv_module(module):
    return isinstance(module, SubMConv3d)


modules = SimpleNamespace(is_spconv_module=is_spconv_module)
