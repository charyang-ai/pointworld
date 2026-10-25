"""Plain-PyTorch segment_csr, standing in for torch_scatter.segment_csr in GridPooling."""

import torch

_REDUCE = {"sum": "sum", "mean": "mean", "max": "amax", "min": "amin"}


def segment_csr(src, indptr, reduce="sum"):
    """Reduce the rows src[indptr[i]:indptr[i + 1]] for each i; empty segments give 0."""
    counts = indptr[1:] - indptr[:-1]
    segment = torch.repeat_interleave(torch.arange(counts.numel(), device=src.device), counts)
    index = segment.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out = src.new_zeros((counts.numel(),) + tuple(src.shape[1:]))
    return out.scatter_reduce_(0, index, src, reduce=_REDUCE[reduce], include_self=False)
