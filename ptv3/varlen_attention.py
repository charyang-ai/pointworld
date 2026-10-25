"""scaled_dot_product_attention stand-in for flash_attn_varlen_qkvpacked_func."""

import torch
import torch.nn.functional as F


def flash_attn_varlen_qkvpacked_func(
    qkv, cu_seqlens, max_seqlen=None, dropout_p=0.0, softmax_scale=None, causal=False
):
    """qkv: (total, 3, H, D), split into independent sequences by cu_seqlens. Returns (total, H, D)."""
    _, _, heads, head_dim = qkv.shape
    starts = cu_seqlens[:-1].long()
    lengths = cu_seqlens[1:].long() - starts
    out = qkv.new_empty((qkv.shape[0], heads, head_dim))
    # PTv3 pads every sample to whole patches, so there are only a few distinct lengths.
    for length in torch.unique(lengths).tolist():
        if length == 0:
            continue
        group_starts = starts[lengths == length]
        rows = (group_starts[:, None] + torch.arange(length, device=qkv.device)).reshape(-1)
        q, k, v = qkv[rows].view(-1, length, 3, heads, head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
        )
        out[rows] = o.transpose(1, 2).reshape(-1, heads, head_dim)
    return out
