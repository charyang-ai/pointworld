"""Check ptv3/sparse_conv.py, ptv3/segment.py and ptv3/varlen_attention.py.

SubMConv3d is compared with spconv 2.3.8's CPU build. spconv only runs Native on CPU,
so the "native" mode must match it with or without duplicate coordinates, and every
mode must match it when coordinates are unique.

    PYTHONPATH=/path/to/spconv-cpu python tests/check_sparse_ops.py
"""

import importlib.util
import json
import os
import sys

if os.environ.get("OMP_NUM_THREADS") != "1":
    # spconv's CPU scatter-add is not atomic: with repeated coordinates, threads lose updates.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch  # noqa: E402

PTV3_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ptv3")


def load(name):
    # By file path, so the check needs torch and spconv only (ptv3/__init__ pulls in timm, addict).
    spec = importlib.util.spec_from_file_location(name, os.path.join(PTV3_DIR, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sparse_conv = load("sparse_conv")
segment_csr = load("segment").segment_csr
flash_attn_varlen_qkvpacked_func = load("varlen_attention").flash_attn_varlen_qkvpacked_func

results = []


def record(name, value, tol):
    ok = bool(value <= tol)
    results.append({"check": name, "max_abs_diff": float(value), "tol": tol, "pass": ok})
    print(f"{'PASS' if ok else 'FAIL'} {name}: {value:.3e} (tol {tol:.0e})")


def random_cloud(n, grid, batch_size, channels, seed):
    g = torch.Generator().manual_seed(seed)
    coord = torch.randint(0, grid, (n, 3), generator=g)
    batch = torch.randint(0, batch_size, (n,), generator=g).sort().values
    indices = torch.cat([batch[:, None], coord], dim=1).int()
    feats = torch.randn(n, channels, generator=g)
    return indices, feats


def unique_rows(indices):
    _, first = torch.unique(indices.long(), dim=0, return_inverse=True)
    keep = torch.zeros(indices.shape[0], dtype=torch.bool)
    seen = set()
    for i, v in enumerate(first.tolist()):
        if v not in seen:
            seen.add(v)
            keep[i] = True
    return keep


def check_subm():
    import spconv.pytorch as spconv_ref

    torch.manual_seed(0)
    channels = 24
    ours = sparse_conv.SubMConv3d(channels, channels, 3, bias=True)
    torch.nn.init.normal_(ours.bias)
    ref = spconv_ref.SubMConv3d(channels, channels, 3, bias=True)
    assert tuple(ref.weight.shape) == tuple(ours.weight.shape), (ref.weight.shape, ours.weight.shape)
    ref.weight.data.copy_(ours.weight.data)
    ref.bias.data.copy_(ours.bias.data)

    for label, n, grid in (("unique", 4000, 64), ("duplicates", 6000, 14), ("duplicates", 400, 6)):
        indices, feats = random_cloud(n, grid, 2, channels, seed=1 if label == "unique" else 2)
        if label == "unique":
            keep = unique_rows(indices)
            indices, feats = indices[keep], feats[keep]
        dup_frac = 1.0 - torch.unique(indices.long(), dim=0).shape[0] / indices.shape[0]
        shape = (indices[:, 1:].max(0).values + 96).tolist()
        x_ref = spconv_ref.SparseConvTensor(feats, indices, shape, 2)
        y_ref = ref(x_ref).features
        scale = y_ref.abs().max().item()
        modes = ("cuda", "cuda_first", "native", "full") if label == "unique" else ("native",)
        for mode in modes:
            ours.mode = mode
            y = ours(sparse_conv.SparseConvTensor(feats, indices, shape, 2)).features
            record(f"subm[{label}, dup={dup_frac:.2f}] {mode} vs spconv-cpu", (y - y_ref).abs().max().item() / scale, 1e-5)
        if label == "duplicates":
            ours.mode = "cuda"
            y = ours(sparse_conv.SparseConvTensor(feats, indices, shape, 2)).features
            print(f"info: duplicates make cuda and native differ by {(y - y_ref).abs().max().item() / scale:.3e} (expected > 0)")


def check_tap_order():
    # One point at the centre, one neighbour per tap: tap k must read neighbour k.
    centre = torch.tensor([5, 5, 5])
    coords = [centre.tolist()] + [(centre + torch.tensor(o)).tolist() for o in sparse_conv._OFFSETS if o != (0, 0, 0)]
    indices = torch.tensor([[0] + c for c in coords], dtype=torch.int32)
    feats = torch.arange(len(coords), dtype=torch.float64)[:, None]
    conv = sparse_conv.SubMConv3d(1, 27, 3, bias=False).double()
    with torch.no_grad():
        conv.weight.zero_()
        for k in range(27):
            conv.weight[k].view(27)[k] = 1.0
    y = conv(sparse_conv.SparseConvTensor(feats, indices, [16, 16, 16], 1)).features[0]
    expected = []
    neighbour = 1
    for o in sparse_conv._OFFSETS:
        if o == (0, 0, 0):
            expected.append(0.0)
        else:
            expected.append(float(neighbour))
            neighbour += 1
    record("subm tap k reads indices + offset_k", (y - torch.tensor(expected, dtype=torch.float64)).abs().max().item(), 0.0)


def check_segment():
    g = torch.Generator().manual_seed(3)
    counts = torch.randint(1, 9, (200,), generator=g)
    indptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    src = torch.randn(int(indptr[-1]), 16, generator=g, dtype=torch.float64)
    for reduce in ("sum", "mean", "max", "min"):
        ref = torch.stack([
            getattr(src[a:b], {"sum": "sum", "mean": "mean", "max": "amax", "min": "amin"}[reduce])(0)
            for a, b in zip(indptr[:-1].tolist(), indptr[1:].tolist())
        ])
        record(f"segment_csr {reduce}", (segment_csr(src, indptr, reduce) - ref).abs().max().item(), 1e-12)


def check_attention(device):
    g = torch.Generator().manual_seed(4)
    heads, head_dim, patch = 4, 32, 64
    lengths = [patch] * 5 + [17] + [patch] * 3 + [40]
    cu = torch.tensor([0] + lengths).cumsum(0).to(torch.int32)
    qkv = torch.randn(int(cu[-1]), 3, heads, head_dim, generator=g).to(device)
    scale = head_dim ** -0.5
    ref = torch.empty(qkv.shape[0], heads, head_dim, device=device)
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        q, k, v = qkv[a:b].float().permute(1, 2, 0, 3)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
        ref[a:b] = (attn @ v).transpose(0, 1)
    out = flash_attn_varlen_qkvpacked_func(qkv, cu.to(device), patch, 0.0, scale)
    record(f"varlen attention fp32 [{device}]", (out - ref).abs().max().item(), 1e-5)
    out_bf16 = flash_attn_varlen_qkvpacked_func(qkv.bfloat16(), cu.to(device), patch, 0.0, scale)
    record(f"varlen attention bf16 [{device}]", (out_bf16.float() - ref).abs().max().item(), 3e-2)


if __name__ == "__main__":
    check_tap_order()
    check_subm()
    check_segment()
    check_attention("cpu")
    if torch.cuda.is_available():
        check_attention("cuda")
    passed = all(r["pass"] for r in results)
    print(json.dumps({"passed": passed, "results": results}, indent=1))
    sys.exit(0 if passed else 1)
