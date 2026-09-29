"""Partition attention for Apple GPUs, with an optional CUDA Flex backend.

The public layout is [KV batch/head, query, dim]. GQA and shared-state folding
belong to attention.py; neither backend replicates K/V. Both return natural-log
LSE with its gradient, required by T5Gemma2's joint self/cross softmax.
On measured M1 Max / M4 D=256 shapes, GEMM + fused softmax beats full fusion.
"""

from __future__ import annotations

import json
import platform
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

# Temporary scores are overwritten by probabilities and never saved by autograd.
METAL_SCORE_TILE_ELEMENTS = 4 * 1024 * 1024
# Prefer useful GEMM row counts before adding more head/window batches. Packing
# every window into the batch can otherwise leave only 32-64 query rows per GEMM.
METAL_QUERY_TILE_ROWS = 1024


@lru_cache(maxsize=1)
def apple_gpu_info():
    """Identify the chip before selecting a launch profile (once per process)."""
    if platform.system() != "Darwin":
        return {"chip": "unknown", "gpu": "unknown", "gpu_cores": None}
    chip = subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
    ).strip()
    info = {"chip": chip, "gpu": chip, "gpu_cores": None}
    try:
        result = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"], text=True, timeout=10
        )
        devices = json.loads(result)["SPDisplaysDataType"]
        gpu = next((d for d in devices if "Apple" in d.get("sppci_model", "")), {})
        info["gpu"] = gpu.get("sppci_model", chip)
        info["gpu_cores"] = int(gpu["sppci_cores"]) if "sppci_cores" in gpu else None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        # GPU inventory is advisory; a missing profiler must not stop inference.
        pass
    return info


def metal_profile(dim, *, keys=0):
    """Measured M1 Max / M4 profiles; unknown chips remain conservative."""
    info = apple_gpu_info()
    m1_max = (info["chip"], info["gpu_cores"], dim) == ("Apple M1 Max", 24, 256)
    measured = m1_max or (info["chip"], info["gpu_cores"], dim) == ("Apple M4", 10, 256)
    return {
        "algorithm": "gemm" if measured and keys <= 8192 else "fused",
        "simd_groups": 2 if m1_max else 1,
        "query_tile": 8,
        "key_tile": 32,
        "softmax_threads": 32 if keys <= 128 else 128 if keys <= 1024 else 256,
        "encoder_window_block": (512 if m1_max else 256)
        if measured and 4096 <= keys <= 8192
        else 0,
        "head_dim": dim,
    }


@lru_cache(maxsize=1)
def _qk_norm_rope_kernel():
    return torch.mps.compile_shader(
        Path(__file__).with_name("qk_norm_rope.metal").read_text()
    ).qk_norm_rope


def metal_qk_norm_rope(q, k, q_norm, k_norm, position_embeddings):
    """Fuse FP32 Gemma Q/K RMSNorm, RoPE and BLHD -> BHLD layout conversion.

    Return None for unsupported layouts/precision so the caller retains native
    operations. In particular, low precision rounds between norm and RoPE and
    must not silently use this FP32-only fusion.
    """
    cosine, sine = position_embeddings
    tensors = (q, k, q_norm.weight, k_norm.weight, cosine, sine)
    if (
        torch.is_grad_enabled()
        or q_norm.training
        or k_norm.training
        or not hasattr(torch.mps, "compile_shader")
        or not all(t.device.type == "mps" and t.dtype == torch.float32 for t in tensors)
        or not all(t.is_contiguous() for t in tensors)
        or q.ndim != 4
        or k.ndim != 4
        or q.shape[:2] != k.shape[:2]
        or q.shape[-1] != k.shape[-1]
        or q.shape[-1] not in (16, 32, 64, 128, 256)
        or not all(q.shape)
        or not all(k.shape)
        or q_norm.weight.shape != (q.shape[-1],)
        or k_norm.weight.shape != (q.shape[-1],)
        or cosine.shape != sine.shape
        or cosine.ndim != 3
        or cosine.shape[0] not in (1, q.shape[0])
        or cosine.shape[1:] != (q.shape[1], q.shape[-1])
    ):
        return None
    batch, length, heads, dim = q.shape
    kv_heads = k.shape[2]
    output_q = torch.empty((batch, heads, length, dim), device=q.device, dtype=q.dtype)
    output_k = torch.empty((batch, kv_heads, length, dim), device=k.device, dtype=k.dtype)
    _qk_norm_rope_kernel()(
        *tensors,
        output_q,
        output_k,
        length,
        heads,
        kv_heads,
        dim,
        cosine.shape[0],
        float(q_norm.eps),
        float(k_norm.eps),
        threads=batch * length * (heads + kv_heads) * 32,
        group_size=32,
    )
    return output_q, output_k


@lru_cache(maxsize=32)
def _metal_library(dim, groups):
    source = Path(__file__).with_suffix(".metal").read_text()
    return torch.mps.compile_shader(
        f"#define HEAD_DIM {dim}\n#define SIMD_GROUPS {groups}\n" + source
    )


@lru_cache(maxsize=96)
def _metal_kernel(dim, groups, name):
    return getattr(_metal_library(dim, groups), name)


@lru_cache(maxsize=32)
def _metal_softmax_kernel(columns, threads):
    source = Path(__file__).with_name("attention_softmax.metal").read_text()
    return torch.mps.compile_shader(
        f"#define THREADS {threads}\n#define ITEMS {(columns + threads - 1) // threads}\n" + source
    ).softmax_lse


def _gemm_attention(q, k, v, mask, scale, threads):
    """Use tuned GEMMs and fuse the elementwise/reduction middle, in bounded tiles."""
    batch, rows, _ = q.shape
    keys = k.shape[1]
    columns = 1 << (keys - 1).bit_length()
    kernel = _metal_softmax_kernel(columns, threads)
    empty = torch.empty(0, device=q.device, dtype=torch.bool)

    def tile(qt, kt, vt, visible):
        scores = torch.bmm(qt, kt.transpose(1, 2))
        z = torch.empty(qt.shape[:2], device=q.device, dtype=torch.float32)
        kernel(
            scores,
            empty if visible is None else visible.contiguous(),
            z,
            keys,
            int(visible is not None),
            float(scale),
            threads=qt.shape[0] * qt.shape[1] * threads,
            group_size=threads,
        )
        return torch.bmm(scores, vt), z

    with torch.autocast("mps", enabled=False):
        if batch * rows * keys <= METAL_SCORE_TILE_ELEMENTS:
            return tile(q, k, v, mask)
        output = torch.empty_like(q)
        lse = torch.empty((batch, rows), device=q.device, dtype=torch.float32)
        query_rows = min(rows, METAL_QUERY_TILE_ROWS) if METAL_QUERY_TILE_ROWS else 1
        batch_width = max(1, METAL_SCORE_TILE_ELEMENTS // (keys * query_rows))
        for b in range(0, batch, batch_width):
            bs = slice(b, min(batch, b + batch_width))
            width = max(1, METAL_SCORE_TILE_ELEMENTS // ((bs.stop - b) * keys))
            for start in range(0, rows, width):
                qs = slice(start, min(rows, start + width))
                output[bs, qs], lse[bs, qs] = tile(
                    q[bs, qs], k[bs], v[bs], None if mask is None else mask[bs, qs]
                )
        return output, lse


def metal_supported(q, k, v):
    return (
        q.device.type == "mps"
        and q.device == k.device == v.device
        and hasattr(torch.mps, "compile_shader")
        and q.ndim == k.ndim == v.ndim == 3
        and all(t.dtype in (torch.float32, torch.float16, torch.bfloat16) for t in (q, k, v))
        and q.shape[0] == k.shape[0] == v.shape[0]
        and k.shape == v.shape
        and q.shape[-1] == k.shape[-1] == v.shape[-1]
        and q.shape[-1] in (8, 16, 32, 64, 128, 256)
        and q.shape[1] > 0
        and k.shape[1] > 0
    )


def _pad(x, length):
    x = x.float().contiguous()
    return F.pad(x, (0, 0, 0, length - x.shape[1])) if x.shape[1] != length else x


class _MetalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, scale):
        batch, rows, dim = q.shape
        keys = k.shape[1]
        profile = metal_profile(dim, keys=keys)
        groups = profile["simd_groups"]
        ctx.gemm = profile.get("algorithm") == "gemm"
        # Both orientations of backward read 32-row tiles.
        padded_rows, padded_keys = (rows + 31) // 32 * 32, (keys + 31) // 32 * 32
        visible = (
            mask.contiguous()
            if mask is not None
            else torch.empty(0, device=q.device, dtype=torch.bool)
        )
        constants = (rows, keys, padded_rows, padded_keys, int(mask is not None), float(scale))
        if ctx.gemm:
            qf, kf, vf = (x.float().contiguous() for x in (q, k, v))
            output, lse = _gemm_attention(qf, kf, vf, mask, scale, profile["softmax_threads"])
        else:
            qf, kf, vf = _pad(q, padded_rows), _pad(k, padded_keys), _pad(v, padded_keys)
            output = torch.empty_like(qf)
            lse = torch.empty((batch, padded_rows), device=q.device, dtype=torch.float32)
            kernel = _metal_kernel(dim, groups, "attention_forward")
            launch = {"threads": (padded_rows // 8 * 32, batch), "group_size": (groups * 32, 1)}
            kernel(qf, kf, vf, visible, output, lse, *constants, **launch)
        ctx.save_for_backward(qf, kf, vf, output, lse, visible)
        ctx.constants, ctx.groups = constants, groups
        ctx.dtypes = q.dtype, k.dtype, v.dtype
        return output[:, :rows], lse[:, :rows]

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_lse):
        q, k, v, output, lse, visible = ctx.saved_tensors
        rows, keys, padded_rows, padded_keys, _, _ = ctx.constants
        if ctx.gemm:
            q, k, v = _pad(q, padded_rows), _pad(k, padded_keys), _pad(v, padded_keys)
            output = _pad(output, padded_rows)
            lse = F.pad(lse, (0, padded_rows - rows), value=-torch.inf).contiguous()
        go = _pad(grad_output, padded_rows)
        gz = F.pad(grad_lse.float(), (0, padded_rows - rows)).contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        delta = torch.empty_like(lse)
        backward_q = _metal_kernel(q.shape[-1], ctx.groups, "attention_backward_q")
        backward_kv = _metal_kernel(q.shape[-1], ctx.groups, "attention_backward_kv")
        launch = {
            "threads": (padded_rows // 8 * 32, q.shape[0]),
            "group_size": (ctx.groups * 32, 1),
        }
        backward_q(q, k, v, visible, output, lse, go, gz, dq, delta, *ctx.constants, **launch)
        launch["threads"] = (padded_keys // 8 * 32, q.shape[0])
        backward_kv(q, k, v, visible, lse, delta, go, gz, dk, dv, *ctx.constants, **launch)
        return (
            dq[:, :rows].to(ctx.dtypes[0]),
            dk[:, :keys].to(ctx.dtypes[1]),
            dv[:, :keys].to(ctx.dtypes[2]),
            None,
            None,
        )


def metal_attention(q, k, v, mask, scale):
    if not metal_supported(q, k, v):
        raise ValueError(
            "Metal attention requires compatible 3D MPS Q/K/V, equal head dim in "
            "8..256 powers of two, and torch.mps.compile_shader"
        )
    if mask is not None and (
        mask.dtype != torch.bool
        or mask.device != q.device
        or mask.shape != (q.shape[0], q.shape[1], k.shape[1])
    ):
        raise ValueError("Metal attention mask must be bool [batch, queries, keys] on Q's device")
    return _MetalAttention.apply(q, k, v, mask, scale)


@lru_cache(maxsize=1)
def _cuda_attention():
    # Lazy import: installing Triton/CUDA is unnecessary on an Apple host.
    return torch.compile(_flex_attention, fullgraph=True, dynamic=True)


def _flex_attention(q, k, v, mask, scale):
    from torch.nn.attention.flex_attention import AuxRequest, flex_attention

    def score_mod(score, batch, head, query, key):
        return torch.where(mask[batch, query, key], score, -torch.inf)

    output, aux = flex_attention(
        q[:, None], k[:, None], v[:, None],
        score_mod=score_mod if mask is not None else None,
        scale=scale,
        return_aux=AuxRequest(lse=True),
    )
    # Natural-log LSE and its gradient join self and cross partitions into
    # the native single softmax; detaching it would change training.
    return output[:, 0], aux.lse[:, 0]


def cuda_attention(q, k, v, mask, scale):
    """Compiled CUDA partitions; retain BF16 operands and FP32 LSE."""
    if q.device.type != "cuda":
        raise ValueError("CUDA attention requires CUDA tensors")
    return _cuda_attention()(q, k, v, mask, scale)
