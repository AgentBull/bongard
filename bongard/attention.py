"""Exact decoder attention over private questions and a shared state.

Queries sharing K/V are folded into the query dimension, never the KV batch.
Each partition returns its log normalizer; one softmax merges self and state
partitions. Backward recomputes bounded score tiles instead of retaining
an attention matrix per decoder layer. All projections and weights stay native.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from transformers.models.t5gemma2.modeling_t5gemma2 import (
    T5Gemma2MergedAttention,
    apply_rotary_pos_emb,
)

from bongard.flash_attention import cuda_attention, metal_attention, metal_qk_norm_rope

# Bound each temporary score matrix to 16 MiB in FP32, independently of state
# length and number of questions. This is workspace, not an input truncation.
SCORE_TILE_ELEMENTS = 4 * 1024 * 1024
_BACKEND = ContextVar("bongard_attention_backend", default="reference")


@contextmanager
def attention_backend(name):
    """Select reference/metal/cuda locally, including for validation.

    The bounded reference remains the default: Metal tuning is device-specific
    and its backward is still slower. CUDA needs target-server validation.
    """
    if name not in ("reference", "metal", "cuda"):
        raise ValueError(f"Unknown attention backend: {name}")
    token = _BACKEND.set(name)
    try:
        yield
    finally:
        _BACKEND.reset(token)


def _tiles(batch, queries, keys, budget):
    batch_width = max(1, budget // keys)
    for b in range(0, batch, batch_width):
        end = min(batch, b + batch_width)
        query_width = max(1, budget // ((end - b) * keys))
        for q in range(0, queries, query_width):
            yield slice(b, end), slice(q, min(queries, q + query_width))


def _probabilities(q, k, mask, scale):
    scores = torch.bmm(q, k.transpose(1, 2)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    lse = scores.logsumexp(-1)
    # A padded query can have no visible self keys. Its partition has
    # zero weight; the state partition remains visible in the joint softmax.
    normalizer = torch.where(torch.isfinite(lse), lse, 0)
    return (scores - normalizer.unsqueeze(-1)).exp(), lse


class _PartitionAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, scale):
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        with torch.autocast(q.device.type, enabled=False):
            kf, vf = k.to(dtype), v.to(dtype)
            if q.shape[0] * q.shape[1] * k.shape[1] <= SCORE_TILE_ELEMENTS:
                p, lse = _probabilities(q.to(dtype), kf, mask, scale)
                output = torch.bmm(p, vf)
            else:
                output = torch.empty((*q.shape[:-1], v.shape[-1]), device=q.device, dtype=dtype)
                lse = torch.empty(q.shape[:-1], device=q.device, dtype=dtype)
                for batch, rows in _tiles(*q.shape[:2], k.shape[1], SCORE_TILE_ELEMENTS):
                    p, z = _probabilities(
                        q[batch, rows].to(dtype),
                        kf[batch],
                        None if mask is None else mask[batch, rows],
                        scale,
                    )
                    output[batch, rows] = torch.bmm(p, vf[batch])
                    lse[batch, rows] = z
        ctx.save_for_backward(q, k, v, output, lse, mask)
        ctx.scale, ctx.budget = scale, SCORE_TILE_ELEMENTS
        return output, lse

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_lse):
        q, k, v, output, lse, mask = ctx.saved_tensors
        dtype = output.dtype
        with torch.autocast(q.device.type, enabled=False):
            kf, vf = k.to(dtype), v.to(dtype)
            dq = torch.empty_like(q, dtype=dtype)
            dk, dv = torch.zeros_like(kf), torch.zeros_like(vf)
            for batch, rows in _tiles(*q.shape[:2], k.shape[1], ctx.budget):
                qt = q[batch, rows].to(dtype)
                kt, vt = kf[batch], vf[batch]
                scores = torch.bmm(qt, kt.transpose(1, 2)) * ctx.scale
                if mask is not None:
                    scores = scores.masked_fill(~mask[batch, rows], -torch.inf)
                z = lse[batch, rows]
                p = (scores - torch.where(torch.isfinite(z), z, 0).unsqueeze(-1)).exp()
                go = grad_output[batch, rows].to(dtype)
                # LSE participates in the joint softmax, so its gradient is
                # essential: dropping it changes both self and shared gradients.
                ds = torch.bmm(go, vt.transpose(1, 2))
                ds -= (go * output[batch, rows]).sum(-1, keepdim=True)
                ds += grad_lse[batch, rows].to(dtype).unsqueeze(-1)
                ds *= p
                dq[batch, rows] = torch.bmm(ds, kt) * ctx.scale
                dk[batch].add_(torch.bmm(ds.transpose(1, 2), qt), alpha=ctx.scale)
                dv[batch].add_(torch.bmm(p.transpose(1, 2), go))
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def partition_attention(q, k, v, mask=None, *, shared, scale, dropout=0.0, backend=None):
    """Return output and LSE in [branch, query_head, token, ...] order.

    GQA folds query heads into rows as well, so K/V are never repeated by head.
    A mask, when needed, is [branch, query_token, key_token].
    """
    batch, heads, length, dim = q.shape
    kv_heads, keys = k.shape[1:3]
    groups = heads // kv_heads
    q = q.reshape(batch, kv_heads, groups, length, dim)
    if shared:
        q = q.permute(1, 0, 2, 3, 4).reshape(kv_heads, batch * groups * length, dim)
        k, v = k[0], v[0]
        if mask is not None:
            mask = mask[None, :, None].expand(kv_heads, batch, groups, length, keys)
            mask = mask.reshape(kv_heads, batch * groups * length, keys)
    else:
        q = q.reshape(batch * kv_heads, groups * length, dim)
        k = k.reshape(batch * kv_heads, keys, dim)
        v = v.reshape(batch * kv_heads, keys, v.shape[-1])
        if mask is not None:
            mask = mask[:, None, None].expand(batch, kv_heads, groups, length, keys)
            mask = mask.reshape(batch * kv_heads, groups * length, keys)
    if dropout:
        # Native T5Gemma2 checkpoints use zero attention dropout. Preserve the
        # general training contract if a caller explicitly configures it.
        with torch.autocast(q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            p, lse = _probabilities(q.to(dtype), k.to(dtype), mask, scale)
            output = nn.functional.dropout(p, p=dropout, training=True) @ v.to(dtype)
    else:
        backend = _BACKEND.get() if backend is None else backend
        if backend == "metal":
            output, lse = metal_attention(q, k, v, mask, scale)
        elif backend == "cuda":
            output, lse = cuda_attention(q, k, v, mask, scale)
        else:
            output, lse = _PartitionAttention.apply(q, k, v, mask, scale)
    if shared:
        output = output.reshape(kv_heads, batch, groups, length, -1).permute(1, 0, 2, 3, 4)
        lse = lse.reshape(kv_heads, batch, groups, length).permute(1, 0, 2, 3)
    return output.reshape(batch, heads, length, -1), lse.reshape(batch, heads, length)


def merge_partitions(parts):
    outputs, lses = zip(*parts)
    weights = torch.stack(lses).softmax(0)
    return (torch.stack(outputs) * weights.unsqueeze(-1)).sum(0)


def project_qk(module, hidden_states, position_embeddings, backend):
    shape = (*hidden_states.shape[:-1], -1, module.head_dim)
    q, k = module.q_proj(hidden_states).view(shape), module.k_proj(hidden_states).view(shape)
    if backend == "metal" and not module.training:
        fused = metal_qk_norm_rope(q, k, module.q_norm, module.k_norm, position_embeddings)
        if fused is not None:
            return fused
    q, k = module.q_norm(q.transpose(1, 2)), module.k_norm(k.transpose(1, 2))
    return apply_rotary_pos_emb(q, k, *position_embeddings)


@dataclass
class QuestionLayout:
    valid: torch.Tensor
    positions: torch.Tensor
    masks: dict = field(default_factory=dict)
    # Contiguous question rows, state-bank row, and unpadded state length.
    # Empty retains the original single shared-state path.
    state_groups: tuple[tuple[slice, int, int], ...] = ()
    # Capture at the decoder call site: autograd checkpoint replay may run in
    # another thread or after the caller has left attention_backend().
    backend: str = field(default_factory=_BACKEND.get)

    def self_mask(self, window):
        if window not in self.masks:
            distance = self.positions[:, :, None] - self.positions[:, None, :]
            visible = (distance >= 0) & self.valid[:, None] & self.valid[:, :, None]
            if window is not None:
                visible &= distance < window
            self.masks[window] = visible
        return self.masks[window]


class SharedMergedAttention(T5Gemma2MergedAttention):
    def __init__(self, native):
        # Reuse the native parameter objects and names, including PEFT targets.
        # No second set of weights or additional checkpoint format is introduced.
        nn.Module.__init__(self)
        for name in (
            "config",
            "layer_idx",
            "layer_type",
            "head_dim",
            "num_key_value_groups",
            "scaling",
            "attention_dropout",
            "is_causal",
            "attn_logit_softcapping",
            "sliding_window",
            "is_sliding",
        ):
            setattr(self, name, getattr(native, name))
        for name, module in native.named_children():
            self.add_module(name, module)
        self.train(native.training)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        merged_attention_mask,
        encoder_hidden_states,
        past_key_values=None,
        *,
        question_layout=None,
        packed_cross=None,
        **kwargs,
    ):
        from .packing import PackedLayout, decoder_attention

        if isinstance(question_layout, PackedLayout):
            return decoder_attention(
                self, hidden_states, position_embeddings, past_key_values, question_layout,
                packed_cross,
            )
        if question_layout is None:
            return super().forward(
                hidden_states,
                position_embeddings,
                merged_attention_mask,
                encoder_hidden_states,
                past_key_values,
                **kwargs,
            )
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        q, k = project_qk(self, hidden_states, position_embeddings, question_layout.backend)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        cache, layout = past_key_values, question_layout
        dropout = self.attention_dropout if self.training else 0.0
        options = {"scale": self.scaling, "dropout": dropout, "backend": layout.backend}
        mask = layout.self_mask(self.sliding_window)
        parts = [partition_attention(q, k, v, mask, shared=False, **options)]
        cross = cache.cross_attention_cache.layers[self.layer_idx]
        if layout.state_groups:
            state_parts = [
                partition_attention(
                    q[rows],
                    cross.keys[index : index + 1, :, :length],
                    cross.values[index : index + 1, :, :length],
                    shared=True,
                    **options,
                )
                for rows, index, length in layout.state_groups
            ]
            parts.append(tuple(torch.cat(values) for values in zip(*state_parts)))
        else:
            parts.append(
                partition_attention(q, cross.keys[:1], cross.values[:1], shared=True, **options)
            )
        output = merge_partitions(parts).to(q.dtype).transpose(1, 2)
        output = self.o_proj(output.reshape(*hidden_states.shape[:-1], -1))
        return output, None, None


def install_shared_attention(backbone):
    for layer in backbone.decoder.layers:
        if not isinstance(layer.self_attn, SharedMergedAttention):
            layer.self_attn = SharedMergedAttention(layer.self_attn)
