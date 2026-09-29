"""Measured MPS inference substitutions; native weights and training are retained."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.t5gemma2.modeling_t5gemma2 import (
    T5Gemma2RMSNorm,
    T5Gemma2SelfAttention,
)

from .attention import _BACKEND, partition_attention, project_qk
from .flash_attention import metal_profile


def _window_attention(q, k, v, mask, scale, window, block, *, mask_cache=None):
    """Batch overlapping key bands, preserving each native mask entry.

    Caller guarantees that the mask came from the native encoder's bidirectional
    sliding-window factory. Arbitrary user-supplied masks must use the dense path.
    """
    batch, heads, length, dim = q.shape
    chunks = (length + block - 1) // block
    tail = chunks * block - length
    left, right = (window + 1) // 2, window // 2
    width = block + left + right
    queries = F.pad(q, (0, 0, 0, tail))
    queries = (
        queries.reshape(batch, heads, chunks, block, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * chunks, heads, block, dim)
    )

    def bands(t):
        return (
            F.pad(t, (0, 0, left, right + tail))
            .unfold(2, width, block)
            .permute(0, 2, 1, 4, 3)
            .reshape(batch * chunks, t.shape[1], width, dim)
        )

    key = (mask.data_ptr(), window, block)
    visible = None if mask_cache is None else mask_cache.get(key)
    if visible is None:
        starts = torch.arange(chunks, device=q.device) * block
        qi = starts[:, None, None] + torch.arange(block, device=q.device)[None, :, None]
        ki = starts[:, None, None] + torch.arange(width, device=q.device)[None, None, :] - left
        visible = mask[:, qi.clamp(max=length - 1), ki.clamp(min=0, max=length - 1)]
        visible = visible & (qi < length) & (ki >= 0) & (ki < length)
        visible = visible.reshape(batch * chunks, block, width)
        if mask_cache is not None:
            mask_cache[key] = visible
    output, _ = partition_attention(
        queries,
        bands(k),
        bands(v),
        visible,
        shared=False,
        scale=scale,
        backend="metal",
    )
    return (
        output.reshape(batch, chunks, heads, block, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, heads, chunks * block, dim)[:, :, :length]
    )


class MPSRMSNorm(T5Gemma2RMSNorm):
    def __init__(self, native):
        nn.Module.__init__(self)
        self.weight, self.eps = native.weight, native.eps
        self.train(native.training)

    def forward(self, x):
        if x.device.type == "mps" and not self.training and not torch.is_grad_enabled():
            # Gemma multiplies (1 + weight) BEFORE rounding to the input dtype.
            # Calling rms_norm on x's low precision dtype would change that rule.
            return F.rms_norm(x.float(), (x.shape[-1],), 1.0 + self.weight.float(), self.eps).to(
                x.dtype
            )
        return super().forward(x)


class MPSEncoderAttention(T5Gemma2SelfAttention):
    def __init__(self, native):
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
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        *,
        bongard_encoder_windows=None,
        packed_layout=None,
        **kwargs,
    ):
        if packed_layout is not None:
            from .packing import encoder_attention

            return encoder_attention(self, hidden_states, position_embeddings, packed_layout)
        eligible = (
            hidden_states.device.type == "mps"
            and not self.training
            and not torch.is_grad_enabled()
            and _BACKEND.get() == "metal"
            and self.config._attn_implementation == "sdpa"
            and self.attn_logit_softcapping is None
            and past_key_values is None
            and not kwargs.get("output_attentions", False)
            and (
                attention_mask is None
                or (
                    attention_mask.dtype == torch.bool
                    and attention_mask.ndim == 4
                    and attention_mask.shape[1] == 1
                )
            )
            and metal_profile(self.head_dim, keys=hidden_states.shape[1])["algorithm"] == "gemm"
        )
        if not eligible:
            return super().forward(
                hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs
            )
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        q, k = project_qk(self, hidden_states, position_embeddings, "metal")
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        mask = (
            None
            if attention_mask is None
            else attention_mask[:, 0].expand(q.shape[0], q.shape[2], k.shape[2])
        )
        block = metal_profile(self.head_dim, keys=q.shape[2]).get("encoder_window_block", 0)
        if bongard_encoder_windows is not None and self.is_sliding and mask is not None and block:
            output = _window_attention(
                q,
                k,
                v,
                mask,
                self.scaling,
                self.sliding_window,
                block,
                mask_cache=bongard_encoder_windows,
            )
        else:
            output, _ = partition_attention(
                q, k, v, mask, shared=False, scale=self.scaling, backend="metal"
            )
        output = output.to(q.dtype).transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return self.o_proj(output), None


def install_mps_inference(backbone):
    for layer in backbone.encoder.text_model.layers:
        if type(layer.self_attn) is T5Gemma2SelfAttention:
            layer.self_attn = MPSEncoderAttention(layer.self_attn)
    for parent in backbone.modules():
        for name, child in parent.named_children():
            if type(child) is T5Gemma2RMSNorm:
                setattr(parent, name, MPSRMSNorm(child))
