"""Transformer Engine projections with ordinary Hugging Face checkpoint weights.

An alternative to fused_mlp for GPUs where TE's runtime-compiled normalization kernels do
not run (B300 / sm_103, 2026-09-28): the model's own RMSNorm stays and only the projections
execute as te.Linear under a low-precision recipe. The MLP gate and up projections share one
GEMM. Weights remain ordinary BF16 parameters for the optimizer; restore_hf_layout puts the
Hugging Face module layout back for save_pretrained.
"""

import os
from collections import Counter
from functools import lru_cache

import torch
from torch.nn import functional as F

RECIPES = ('nvfp4', 'mxfp8', 'fp8')
ATTENTION_PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')


def _te_linear(module):
    return hasattr(module, 'bongard_recipe') and hasattr(module, 'splits')


def _te_mlp(module):
    return hasattr(module, 'bongard_recipe') and hasattr(module, 'fused_gate_up')


@lru_cache(maxsize=1)
def _classes():
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import (
        Float8CurrentScaling,
        MXFP8BlockScaling,
        NVFP4BlockScaling,
    )

    recipes = {'nvfp4': NVFP4BlockScaling, 'mxfp8': MXFP8BlockScaling,
               'fp8': Float8CurrentScaling}

    class TELinear(te.Linear):
        """One te.Linear serving one or more stacked HF Linear weights of the same input."""

        def __init__(self, linears, recipe):
            if any(linear.bias is not None for linear in linears):
                raise ValueError('TE projections require bias-free HF linears')
            splits = [linear.out_features for linear in linears]
            super().__init__(
                linears[0].in_features, sum(splits), bias=False, params_dtype=torch.bfloat16,
                device=linears[0].weight.device, init_method=torch.nn.init.zeros_)
            self.bongard_recipe = recipe
            self.recipe = recipes[recipe]()
            self.splits = splits
            with torch.no_grad():
                self.weight.copy_(torch.cat([linear.weight for linear in linears], 0))
            self.counts = Counter()
            self._cached_weight_version = None

        @torch.compiler.disable
        def forward(self, x):
            if not self.training:
                return F.linear(x, self.weight)
            flat = x.reshape(-1, x.shape[-1])
            length = flat.shape[0]
            aligned = (length + 31) // 32 * 32
            version = self.weight._version
            first_microbatch = version != self._cached_weight_version
            self._cached_weight_version = version
            self.counts['weight_quantizations' if first_microbatch else 'cached_weights'] += 1
            # Alignment only: at most 31 extra rows per call, and no copy when aligned.
            if aligned != length:
                flat = F.pad(flat, (0, 0, 0, aligned - length))
            with te.autocast(enabled=True, recipe=self.recipe):
                out = super().forward(flat, is_first_microbatch=first_microbatch)
            if aligned != length:
                out = out[:length]
            return out.reshape(*x.shape[:-1], out.shape[-1])

    class TEMLP(torch.nn.Module):
        """The HF gated MLP with its three projections as two TE GEMMs."""

        def __init__(self, mlp, recipe):
            super().__init__()
            if mlp.config.hidden_activation != 'gelu_pytorch_tanh' or mlp.dropout.p:
                raise ValueError('TE MLP requires tanh GELU and zero dropout')
            self.bongard_recipe = recipe
            self.intermediate_size = mlp.intermediate_size
            # One fused gate/up GEMM, or two GEMMs on the same input: the fused form splits
            # its output and concatenates the gradient again (about 0.1 s per B300 update).
            self.fused_gate_up = os.environ.get('BONGARD_TE_SPLIT_GATE_UP', '0') != '1'
            if self.fused_gate_up:
                self.gate_up = TELinear([mlp.gate_proj, mlp.up_proj], recipe)
            else:
                self.gate_proj = TELinear([mlp.gate_proj], recipe)
                self.up_proj = TELinear([mlp.up_proj], recipe)
            self.down = TELinear([mlp.down_proj], recipe)

        def forward(self, x):
            if self.fused_gate_up:
                gate, up = self.gate_up(x).split(self.intermediate_size, dim=-1)
            else:
                gate, up = self.gate_proj(x), self.up_proj(x)
            return self.down(F.gelu(gate, approximate='tanh') * up)

    return TELinear, TEMLP


def enable_te_linear(model, *, mlp=None, attention=None):
    """Convert MLP projections and/or attention projections of every text layer."""
    if model.device.type != 'cuda' or not (mlp or attention):
        raise ValueError('TE projections require CUDA and at least one recipe')
    for recipe in (mlp, attention):
        if recipe is not None and recipe not in RECIPES:
            raise ValueError(f'TE recipe must be one of {RECIPES}')
    TELinear, TEMLP = _classes()
    # Packed forwards round every pack to 32 tokens, so no projection pads or slices.
    model.pack_multiple = 32
    counts = Counter()
    for layers in (model.backbone.encoder.text_model.layers, model.backbone.decoder.layers):
        for layer in layers:
            if mlp:
                if isinstance(layer.mlp, TEMLP):
                    if layer.mlp.bongard_recipe != mlp:
                        raise ValueError('An existing TE MLP must retain its recipe')
                else:
                    layer.mlp = TEMLP(layer.mlp, mlp)
                counts['mlp'] += 1
            if attention:
                for name in ATTENTION_PROJECTIONS:
                    module = getattr(layer.self_attn, name)
                    if isinstance(module, TELinear):
                        if module.bongard_recipe != attention:
                            raise ValueError('An existing TE projection must retain its recipe')
                    else:
                        setattr(layer.self_attn, name, TELinear([module], attention))
                    counts['attention'] += 1
    return dict(counts)


def present(backbone):
    return any(_te_linear(m) or _te_mlp(m) for m in backbone.modules())


def restore_hf_layout(backbone, state):
    """Rename TE projection weights to the HF module keys and drop TE extra state."""
    for name, module in backbone.named_modules():
        if _te_mlp(module):
            prefix = name + '.'
            projections = ('gate_up',) if module.fused_gate_up else ('gate_proj', 'up_proj')
            for key in (*projections, 'down'):
                if state.pop(prefix + key + '._extra_state').numel():
                    raise ValueError('TE export requires stateless quantization scaling')
            if module.fused_gate_up:
                gate, up = state.pop(prefix + 'gate_up.weight').split(module.intermediate_size)
                state[prefix + 'gate_proj.weight'] = gate
                state[prefix + 'up_proj.weight'] = up
            state[prefix + 'down_proj.weight'] = state.pop(prefix + 'down.weight')
        elif _te_linear(module) and name.rsplit('.', 1)[-1] in ATTENTION_PROJECTIONS:
            if state.pop(name + '._extra_state').numel():
                raise ValueError('TE export requires stateless quantization scaling')
    return state
