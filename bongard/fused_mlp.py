"""Upstream TE training execution with ordinary Hugging Face checkpoint weights."""

from collections import Counter
from functools import lru_cache

import torch
from torch.nn import functional as F

RECIPES = ('nvfp4', 'mxfp8', 'fp8')


@lru_cache(maxsize=1)
def _fused_class():
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import (
        Float8CurrentScaling,
        MXFP8BlockScaling,
        NVFP4BlockScaling,
    )

    class FusedMLP(te.LayerNormMLP):
        def __init__(self, mlp, norm, recipe):
            if mlp.config.hidden_activation != 'gelu_pytorch_tanh' or mlp.dropout.p:
                raise ValueError('Fused MLP requires tanh GELU and zero dropout')
            super().__init__(
                mlp.hidden_size, mlp.intermediate_size, eps=norm.eps, bias=False,
                normalization='RMSNorm', activation='geglu', zero_centered_gamma=True,
                params_dtype=torch.bfloat16, device=mlp.gate_proj.weight.device,
                init_method=torch.nn.init.zeros_, output_layer_init_method=torch.nn.init.zeros_,
            )
            self.bongard_recipe = recipe
            self.recipe = {'nvfp4': NVFP4BlockScaling, 'mxfp8': MXFP8BlockScaling,
                           'fp8': Float8CurrentScaling}[recipe]()
            self.intermediate_size = mlp.intermediate_size
            with torch.no_grad():
                self.layer_norm_weight.copy_(norm.weight)
                self.fc1_weight[:self.intermediate_size].copy_(mlp.gate_proj.weight)
                self.fc1_weight[self.intermediate_size:].copy_(mlp.up_proj.weight)
                self.fc2_weight.copy_(mlp.down_proj.weight)
            # These HF evaluation views share weight storage. They are not a
            # second registered parameter tree or optimizer owner.
            object.__setattr__(self, '_hf_mlp', mlp.eval())
            object.__setattr__(self, '_hf_norm', norm.eval())
            self._refresh_hf_views()
            self.counts = Counter()
            self._cached_weight_versions = None

        def _refresh_hf_views(self):
            self._hf_norm.weight = self.layer_norm_weight
            self._hf_mlp.gate_proj.weight = torch.nn.Parameter(
                self.fc1_weight[:self.intermediate_size], requires_grad=False)
            self._hf_mlp.up_proj.weight = torch.nn.Parameter(
                self.fc1_weight[self.intermediate_size:], requires_grad=False)
            self._hf_mlp.down_proj.weight = self.fc2_weight

        def _apply(self, fn, recurse=True):
            super()._apply(fn, recurse=recurse)
            self._refresh_hf_views()
            self._cached_weight_versions = None
            return self

        @torch.compiler.disable
        def forward(self, x):
            if not self.training:
                return self._hf_mlp(self._hf_norm(x))
            flat = x.reshape(-1, x.shape[-1])
            length = flat.shape[0]
            aligned = (length + 31) // 32 * 32
            versions = (self.fc1_weight._version, self.fc2_weight._version)
            first_microbatch = versions != self._cached_weight_versions
            self._cached_weight_versions = versions
            self.counts['weight_quantizations' if first_microbatch else 'cached_weights'] += 1
            self.counts['real_tokens'] += length
            self.counts['kernel_tokens'] += aligned
            # Alignment only: at most 31 extra positions per physical pack.
            with te.autocast(enabled=True, recipe=self.recipe):
                out = super().forward(F.pad(flat, (0, 0, 0, aligned - length)),
                                      is_first_microbatch=first_microbatch)
            return out[:length].reshape_as(x)

    return FusedMLP


def enable_fused_mlp(model, recipe):
    if recipe not in RECIPES or model.device.type != 'cuda':
        raise ValueError('Fused MLP requires CUDA and a supported low-precision recipe')
    fused_class = _fused_class()
    count = 0
    for layers in (model.backbone.encoder.text_model.layers, model.backbone.decoder.layers):
        for layer in layers:
            if isinstance(layer.mlp, fused_class):
                if layer.mlp.bongard_recipe != recipe:
                    raise ValueError('An existing fused MLP must retain its recipe')
            else:
                layer.mlp = fused_class(layer.mlp, layer.pre_feedforward_layernorm, recipe)
                layer.pre_feedforward_layernorm = torch.nn.Identity()
            count += 1
    return count


def hf_state_dict(backbone):
    """Invert the fused gate/up layout (and TE projections) for Transformers.save_pretrained."""
    from . import te_linear

    fused = [(name, module) for name, module in backbone.named_modules()
             if hasattr(module, 'bongard_recipe') and hasattr(module, 'fc1_weight')]
    te_projections = te_linear.present(backbone)
    if not fused and not te_projections:
        return None
    state = backbone.state_dict()
    if te_projections:
        state = te_linear.restore_hf_layout(backbone, state)
    for name, module in fused:
        prefix = name + '.'
        extra = state.pop(prefix + '_extra_state')
        if extra.numel():
            raise ValueError('Fused MLP export requires stateless quantization scaling')
        gate, up = state.pop(prefix + 'fc1_weight').split(module.intermediate_size)
        state[prefix + 'gate_proj.weight'] = gate
        state[prefix + 'up_proj.weight'] = up
        state[prefix + 'down_proj.weight'] = state.pop(prefix + 'fc2_weight')
        state[name.rsplit('.', 1)[0] + '.pre_feedforward_layernorm.weight'] = state.pop(
            prefix + 'layer_norm_weight')
    return state
