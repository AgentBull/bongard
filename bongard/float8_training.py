"""FP8 text GEMMs preserving the model's chosen parameter storage dtype."""

from dataclasses import replace
from functools import lru_cache

import torch
from torch.nn import functional as F

RECIPES = ("tensorwise", "rowwise", "rowwise_with_gw_hp")


@lru_cache(maxsize=1)
def _training_linear_class():
    from torchao.float8.float8_linear import Float8Linear

    class TrainingLinear(Float8Linear):
        def forward(self, inputs):
            # Validate and serve the ordinary saved model. Quantization is a
            # training execution policy, like regional_compile, not an export.
            if not self.training:
                return F.linear(inputs, self.weight, self.bias)
            return super().forward(inputs)

    return TrainingLinear


def enable_float8(model, recipe, *, emulate=False):
    """Convert text projections only; CPU emulation verifies wiring, not GPU speed."""
    if recipe not in RECIPES:
        raise ValueError(f"fp8 must be one of {RECIPES}")
    if not emulate and (
        model.device.type != "cuda" or torch.cuda.get_device_capability(model.device) < (8, 9)
    ):
        raise ValueError("FP8 training requires a CUDA GPU with compute capability >= 8.9")
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    from torchao.float8.float8_linear import Float8Linear

    config = replace(
        Float8LinearConfig.from_recipe_name(recipe),
        emulate=emulate,
        # Variable-length requests also produce non-aligned gradient GEMMs.
        pad_inner_dim=True,
    )

    def selected(module, name):
        if isinstance(module, Float8Linear):
            if module.config != config:
                raise ValueError('An already converted model must retain its FP8 recipe')
            return False
        return (
            isinstance(module, torch.nn.Linear)
            and name.startswith(("backbone.encoder.text_model.layers.", "backbone.decoder.layers."))
            and module.weight.requires_grad
            and module.in_features % 16 == module.out_features % 16 == 0
        )

    convert_to_float8_training(model, config=config, module_filter_fn=selected)
    count = 0
    for module in model.modules():
        if isinstance(module, Float8Linear):
            module.__class__ = _training_linear_class()
            count += 1
    if not count:
        raise ValueError("FP8 found no eligible trainable text Linear modules")
    return count
