"""Muon or Lion for hidden matrices, AdamW under one scheduler/checkpoint."""

from __future__ import annotations

import torch

from .lion import Lion


class MatrixAdamW(torch.optim.Optimizer):
    """Delegate matrix/auxiliary groups, sharing one state dictionary.

    Moonlight scales the orthogonalized update by 0.2 * sqrt(max(rows, cols)),
    while weight decay uses the unscaled learning rate. Existing AdamW learning
    rates are a starting point, not an exact update-RMS match: short, spectrally
    concentrated gradients and transient Adam moments can differ substantially.
    Native Muon uses five BF16 Newton-Schulz iterations; model parameters and
    momentum stay FP32.

    Lion uses the official sign-momentum rule. Its learning rate and decay
    multipliers are set explicitly on its groups by optimizer_for().

    References:
      https://platform.kimi.com/blog/posts/moonlight
      https://kexue.fm/archives/11416
      https://kexue.fm/archives/10739
    """

    def __init__(self, parameter_groups, *, weight_decay, lion_betas=(0.9, 0.99)):
        self.optimizers = {}
        for algorithm in ("muon", "lion", "adamw"):
            groups = [g for g in parameter_groups if g["algorithm"] == algorithm]
            if not groups:
                continue
            if algorithm == "muon":
                optimizer = torch.optim.Muon(
                    groups,
                    weight_decay=weight_decay,
                    momentum=0.95,
                    nesterov=True,
                    ns_steps=5,
                    adjust_lr_fn="match_rms_adamw",
                )
            elif algorithm == "lion":
                optimizer = Lion(groups, weight_decay=weight_decay, betas=lion_betas)
            else:
                optimizer = torch.optim.AdamW(
                    groups,
                    betas=(0.9, 0.95),
                    weight_decay=weight_decay,
                    foreach=False,
                    fused=False,
                )
            self.optimizers[algorithm] = optimizer
        super().__init__(parameter_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for algorithm, optimizer in self.optimizers.items():
            # load_state_dict replaces these objects, including when Accelerate
            # moves state to the device. Always bind to the current shared state.
            optimizer.param_groups = [g for g in self.param_groups if g["algorithm"] == algorithm]
            optimizer.state = self.state
            optimizer.step()
        return loss
