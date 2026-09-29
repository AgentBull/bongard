# Copyright 2023 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dense Lion, adapted from google/automl/lion/lion_pytorch.py.

Changes: validate finite hyperparameters and reject sparse gradients explicitly.
The upstream Apache 2.0 license is included in lion.LICENSE.
Both beta values blend gradients; Lion has no second-moment state.
"""

import math

import torch


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("Lion lr must be finite and nonnegative")
        if len(betas) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in betas):
            raise ValueError("Lion betas must contain two finite values in [0, 1)")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("Lion weight_decay must be finite and nonnegative")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Lion requires dense gradients")
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                momentum = state["exp_avg"]
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                update = momentum * beta1 + gradient * (1 - beta1)
                parameter.add_(update.sign_(), alpha=-group["lr"])
                momentum.mul_(beta2).add_(gradient, alpha=1 - beta2)
        return loss
