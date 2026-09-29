"""Text LoRA with a trainable judgment head and native visual projector."""

from __future__ import annotations

import importlib.util
import math
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LoraSettings:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    def validate(self):
        if type(self.enabled) is not bool:
            raise ValueError("lora.enabled must be boolean")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("lora.rank must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("lora.alpha must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("lora.dropout must lie in [0, 1)")
        supported = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        if (
            not isinstance(self.targets, list)
            or not self.targets
            or any(not isinstance(name, str) or name not in supported for name in self.targets)
            or len(set(self.targets)) != len(self.targets)
        ):
            raise ValueError("lora.targets must be unique attention/MLP projection names")


def require_peft():
    if importlib.util.find_spec("peft") is None:
        raise ImportError("LoRA requires: uv sync --extra peft")


def configure_lora(model, settings, base_directory, *, resume=False):
    """Save the base once per run, then train text adapters, projector and head.

    Each checkpoint references this sibling snapshot, not a mutable Hub branch
    or a previous course's weights. Keep the whole run directory when moving it.
    """
    if not settings.enabled:
        if model.lora_settings is not None:
            raise ValueError("An adapter checkpoint requires matching lora settings")
        return
    require_peft()
    from peft import LoraConfig

    if model.lora_settings is not None and model.lora_settings != asdict(settings):
        raise ValueError("Cannot change LoRA structure when continuing an adapter checkpoint")
    if resume:
        if model.lora_settings is None:
            raise ValueError("LoRA resume requires an adapter checkpoint")
        return
    base_directory = Path(base_directory).resolve()
    if base_directory.exists():
        raise FileExistsError(f"Frozen backbone already exists; use a new run: {base_directory}")
    if model.lora_base is not None:
        shutil.copytree(model.lora_base, base_directory)
    else:
        # Snapshot before injection so native save_pretrained stores the full
        # backbone, including an already fine-tuned course's exact parameters.
        model.backbone.save_pretrained(base_directory)
        pattern = (
            r"(?:encoder\.text_model|decoder)\..*\.(?:"
            + "|".join(re.escape(name) for name in settings.targets)
            + ")"
        )
        model.backbone.add_adapter(
            LoraConfig(
                r=settings.rank,
                lora_alpha=settings.alpha,
                lora_dropout=settings.dropout,
                bias="none",
                target_modules=pattern,
                modules_to_save=["encoder.multi_modal_projector"],
            )
        )
    model.lora_base = base_directory
    model.lora_settings = asdict(settings)
