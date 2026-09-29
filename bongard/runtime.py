"""Accelerate execution and opt-in W&B tracking: one CPU/MPS/CUDA process, or one process per
GPU under torchrun (training.py splits each update across them itself)."""

from __future__ import annotations

import importlib.util
import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import torch
from accelerate import Accelerator
from torch.nn.attention import SDPBackend, sdpa_kernel

from .model import device_for


@dataclass
class WandbConfig:
    mode: str = "disabled"
    project: str = "bongard"
    entity: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)

    def validate(self):
        if self.mode not in {"disabled", "offline", "online"}:
            raise ValueError("wandb.mode must be disabled, offline or online")
        if not isinstance(self.project, str) or not self.project.strip():
            raise ValueError("wandb.project must be a nonempty string")
        for value in (self.entity, self.name):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("wandb.entity/name must be nonempty strings or null")
        if not isinstance(self.tags, list) or any(not isinstance(tag, str) for tag in self.tags):
            raise ValueError("wandb.tags must be a list of strings")


@contextmanager
def training_runtime(config):
    # Several processes (torchrun) are plain data parallelism implemented in training.py: each
    # update's records are split across ranks, weighted by the whole update and summed once.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    expected = device_for(config.device)
    if world != 1 and expected.type not in {"cuda", "cpu"}:
        raise ValueError("Multi-process training needs CUDA (CPU for tests); launch with torchrun")
    tracking = config.wandb.mode != "disabled"
    if tracking and importlib.util.find_spec("wandb") is None:
        raise ImportError("W&B tracking requires: uv sync --extra tracking")
    handlers = []
    if world != 1:
        from datetime import timedelta

        from accelerate.utils import InitProcessGroupKwargs

        # Rank 0 evaluates and saves alone while the others wait in the next collective.
        # BONGARD_DIST_BACKEND=gloo runs several processes on one GPU (a test setup only).
        backend = os.environ.get("BONGARD_DIST_BACKEND")
        handlers.append(InitProcessGroupKwargs(timeout=timedelta(hours=2),
                                               **({"backend": backend} if backend else {})))
    accelerator = Accelerator(
        cpu=expected.type == "cpu",
        kwargs_handlers=handlers,
        mixed_precision="bf16" if config.precision == "bfloat16" else "no",
        # Each update already divides by its actual total record weight. A
        # second automatic accumulation divisor would change the objective.
        gradient_accumulation_steps=1,
        step_scheduler_with_optimizer=False,
        log_with="wandb" if tracking else None,
        project_dir=config.output,
    )
    if accelerator.device.type != expected.type or accelerator.num_processes != world:
        raise ValueError(
            f"Accelerate selected {accelerator.device} x{accelerator.num_processes}; requested "
            f"{expected} x{world}. Use a fresh process when changing device or precision."
        )
    if world != 1 and expected.type == "cuda":
        torch.cuda.set_device(accelerator.device)
    try:
        # cuDNN SDPA can return NaN dQ for finite masked attention on Blackwell
        # (pytorch/pytorch#196678). Keep this policy active during checkpoint
        # replay as well as forward; other SDPA kernels retain the same math.
        kernels = sdpa_kernel([
            SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH,
        ]) if expected.type == "cuda" else nullcontext()
        with kernels:
            yield accelerator
    except BaseException:
        if accelerator.trackers:
            accelerator.get_tracker("wandb", unwrap=True).finish(exit_code=1)
        raise
    finally:
        accelerator.end_training()
        accelerator.free_memory()


def start_tracking(accelerator, config, *, completed, resume):
    if config.wandb.mode == "disabled" or not accelerator.is_main_process:
        return
    output = Path(config.output)
    identity_file = output / "wandb-run.json"
    if resume and identity_file.exists():
        run_id = json.loads(identity_file.read_text())["id"]
    else:
        run_id = uuid4().hex
        identity_file.write_text(json.dumps({"id": run_id}) + "\n")
    # Offline W&B has no SDK resume; retain the logical run ID and write a new
    # local log directory at the restored update boundary, ready for later sync.
    directory = output / "tracking" / f"from-step-{completed:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    options = {
        "id": run_id,
        "mode": config.wandb.mode,
        "entity": config.wandb.entity,
        "name": config.wandb.name or output.name,
        "tags": config.wandb.tags,
        "dir": str(directory),
        "settings": {"disable_git": True, "save_code": False},
    }
    if config.wandb.mode == "online":
        options["resume"] = "allow" if resume else "never"
    accelerator.init_trackers(
        config.wandb.project, config=asdict(config), init_kwargs={"wandb": options}
    )


def scalar_metrics(row, prefix=""):
    """Log existing numeric diagnostics, never raw records, views, or weights."""
    output = {}
    for key, value in row.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            output[name] = value
        elif isinstance(value, dict):
            output.update(scalar_metrics(value, name))
    return output
