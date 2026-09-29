"""Record-weighted PyTorch training, deterministic view selection and resumable updates."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from array import array
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from accelerate.utils import set_seed

from .adapters import LoraSettings, configure_lora
from .attention import attention_backend
from .choice_permutation import permute
from .compiler import Compiler, Request
from .data import DataError, check_split_groups, eligible_views, metadata_records
from .dataset import file_sha256, read_dataset
from .losses import record_loss, record_losses
from .model import JudgmentModel, device_for
from .optimizers import MatrixAdamW
from .provenance import checkpoint_sources, course_sources, membership
from .runtime import WandbConfig, scalar_metrics, start_tracking, training_runtime
from .token_safety import TokenSafetyError


@dataclass
class JepaConfig:
    enabled: bool = False
    coefficient: float = 0.01
    warmup_ratio: float = 0.05
    data: str | None = None
    objective: str = "paired_views"
    view_supervision_mix: float = 0.5
    # JEPA2 only. "cosine" is the v2 raw 1-cos objective; "centered_contrast"
    # compares pack-centered readouts (see jepa2_training.contrast_objective).
    alignment: str = "cosine"
    temperature: float = 0.1
    content_weight: float = 1.0
    answer_weight: float = 1.0
    control_fraction: float = 0.0
    min_group: int = 4
    # Order each update's origins by relation family and source before packing,
    # so in-pack contrast negatives share the family (no family-level shortcut).
    pack_by_relation: bool = False


@dataclass
class TrainConfig:
    train: str
    output: str
    dev: str | None = None
    calibration: str | None = None
    test: str | None = None
    model: str = "google/t5gemma-2-1b-1b"
    revision: str = "dd0a2683227859151b1730ca3a63087df5b5f39b"
    init_checkpoint: str | None = None
    # Question-aware encoding (compiler.py): None keeps the init checkpoint's saved mode, a bool
    # switches it for this run and the saved bundles then carry the new mode.
    encoder_questions: bool | None = None
    device: str = "mps"
    precision: str = "float32"
    attention: str = "sdpa"
    optimizer: str = "adamw"
    muon_lr_scale: float = 1.0
    lion_lr_scale: float = 0.2
    lion_weight_decay_scale: float = 5.0
    lion_betas: tuple[float, float] = (0.9, 0.99)
    seed: int = 23
    epochs: int = 1
    choice_permutation: bool = True
    max_steps: int | None = None
    accumulation_steps: int = 8
    record_batch_size: int = 2
    branch_batch_size: int = 8
    tokens_per_step: int = 8192
    tokens_per_batch: int | None = None
    packing: bool = False
    packing_attention: str = "triton"
    max_request_tokens: int = 64000
    max_sequence_tokens: int = 16384
    encoder_lr: float = 1e-5
    decoder_lr: float = 2e-5
    embedding_lr: float = 1e-5
    head_lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.02
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    gradient_checkpointing_min_tokens: int | None = None
    save_checkpoints: bool = True
    save_every: int = 100
    keep_checkpoints: int | None = None
    evaluate_every: int = 100
    gradient_diagnostics_every: int = 100
    allow_inferred_split_groups: bool = False
    dataset_cache_dir: str | None = None
    plan_cost_cache: str | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)
    jepa: JepaConfig = field(default_factory=JepaConfig)
    lora: LoraSettings = field(default_factory=LoraSettings)
    frozen_linear_bf16: bool = False
    regional_compile: bool = False
    shared_attention: str = "reference"
    fp8: str | None = None
    fused_mlp: str | None = None
    # Transformer Engine projections with the HF RMSNorm kept (bongard/te_linear.py):
    # recipes for the MLP GEMMs and for the attention projections, or null.
    te_mlp: str | None = None
    te_attention: str | None = None
    # Exponential moving average of the weights (bongard/training.py WeightAverage): kept
    # beside training, never fed back; checkpoints add ema.pt and ema/. Null disables it.
    ema_decay: float | None = None

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["jepa"] = JepaConfig(**value.get("jepa", {}))
        value["wandb"] = WandbConfig(**value.get("wandb", {}))
        value["lora"] = LoraSettings(**value.get("lora", {}))
        config = cls(**value)
        if type(config.save_checkpoints) is not bool:
            raise ValueError("save_checkpoints must be boolean")
        if config.packing_attention not in {"triton", "flash"}:
            raise ValueError("packing_attention must be triton or flash")
        if config.packing_attention == "flash":
            if not config.packing or config.device != "cuda":
                raise ValueError("packing_attention=flash requires CUDA packing")
            if importlib.util.find_spec("flash_attn") is None:
                raise ImportError("FlashAttention-4 requires: uv sync --extra flash")
        if (
            not isinstance(config.lion_betas, (list, tuple))
            or len(config.lion_betas) != 2
            or any(
                not isinstance(beta, (int, float)) or not math.isfinite(beta) or not 0 <= beta < 1
                for beta in config.lion_betas
            )
        ):
            raise ValueError("lion_betas must contain two finite values in [0, 1)")
        config.lion_betas = tuple(config.lion_betas)
        for name in (
            "epochs",
            "accumulation_steps",
            "record_batch_size",
            "branch_batch_size",
            "tokens_per_step",
            "save_every",
            "evaluate_every",
            "gradient_diagnostics_every",
            "max_request_tokens",
            "max_sequence_tokens",
        ):
            if type(getattr(config, name)) is not int or getattr(config, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if config.max_steps is not None and (
            type(config.max_steps) is not int or config.max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")
        if config.tokens_per_batch is not None and (
            type(config.tokens_per_batch) is not int or config.tokens_per_batch <= 0
        ):
            raise ValueError("tokens_per_batch must be a positive integer")
        if config.gradient_checkpointing_min_tokens is not None and (
            type(config.gradient_checkpointing_min_tokens) is not int
            or config.gradient_checkpointing_min_tokens <= 0 or not config.packing
        ):
            raise ValueError("gradient_checkpointing_min_tokens requires packing and a positive integer")
        if config.keep_checkpoints is not None and (
            type(config.keep_checkpoints) is not int or config.keep_checkpoints <= 0
        ):
            raise ValueError("keep_checkpoints must be a positive integer")
        for name in (
            "encoder_lr",
            "decoder_lr",
            "embedding_lr",
            "head_lr",
            "max_grad_norm",
            "muon_lr_scale",
            "lion_lr_scale",
        ):
            value = getattr(config, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if not math.isfinite(config.lion_weight_decay_scale) or config.lion_weight_decay_scale < 0:
            raise ValueError("lion_weight_decay_scale must be finite and nonnegative")
        if not math.isfinite(config.jepa.coefficient) or config.jepa.coefficient < 0:
            raise ValueError("JEPA coefficient must be nonnegative")
        for ratio in (config.warmup_ratio, config.jepa.warmup_ratio):
            if not math.isfinite(ratio) or not 0 <= ratio <= 1:
                raise ValueError("warmup ratios must lie in [0, 1]")
        if config.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        if config.attention not in {"eager", "sdpa"}:
            raise ValueError("attention must be eager or sdpa")
        if config.optimizer not in {"adamw", "adamw8bit", "adamw8bit_sr", "adamw_sr", "adamwfp8_sr",
                                    "muon", "lion"}:
            raise ValueError("optimizer must be adamw, adamw8bit, adamw8bit_sr, adamw_sr, "
                             "adamwfp8_sr, muon or lion")
        if config.optimizer in {"adamw8bit_sr", "adamw_sr", "adamwfp8_sr"}:
            if config.device != "cuda" or config.precision != "bfloat16":
                raise ValueError("adamw8bit_sr requires CUDA BF16 parameter storage")
            if importlib.util.find_spec("torchao") is None:
                raise ImportError("adamw8bit_sr requires: uv sync --extra fp8")
        if config.optimizer == "muon" and not hasattr(torch.optim, "Muon"):
            raise ImportError("Muon requires a PyTorch version providing torch.optim.Muon")
        if config.optimizer == "adamw8bit":
            if config.device == "mps":
                raise ValueError("bitsandbytes AdamW8bit requires CPU/CUDA; use adamw on MPS")
            if importlib.util.find_spec("bitsandbytes") is None:
                raise ImportError("AdamW8bit requires: uv sync --extra bitsandbytes")
        if type(config.jepa.enabled) is not bool:
            raise ValueError("jepa.enabled must be boolean")
        if config.jepa.objective not in {"paired_views", "jepa2"}:
            raise ValueError("jepa.objective must be paired_views or jepa2")
        if not math.isfinite(config.jepa.view_supervision_mix) or not 0 <= config.jepa.view_supervision_mix <= 1:
            raise ValueError("jepa.view_supervision_mix must be in [0, 1]")
        if config.jepa.objective == "jepa2" and (
            not config.jepa.enabled or config.jepa.data is not None or config.epochs != 1
            or not config.packing or config.choice_permutation or config.lora.enabled
        ):
            raise ValueError("JEPA2 requires one packed full-parameter origin-once course with fixed views")
        if config.jepa.alignment not in {"cosine", "centered_contrast"}:
            raise ValueError("jepa.alignment must be cosine or centered_contrast")
        if config.jepa.alignment != "cosine" and config.jepa.objective != "jepa2":
            raise ValueError("centered_contrast is a JEPA2 objective")
        for name in ("temperature", "content_weight", "answer_weight", "control_fraction"):
            value = getattr(config.jepa, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"jepa.{name} must be finite and nonnegative")
        if config.jepa.temperature <= 0 or config.jepa.control_fraction >= 1:
            raise ValueError("jepa.temperature must be positive and control_fraction below one")
        if type(config.jepa.min_group) is not int or config.jepa.min_group < 2:
            raise ValueError("jepa.min_group must be an integer of at least two")
        if type(config.jepa.pack_by_relation) is not bool or (
                config.jepa.pack_by_relation and config.jepa.objective != "jepa2"):
            raise ValueError("jepa.pack_by_relation is a boolean JEPA2 packing option")
        if config.jepa.data is not None and (
            not config.jepa.enabled or not isinstance(config.jepa.data, str) or not config.jepa.data
        ):
            raise ValueError("jepa.data requires enabled JEPA and a dataset path")
        if type(config.frozen_linear_bf16) is not bool:
            raise ValueError("frozen_linear_bf16 must be boolean")
        if type(config.regional_compile) is not bool:
            raise ValueError("regional_compile must be boolean")
        if type(config.packing) is not bool:
            raise ValueError("packing must be boolean")
        if config.packing and config.device not in {"cpu", "cuda"}:
            raise ValueError("packing supports CPU reference or CUDA sparse attention")
        if type(config.choice_permutation) is not bool:
            raise ValueError("choice_permutation must be boolean")
        if config.plan_cost_cache is not None and (
            not isinstance(config.plan_cost_cache, str) or not config.plan_cost_cache
            or config.epochs != 1 or config.jepa.enabled
        ):
            raise ValueError("plan_cost_cache requires a path and one SFT-only epoch")
        config.wandb.validate()
        config.lora.validate()
        if config.frozen_linear_bf16 and (
            not config.lora.enabled or config.precision != "bfloat16" or config.device != "cuda"
        ):
            raise ValueError("frozen_linear_bf16 requires CUDA BF16 LoRA training")
        if config.regional_compile and (
            config.device != "cuda" or config.precision != "bfloat16" or config.attention != "sdpa"
        ):
            raise ValueError("regional_compile requires CUDA BF16 SDPA training")
        if config.shared_attention not in {"reference", "cuda"}:
            raise ValueError("shared_attention must be reference or cuda")
        if config.shared_attention == "cuda" and config.device != "cuda":
            raise ValueError("shared_attention=cuda requires CUDA")
        if config.fp8 is not None:
            from .float8_training import RECIPES

            if config.fp8 not in RECIPES:
                raise ValueError(f"fp8 must be null or one of {RECIPES}")
            if config.device != "cuda" or config.precision != "bfloat16" or config.lora.enabled:
                raise ValueError("FP8 requires CUDA BF16 full finetuning")
            if importlib.util.find_spec("torchao") is None:
                raise ImportError("FP8 training requires: uv sync --extra fp8")
        if config.fused_mlp is not None:
            from .fused_mlp import RECIPES as FUSED_RECIPES

            if config.fused_mlp not in FUSED_RECIPES:
                raise ValueError(f"fused_mlp must be null or one of {FUSED_RECIPES}")
            if config.device != 'cuda' or config.precision != 'bfloat16' or config.lora.enabled:
                raise ValueError('Fused MLP requires CUDA BF16 full finetuning')
            if importlib.util.find_spec('transformer_engine') is None:
                raise ImportError('Fused MLP requires the Transformer Engine training extra')
        if config.ema_decay is not None and not 0 < config.ema_decay < 1:
            raise ValueError("ema_decay must be null or in (0, 1)")
        if config.te_mlp is not None or config.te_attention is not None:
            from .te_linear import RECIPES as TE_RECIPES

            for recipe in (config.te_mlp, config.te_attention):
                if recipe is not None and recipe not in TE_RECIPES:
                    raise ValueError(f"te_mlp/te_attention must be null or one of {TE_RECIPES}")
            if config.te_mlp is not None and config.fused_mlp is not None:
                raise ValueError("te_mlp and fused_mlp both replace the MLP; choose one")
            if config.device != 'cuda' or config.precision != 'bfloat16' or config.lora.enabled:
                raise ValueError('TE projections require CUDA BF16 full finetuning')
            if importlib.util.find_spec('transformer_engine') is None:
                raise ImportError('TE projections require the Transformer Engine training extra')
        device_for(config.device)
        return config


@dataclass
class Prepared:
    record: dict
    base: Request
    alternate: Request | None
    view_id: str | None
    rejected: list[dict]
    encoder_tokens: int
    decoder_tokens: int


class JoinedJepaDataset(Sequence):
    """Attach JEPA views at the same SFT ID; include JEPA-only records once."""

    def __init__(self, sft, jepa):
        self.sft, self.jepa = sft, jepa
        self.tokens_ready = bool(getattr(sft, "tokens_ready", False)
                                 and getattr(jepa, "tokens_ready", False))
        by_id = {}
        for index, record in enumerate(metadata_records(jepa)):
            record_id = record["id"]
            if record_id in by_id:
                raise DataError(f"duplicate JEPA record id: {record_id}")
            if self.tokens_ready:
                valid = len(jepa.costs[index].as_py()) > 1
            else:
                valid = bool(eligible_views(jepa[index])[0])
            if not valid:
                raise DataError(f"JEPA data has no eligible view: {record_id}")
            by_id[record_id] = index
        if not by_id:
            raise DataError("JEPA data is empty")
        self.pairs = {}
        matched = set()
        for index, record in enumerate(metadata_records(sft)):
            paired = by_id.get(record["id"])
            if paired is not None:
                if not self.tokens_ready:
                    self._check_pair(sft[index], jepa[paired])
                self.pairs[index] = paired
                matched.add(record["id"])
        self.extra = [index for record_id, index in by_id.items() if record_id not in matched]
        self.paired = len(self.pairs)

    @staticmethod
    def _check_pair(primary, secondary):
        def content(record):
            result = {k: v for k, v in record.items()
                      if k not in {"views", "_tokenized", "_token_settings"}}
            result["metadata"] = {k: v for k, v in record["metadata"].items() if k != "training_use"}
            return result
        if content(primary) != content(secondary):
            raise DataError(f"SFT/JEPA paired record differs beyond views: {primary['id']}")

    def __len__(self):
        return len(self.sft) + len(self.extra)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index >= len(self.sft):
            return self.jepa[self.extra[index - len(self.sft)]]
        record = self.sft[index]
        paired_index = self.pairs.get(index)
        if paired_index is None:
            return record
        paired = self.jepa[paired_index]
        self._check_pair(record, paired)
        result = {**record, "views": paired["views"]}
        for key in ("_tokenized", "_token_settings"):
            result.pop(key, None)
            if key in paired:
                result[key] = paired[key]
        return result

    def metadata_records(self):
        yield from metadata_records(self.sft)
        extra = set(self.extra)
        for index, record in enumerate(metadata_records(self.jepa)):
            if index in extra:
                yield record

    def token_cost(self, index, config, epoch):
        if index >= len(self.sft):
            return self.jepa.token_cost(self.extra[index - len(self.sft)], config, epoch)
        paired = self.pairs.get(index)
        return (self.sft.token_cost(index, config, epoch) if paired is None
                else self.jepa.token_cost(paired, config, epoch))

    def verify_compiler(self, compiler):
        self.sft.verify_compiler(compiler)
        self.jepa.verify_compiler(compiler)


def prepare(record, compiler, config: TrainConfig, epoch: int) -> Prepared:
    if config.jepa.objective == "jepa2":
        from .jepa2_training import prepare_pair

        if epoch != 0:
            raise DataError("JEPA2 cache contains only the authorized origin-once epoch")
        item = prepare_pair(record["record"], record["pair"], compiler, saved=record["saved"])
        if (item.encoder_tokens != record["encoder_tokens"]
                or item.decoder_tokens != record["decoder_tokens"]):
            raise DataError("JEPA2 cached cost differs from the actual packed request")
        return item
    saved = record.get("_tokenized")
    if saved is not None:
        from .tokenized_data import restore_request, verify_compiler

        verify_compiler(record["_token_settings"], compiler)
        record = {k: v for k, v in record.items() if k not in {"_tokenized", "_token_settings"}}
    if config.choice_permutation:
        record = permute(record, f"training:{config.seed}:{epoch}")
    # Unsupervised questions cannot alter the supervised question denominator.
    request = dict(record["request"])
    request["questions"] = {q: request["questions"][q] for q in record["targets"]}
    pixels = {}
    if saved is not None:
        base = restore_request(saved["base"], request, compiler, pixel_cache=pixels)
    else:
        base = compiler.compile(request)
    alternate, view_id, rejected = None, None, []
    if config.jepa.enabled:
        candidates, rejected = eligible_views(record)
        if candidates:
            seed = f"{config.seed}:{epoch}:{record['id']}"
            selected = random.Random(seed).choice(candidates)
            local = dict(selected["request"])
            local["questions"] = {
                q: v for q, v in local["questions"].items() if q in record["targets"]
            }
            try:
                if saved is None:
                    alternate = compiler.compile(local)
                else:
                    cached = next(view for view in saved["views"]
                                  if view["view_id"] == selected["view_id"])
                    if cached["request"] is None:
                        raise DataError(cached["reason"])
                    alternate = restore_request(cached["request"], local, compiler, pixel_cache=pixels)
                view_id = selected["view_id"]
            except (DataError, TokenSafetyError) as exc:
                rejected.append({"view_id": selected["view_id"], "reason": str(exc)})
    encoder_tokens = len(base.state)
    if alternate is not None and alternate.state_key != base.state_key:
        encoder_tokens += len(alternate.state)
    decoder_tokens = sum(
        len(q.tokens) for r in (base, alternate) if r is not None for q in r.questions.values()
    )
    return Prepared(record, base, alternate, view_id, rejected, encoder_tokens, decoder_tokens)


def forward_loss(model, item: Prepared, coefficient, *, branch_batch_size=8):
    cache = {}
    base = model(item.base, state_cache=cache, branch_batch_size=branch_batch_size)
    alternate = (
        model(item.alternate, state_cache=cache, branch_batch_size=branch_batch_size)
        if item.alternate is not None
        else None
    )
    result = record_loss(item.record, base, alternate, coefficient, item.view_id)
    return result, base


def forward_losses(model, items, coefficient, *, branch_batch_size=8, view_supervision_mix=0.5,
                   jepa=None):
    """Keep the serial reference for a singleton; batch independent records otherwise."""
    from .jepa2_training import PreparedJepa2, forward_pair_losses

    if items and isinstance(items[0], PreparedJepa2):
        if not all(isinstance(item, PreparedJepa2) for item in items):
            raise ValueError("An update cannot mix legacy and JEPA2 courses")
        options = {} if jepa is None else {
            key: getattr(jepa, key) for key in ("alignment", "temperature", "content_weight",
                                                 "answer_weight", "control_fraction", "min_group")}
        return forward_pair_losses(model, items, coefficient, view_mix=view_supervision_mix,
                                   branch_batch_size=branch_batch_size, **options)
    if len(items) == 1 and not model.packing:
        return [forward_loss(model, items[0], coefficient, branch_batch_size=branch_batch_size)[0]]
    requests, owners = [], []
    for index, item in enumerate(items):
        local = [item.base] + ([item.alternate] if item.alternate is not None else [])
        requests.extend(local)
        owners.extend([index] * len(local))
    predictions = model(requests, state_owners=owners, branch_batch_size=branch_batch_size)
    # One synchronization for the entire physical batch, rather than one per
    # question/readout. The same finite-value contract is checked before losses.
    tensors = [value.reshape(-1) for prediction in predictions
               for output in prediction.values() for value in output.values()]
    if not bool(torch.isfinite(torch.cat(tensors)).all()):
        raise ValueError("Non-finite predictions or readouts")
    predictions = iter(predictions)
    return record_losses([
        (
            item.record,
            next(predictions),
            next(predictions) if item.alternate is not None else None,
            item.view_id,
        )
        for item in items
    ], coefficient)


def record_batches(items, config):
    """Length buckets inside an unchanged optimizer update, with bounded padding.

    The token bound counts padded encoder positions and a conservative decoder
    rectangle; branch batching can reduce the latter further. A large record
    stays intact and executes alone, just as in the original update planner.
    """
    if config.packing:
        if config.jepa.pack_by_relation:
            from .jepa2_training import pack_key

            # Same origins in the same update; only their physical packs change.
            items = sorted(items, key=pack_key)
        group, tokens = [], 0
        for item in items:
            cost = item.encoder_tokens + item.decoder_tokens
            if group and (
                len(group) >= config.record_batch_size
                or tokens + cost > (config.tokens_per_batch or config.tokens_per_step)
            ):
                yield group
                group, tokens = [], 0
            group.append(item)
            tokens += cost
        if group:
            yield group
        return
    if config.record_batch_size == 1:
        yield from ([item] for item in items)
        return

    def dimensions(item):
        requests = [item.base] + ([item.alternate] if item.alternate is not None else [])
        return (
            max(len(r.state) for r in requests),
            max(len(q.tokens) for r in requests for q in r.questions.values()),
            len({r.state_key for r in requests}),
            sum(len(r.questions) for r in requests),
        )

    sizes = {id(item): dimensions(item) for item in items}
    ordered = sorted(items, key=lambda item: tuple(n.bit_length() for n in sizes[id(item)][:2]))
    group = []
    for item in ordered:
        shapes = [sizes[id(member)] for member in [*group, item]]
        enc, dec, states, questions = zip(*shapes)
        padded_tokens = max(enc) * sum(states) + max(dec) * sum(questions)
        if group and (
            len(group) >= config.record_batch_size
            or max(enc) > 2 * min(enc)
            or max(dec) > 2 * min(dec)
            or padded_tokens > (config.tokens_per_batch or config.tokens_per_step)
        ):
            yield group
            group = []
        group.append(item)
    if group:
        yield group


def optimizer_for(model, config):
    embedding = {id(p) for p in model.backbone.get_input_embeddings().parameters()}
    groups = {name: [] for name in ("encoder", "decoder", "embedding", "head")}
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        owner = (
            "embedding"
            if id(p) in embedding
            else "head"
            if name.startswith("head.")
            else "encoder"
            if name.startswith("backbone.encoder.")
            else "decoder"
        )
        groups[owner].append(p)
    parameter_groups = [
        {"params": params, "lr": getattr(config, f"{name}_lr"), "name": name}
        for name, params in groups.items()
        if params
    ]
    options = {"betas": (0.9, 0.95), "weight_decay": config.weight_decay}
    if config.optimizer in {"muon", "lion"}:
        # Use the same hidden matrix boundary for controlled comparisons.
        # Embeddings, norms and the complete discriminative head keep AdamW.
        matrix_algorithm = config.optimizer
        matrix_lr_scale = getattr(config, f"{matrix_algorithm}_lr_scale")
        hidden_weights = {
            id(module.weight)
            for module in model.backbone.modules()
            if isinstance(module, torch.nn.Linear)
        }
        mixed_groups = []
        for group in parameter_groups:
            for algorithm in (matrix_algorithm, "adamw"):
                params = [
                    p
                    for p in group["params"]
                    if (
                        group["name"] in {"encoder", "decoder"}
                        and id(p) in hidden_weights
                        and p.ndim == 2
                    )
                    == (algorithm == matrix_algorithm)
                ]
                if params:
                    name = group["name"]
                    if name in {"encoder", "decoder"}:
                        name = f"{name}_{algorithm}"
                    mixed_groups.append(
                        {
                            **group,
                            "params": params,
                            "name": name,
                            "algorithm": algorithm,
                            "lr": group["lr"]
                            * (matrix_lr_scale if algorithm == matrix_algorithm else 1),
                            "weight_decay": config.weight_decay
                            * (config.lion_weight_decay_scale if algorithm == "lion" else 1),
                        }
                    )
        return MatrixAdamW(
            mixed_groups, weight_decay=config.weight_decay, lion_betas=config.lion_betas
        )
    if config.optimizer == "adamw8bit_sr":
        from torchao.optim import AdamW8bit

        # Library stochastic rounding preserves small updates without an FP32
        # master copy. Large moments are 8-bit; small moments follow BF16 params.
        return AdamW8bit(parameter_groups, **options, bf16_stochastic_round=True)
    if config.optimizer in {"adamw_sr", "adamwfp8_sr"}:
        from torchao.optim import AdamWFp8, _AdamW

        # Same stochastic-rounding update as adamw8bit_sr with BF16 (adamw_sr) or FP8
        # (adamwfp8_sr) moments: the 8-bit dynamic-tree quantization kernel ran at about
        # 2% of memory bandwidth on the B300 (0.25 s per update, 2026-09-28).
        cls = _AdamW if config.optimizer == "adamw_sr" else AdamWFp8
        return cls(parameter_groups, **options, bf16_stochastic_round=True)
    if config.optimizer == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit

        # Quantize moments, not model weights. Keep the native small-tensor
        # threshold (4096); no model surgery or process-global overrides.
        return AdamW8bit(parameter_groups, **options)
    # CUDA's fused path is the one measured in the A100 training benchmarks.
    # Keep the single-tensor implementation on CPU/MPS for portability.
    return torch.optim.AdamW(
        parameter_groups, **options, foreach=False, fused=config.device == "cuda"
    )


def data_manifest(config, splits):
    result = {}
    for name in ("train", "dev", "calibration", "test"):
        path = getattr(config, name)
        if path:
            result[name] = {
                "path": str(Path(path).resolve()),
                "sha256": file_sha256(path),
                **membership(splits[name]),
            }
            registry = Path(str(path) + ".registry.json")
            if registry.exists():
                result[name]["registry"] = str(registry.resolve())
                result[name]["registry_sha256"] = file_sha256(registry)
    return result


def load_plan_costs(path, count, compiler, config, dataset_sha256):
    """Read exact per-record token costs precomputed from this immutable split."""
    if config.epochs != 1 or config.jepa.enabled or not dataset_sha256:
        raise ValueError("plan_cost_cache requires one SFT-only epoch")
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    expected = {
        "format": "bongard-plan-costs-v1-le-u32",
        "records": count,
        "dataset_sha256": dataset_sha256,
        "compiler": compiler.manifest,
        "seed": config.seed,
        "choice_permutation": config.choice_permutation,
        "max_request_tokens": compiler.max_request_tokens,
        "max_sequence_tokens": compiler.max_sequence_tokens,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Plan cost cache does not match the dataset, compiler or course")
    if path.stat().st_size != 4 * count or file_sha256(path) != metadata.get("sha256"):
        raise ValueError("Plan cost cache is incomplete or changed")
    if sys.byteorder != "little":
        raise ValueError("Plan cost cache requires a little-endian host")
    costs = array("I")
    with path.open("rb") as stream:
        costs.fromfile(stream, count)
    if costs.itemsize != 4 or len(costs) != count or any(cost < 1 for cost in costs):
        raise ValueError("Plan cost cache contains invalid token counts")
    return costs


def plan_updates(records, compiler, config, *, dataset_sha256=None):
    tokenized = getattr(records, "tokens_ready", False)
    if tokenized:
        records.verify_compiler(compiler)
    costs = (
        load_plan_costs(config.plan_cost_cache, len(records), compiler, config, dataset_sha256)
        if config.plan_cost_cache and not tokenized else None
    )
    plan = []
    for epoch in range(config.epochs):
        order = list(range(len(records)))
        random.Random(config.seed + epoch).shuffle(order)
        batch, budget = [], 0
        for index in order:
            if tokenized:
                cost = records.token_cost(index, config, epoch)
            elif costs is None:
                item = prepare(records[index], compiler, config, epoch)
                cost = item.encoder_tokens + item.decoder_tokens
            else:
                cost = costs[index]
            if batch and (
                len(batch) == config.accumulation_steps or budget + cost > config.tokens_per_step
            ):
                plan.append((epoch, batch))
                if config.max_steps is not None and len(plan) >= config.max_steps:
                    return plan
                batch, budget = [], 0
            batch.append(index)
            budget += cost
        if batch:
            plan.append((epoch, batch))
        if config.max_steps is not None and len(plan) >= config.max_steps:
            return plan[: config.max_steps]
    return plan[: config.max_steps] if config.max_steps is not None else plan


def rng_state(device):
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "mps": torch.mps.get_rng_state() if device.type == "mps" else None,
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "mps" and state["mps"] is not None:
        torch.mps.set_rng_state(state["mps"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def _fsync_path(path):
    """A successful close is not a persistence acknowledgement on GeeseFS."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_checkpoint(model, optimizer, scheduler, config, manifest, sources, completed, output):
    destination = output / f"step-{completed:06d}"
    temporary = output / f".step-{completed:06d}.tmp"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    state = rng_state(model.device)
    model.save(temporary)
    for split, source in manifest.items():
        if "registry" in source:
            (temporary / "registry").mkdir(exist_ok=True)
            shutil.copyfile(source["registry"], temporary / "registry" / f"{split}.json")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "completed": completed,
            "rng": state,
            "config": asdict(config),
            "data": manifest,
            "sources": sources,
        },
        temporary / "training.pt",
    )
    (temporary / "training.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "data": manifest,
                "sources": sources,
                "completed": completed,
            },
            indent=2,
        )
        + "\n"
    )
    # HF/torch serialize the model and optimizer; the filesystem still needs a
    # durability acknowledgement before the resumable pointer can be advanced.
    for path in temporary.rglob('*'):
        if path.is_file():
            _fsync_path(path)
    for path in sorted((p for p in temporary.rglob('*') if p.is_dir()),
                       key=lambda p: len(p.parts), reverse=True):
        _fsync_path(path)
    _fsync_path(temporary)
    temporary.rename(destination)
    _fsync_path(output)
    latest = output / "latest.json"
    latest.write_text(json.dumps({"checkpoint": destination.name}) + "\n")
    _fsync_path(latest)
    _fsync_path(output)
    restore_rng(state, model.device)
    return destination


def prune_checkpoints(output, keep):
    """Keep recent resumable steps and the best validated step on small disks."""
    if keep is None:
        return
    checkpoints = sorted(
        (path for path in output.glob("step-*")
         if path.is_dir() and path.name[5:].isdigit()),
        key=lambda path: int(path.name[5:]),
    )
    protected = set(checkpoints[-keep:])
    best_file = output / "best.json"
    if best_file.exists():
        protected.add(output / json.loads(best_file.read_text())["checkpoint"])
    for path in checkpoints:
        if path not in protected:
            shutil.rmtree(path)


# Settings that change how an update is executed or where the run is written, not which
# update is taken: a resume may change them (more GPUs, smaller packs, a faster output disk).
RESUMABLE_CHANGES = frozenset({
    "output", "tokens_per_batch", "record_batch_size", "branch_batch_size",
    "gradient_checkpointing", "gradient_checkpointing_min_tokens", "keep_checkpoints",
    "dataset_cache_dir", "ema_decay", "wandb",
})


class WeightAverage:
    """Exponential moving average of the trainable weights in FP32 (on rank 0 only). It never
    feeds back into training; checkpoints carry it as ema.pt (to resume it) and ema/ (a model
    bundle, loadable like any checkpoint)."""

    def __init__(self, model, decay, resume=None):
        self.decay = decay
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        saved = {}
        if resume and (Path(resume) / "ema.pt").exists():
            saved = torch.load(Path(resume) / "ema.pt", map_location="cpu", weights_only=True)
        self.shadow = [(saved[n] if n in saved else p.detach()).to(
            device=p.device, dtype=torch.float32, copy=True) for n, p in self.named]
        self.resumed = bool(saved)

    @torch.no_grad()
    def update(self):
        for shadow, (_, p) in zip(self.shadow, self.named):
            shadow.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @contextmanager
    def swapped(self):
        """The model holds the averaged weights inside, its own weights again afterwards."""
        backup = [p.detach().clone() for _, p in self.named]
        with torch.no_grad():
            for shadow, (_, p) in zip(self.shadow, self.named):
                p.copy_(shadow)
        try:
            yield
        finally:
            with torch.no_grad():
                for value, (_, p) in zip(backup, self.named):
                    p.copy_(value)

    def save(self, model, directory):
        directory = Path(directory)
        torch.save({n: s.cpu() for s, (n, _) in zip(self.shadow, self.named)}, directory / "ema.pt")
        with self.swapped():
            model.save(directory / "ema")


def _recipe(values):
    return {k: v for k, v in values.items() if k not in RESUMABLE_CHANGES}


def shard_update(records, indices, compiler, config, epoch, rank, world):
    """This rank's records of one update. Records go longest first to the least-loaded rank
    by token cost; every rank computes the same assignment and keeps the plan's order."""
    if hasattr(records, "token_cost"):
        costs = [records.token_cost(i, config, epoch) for i in indices]
    else:
        costs = []
        for i in indices:
            item = prepare(records[i], compiler, config, epoch)
            costs.append(item.encoder_tokens + item.decoder_tokens)
    loads, owner = [0] * world, {}
    for position in sorted(range(len(indices)), key=lambda p: (-costs[p], p)):
        target = min(range(world), key=lambda r: (loads[r], r))
        owner[position] = target
        loads[target] += costs[position]
    return [index for position, index in enumerate(indices) if owner[position] == rank]


def _all_reduce_sum(value, device):
    import torch.distributed as dist

    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return float(tensor.item())


def _sum_gradients(parameters, device, bucket_elements=1 << 28):
    """Sum gradients over ranks in FP32 buckets. A parameter that no rank touched keeps grad
    None, so the optimizer skips it exactly as a single process would."""
    import torch.distributed as dist

    # FP32 sums by default; BONGARD_ALLREDUCE_DTYPE=bf16 halves the traffic on slow links.
    dtype = torch.bfloat16 if os.environ.get("BONGARD_ALLREDUCE_DTYPE") == "bf16" else torch.float32
    params = [p for p in parameters if p.requires_grad]
    touched = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=device)
    dist.all_reduce(touched)
    live = [p for p, count in zip(params, touched.tolist()) if count]
    for p in live:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    bucket, size = [], 0
    for p in [*live, None]:
        if p is not None:
            bucket.append(p.grad)
            size += p.grad.numel()
        if bucket and (p is None or size >= bucket_elements):
            flat = torch.cat([g.reshape(-1).to(dtype) for g in bucket])
            dist.all_reduce(flat)
            offset = 0
            for g in bucket:
                g.copy_(flat[offset:offset + g.numel()].view_as(g))
                offset += g.numel()
            bucket, size = [], 0


@contextmanager
def _shared_rng(device, seed):
    """Every rank draws identical random numbers inside, then resumes its own stream."""
    if device.type == "cuda":
        state = torch.cuda.get_rng_state(device)
        torch.cuda.manual_seed(seed)
    else:
        state = torch.get_rng_state()
        torch.manual_seed(seed)
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.set_rng_state(state, device)
        else:
            torch.set_rng_state(state)


_SUM_STATS = (
    "checkpointed_record_batches", "record_batches", "paired_questions",
    "jepa_eligible_questions", "jepa_scope_skipped_questions", "supervised_questions",
    "paired_records", "encoder_tokens", "decoder_tokens", "paired_scopes",
    "view_supervised_scopes", "loss", "decision_loss", "jepa_loss",
)


def _gather_stats(stats, records, device):
    """Whole-update statistics from the ranks' shares; returns the update's record count."""
    import torch.distributed as dist

    values = torch.tensor([float(stats.get(k, 0)) for k in _SUM_STATS] + [float(records)],
                          dtype=torch.float64, device=device)
    dist.all_reduce(values)
    largest = torch.tensor([float(stats.get("max_record_batch_size", 0))],
                           dtype=torch.float64, device=device)
    dist.all_reduce(largest, op=dist.ReduceOp.MAX)
    for key, value in zip(_SUM_STATS, values.tolist()):
        if key in stats or value:
            stats[key] = value if key in ("loss", "decision_loss", "jepa_loss") else int(round(value))
    stats["max_record_batch_size"] = int(largest.item())
    return int(round(values[-1].item()))


def _check_replicas(model, device):
    """Parameters must stay identical on every rank; re-broadcast rank 0's if they drift."""
    import torch.distributed as dist

    params = [p for p in model.parameters() if p.requires_grad]
    probe = torch.stack([p.detach().reshape(-1)[:4096].double().sum() for p in params])
    low, high = probe.clone(), probe.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    drift = int((low != high).sum().item())
    if drift:
        for p in params:
            dist.broadcast(p.data, src=0)
    return drift


def train(config: TrainConfig, *, resume=None, model=None, stop_after=None):
    """stop_after interrupts a planned run at an update boundary (also used by resume tests)."""
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop_after must be a positive update number")
    config = TrainConfig.from_dict(asdict(config))
    with training_runtime(config) as accelerator, attention_backend(config.shared_attention):
        return _train(config, accelerator, resume=resume, model=model, stop_after=stop_after)


def _train(config, accelerator, *, resume, model, stop_after):
    set_seed(config.seed)
    device = accelerator.device
    splits = {}
    for name in ("train", "dev", "calibration", "test"):
        if path := getattr(config, name):
            if name == "train" and config.jepa.objective == "jepa2":
                from .jepa2_dataset import Jepa2Dataset

                splits[name] = Jepa2Dataset(path)
            else:
                splits[name] = read_dataset(path, cache_dir=config.dataset_cache_dir)
    if config.jepa.data:
        jepa_records = read_dataset(config.jepa.data, cache_dir=config.dataset_cache_dir)
        splits["train"] = JoinedJepaDataset(splits["train"], jepa_records)
        if Path(config.train).suffix == Path(config.jepa.data).suffix == ".parquet" and not splits["train"].tokens_ready:
            raise DataError("Run bongard tokenize on both SFT and JEPA Parquets before training")
    check_split_groups(splits)
    if not config.allow_inferred_split_groups:
        for records in splits.values():
            if any(r.get("metadata", {}).get("split_group_inferred") for r in metadata_records(records)):
                raise DataError(
                    "Replace inferred split groups with audited source families before training"
                )
    manifest = data_manifest(config, splits)
    ancestor = resume or config.init_checkpoint
    inherited = checkpoint_sources(ancestor, required=True) if ancestor else []
    sources = course_sources(manifest, inherited)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and ((output / "metrics.jsonl").exists() or list(output.glob("step-*"))):
        raise FileExistsError("Output already contains a run; choose a new directory or resume")
    restored = None
    extended_epochs = False
    if resume:
        # Only load this application's own trusted local checkpoint, never arbitrary pickles.
        restored = torch.load(Path(resume) / "training.pt", map_location="cpu", weights_only=True)
        # Older checkpoints omitted membership snapshots but still recorded path/hash.
        unchanged_data = all(
            name in manifest and all(manifest[name].get(k) == v for k, v in info.items())
            for name, info in restored["data"].items()
        ) and set(restored["data"]) == set(manifest)
        # Restore historical defaults for fields absent from older checkpoints;
        # resuming must not silently enable a newly introduced augmentation.
        restored_config = asdict(
            TrainConfig.from_dict(
                {"record_batch_size": 1, "choice_permutation": False, **restored["config"]}
            )
        )
        extended_epochs = config.epochs > restored_config["epochs"]
        if (
            config.epochs < restored_config["epochs"]
            or _recipe({**restored_config, "epochs": config.epochs}) != _recipe(asdict(config))
            or not unchanged_data
        ):
            raise ValueError(
                "Resume requires identical config and datasets, except epochs may increase; "
                "use init_checkpoint for other changes"
            )
        model = JudgmentModel.load(resume, device=config.device, dtype=getattr(torch, config.precision))
    elif model is None:
        model = (
            JudgmentModel.load(
                config.init_checkpoint,
                device=config.device,
                attention=config.attention,
                upgrade_compiler=True,
                dtype=getattr(torch, config.precision),
            )
            if config.init_checkpoint
            else JudgmentModel.from_base(
                config.model,
                revision=config.revision,
                attention=config.attention,
                dtype=getattr(torch, config.precision),
                max_request_tokens=config.max_request_tokens,
                max_sequence_tokens=config.max_sequence_tokens,
            )
        )
    if config.init_checkpoint and not resume:
        model.compiler = Compiler(
            model.compiler.tokenizer,
            model.backbone.config,
            model.compiler.manifest,
            image_processor=model.compiler.image_processor,
            max_request_tokens=config.max_request_tokens,
            max_sequence_tokens=config.max_sequence_tokens,
            encoder_questions=config.encoder_questions,
        )
        for text_config in (
            model.backbone.config.encoder.text_config,
            model.backbone.config.decoder,
        ):
            if text_config._attn_implementation != config.attention:
                raise ValueError(f"Could not activate requested attention: {config.attention}")
    configure_lora(model, config.lora, output / "lora-base", resume=bool(resume))
    if config.frozen_linear_bf16:
        model.store_frozen_linears_bfloat16()
    model.to(device=device, dtype=getattr(torch, config.precision)).train()
    model.set_gradient_checkpointing(config.gradient_checkpointing)
    model.packing = config.packing
    model.packing_attention = config.packing_attention
    if config.te_mlp or config.te_attention:
        from .te_linear import enable_te_linear

        converted = enable_te_linear(model, mlp=config.te_mlp, attention=config.te_attention)
        print(json.dumps({"te_mlp": config.te_mlp, "te_attention": config.te_attention,
                          "te_modules": converted}), flush=True)
    if config.fp8:
        from .float8_training import enable_float8

        converted = enable_float8(model, config.fp8)
        print(json.dumps({"fp8_recipe": config.fp8, "fp8_linear_modules": converted}), flush=True)
    if config.fused_mlp:
        from .fused_mlp import enable_fused_mlp

        converted = enable_fused_mlp(model, config.fused_mlp)
        print(json.dumps({'fused_mlp_recipe': config.fused_mlp,
                          'fused_mlp_modules': converted}), flush=True)
    if config.regional_compile:
        model.enable_regional_compile()
    optimizer = optimizer_for(model, config)
    print(json.dumps({
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizer_parameters": sum(p.numel() for group in optimizer.param_groups
                                    for p in group["params"]),
        "optimizer": config.optimizer,
        "parameter_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }), flush=True)
    world, rank = accelerator.num_processes, accelerator.process_index
    main = accelerator.is_main_process
    if world == 1:
        model, optimizer = accelerator.prepare(model, optimizer)
    ema = WeightAverage(model, config.ema_decay, resume) if config.ema_decay and main else None
    if ema is not None:
        print(json.dumps({"ema_decay": config.ema_decay, "ema_resumed": ema.resumed}), flush=True)
    try:
        records = splits["train"]
        plan = plan_updates(records, model.compiler, config,
                            dataset_sha256=manifest["train"]["sha256"])
        warmup = math.ceil(len(plan) * config.warmup_ratio)
        print(json.dumps({
            "planned_updates": len(plan),
            "training_records": len(records),
            "epochs": config.epochs,
            "warmup_updates": warmup,
        }), flush=True)

        def lr_scale(step):
            if step < warmup:
                return (step + 1) / max(warmup, 1)
            progress = min(1.0, (step - warmup) / max(1, len(plan) - warmup))
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        completed = 0
        if restored:
            optimizer.load_state_dict(restored["optimizer"])
            scheduler.load_state_dict(restored["scheduler"])
            completed = restored["completed"]
            if extended_epochs:
                if len(plan) <= completed:
                    raise ValueError("Increasing epochs adds no updates; check max_steps")
                # Keep Adam moments and the absolute update index, but evaluate
                # the longer schedule now: the old final LR may already be zero.
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_scale, last_epoch=completed - 1
                )
            restore_rng(restored["rng"], device)
        if main:
            (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
        training_rng = rng_state(device)
        start_tracking(accelerator, config, completed=completed, resume=resume)
        restore_rng(training_rng, device)
        best_file = output / "best.json"
        # JEPA2 runs the authorized origin-once course to its final checkpoint.
        # Its narrow external dev set must not select an earlier stage weight.
        best_nll = math.inf
        if config.jepa.objective != "jepa2" and best_file.exists():
            best_nll = json.loads(best_file.read_text())["dev_nll"]
        jepa_warmup = math.ceil(len(plan) * config.jepa.warmup_ratio)
        last_checkpoint = Path(resume) if resume else None
        first_update = completed
        for step in range(completed, len(plan)):
            step_started = time.monotonic()
            from torch._dynamo.utils import counters

            graphs_before = counters["stats"]["unique_graphs"]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            epoch, indices = plan[step]
            if world > 1:
                indices = shard_update(records, indices, model.compiler, config, epoch, rank, world)
            batch = [prepare(records[i], model.compiler, config, epoch) for i in indices]
            data_seconds = time.monotonic() - step_started
            total_weight = sum(item.record.get("weight", 1.0) for item in batch)
            if world > 1:
                # Every rank weights its records by the whole update, so the summed
                # gradient is exactly the single-process gradient.
                total_weight = _all_reduce_sum(total_weight, device)
            coefficient = config.jepa.coefficient * min(1.0, (step + 1) / max(1, jepa_warmup))
            if not config.jepa.enabled:
                coefficient = 0.0
            optimizer.zero_grad(set_to_none=True)
            stats, rejections, gradient_diagnostics = Counter(), [], None
            jepa2_values, processed = [], []
            started = time.monotonic()
            for group in record_batches(batch, config):
                processed.extend(group)
                checkpointed = config.gradient_checkpointing and (
                    config.gradient_checkpointing_min_tokens is None
                    or sum(i.encoder_tokens + i.decoder_tokens for i in group)
                    >= config.gradient_checkpointing_min_tokens
                )
                if model.gradient_checkpointing != checkpointed:
                    model.set_gradient_checkpointing(checkpointed)
                stats["checkpointed_record_batches"] += int(checkpointed)
                stats["record_batches"] += 1
                stats["max_record_batch_size"] = max(stats["max_record_batch_size"], len(group))
                with accelerator.autocast():
                    results = forward_losses(
                        model, group, coefficient, branch_batch_size=config.branch_batch_size,
                        view_supervision_mix=config.jepa.view_supervision_mix, jepa=config.jepa,
                    )
                    values = torch.stack([torch.stack([result[key] for result in results])
                                          for key in ("loss", "decision_loss", "jepa_loss")])
                    weights = values.new_tensor([item.record.get("weight", 1.0) / total_weight
                                                 for item in group])
                    aggregates = (values * weights[None, :]).sum(1)
                    loss = aggregates[0]
                    if config.jepa.objective == "jepa2":
                        # Copy all per-origin scalars once after the update, not
                        # a device synchronization for every record or source.
                        from .jepa2_training import DIAGNOSTIC_KEYS

                        jepa2_values.append(torch.stack([
                            torch.stack([result[k].float() for k in DIAGNOSTIC_KEYS])
                            for result in results
                        ]).detach())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                for item, result in zip(group, results):
                    if (
                        config.jepa.enabled
                        and (step + 1) % config.gradient_diagnostics_every == 0
                        and gradient_diagnostics is None
                        and result.get("paired_scopes", result["paired_questions"]) > 0
                    ):
                        probe = model.backbone.decoder.norm.weight
                        probe_name = "backbone.decoder.norm.weight"
                        if not probe.requires_grad:
                            probe_name, probe = next(
                                (name, p)
                                for name, p in model.named_parameters()
                                if name.startswith("backbone.decoder.")
                                and ".lora_B." in name
                                and p.requires_grad
                            )
                        primary = torch.autograd.grad(
                            result["decision_loss"], probe, retain_graph=True
                        )[0]
                        auxiliary = (
                            torch.autograd.grad(result["jepa_loss"], probe, retain_graph=True)[0]
                            if result["jepa_loss"].requires_grad
                            else torch.zeros_like(primary)
                        )
                        primary_norm = float(primary.norm())
                        auxiliary_norm = float(auxiliary.norm())
                        gradient_diagnostics = {
                            "parameter": probe_name,
                            "sft_norm": primary_norm,
                            "jepa_norm": auxiliary_norm,
                            "weighted_jepa_to_sft": coefficient * auxiliary_norm / primary_norm
                            if primary_norm
                            else None,
                        }
                    for key in (
                        "paired_questions",
                        "jepa_eligible_questions",
                        "jepa_scope_skipped_questions",
                    ):
                        stats[key] += result[key]
                    stats["supervised_questions"] += len(item.record["targets"])
                    stats["paired_records"] += item.alternate is not None
                    if config.jepa.objective == "jepa2":
                        stats["paired_scopes"] += result["paired_scopes"]
                        stats["view_supervised_scopes"] += result["view_supervised_scopes"]
                    stats["encoder_tokens"] += item.encoder_tokens
                    stats["decoder_tokens"] += item.decoder_tokens
                    rejections.extend({"id": item.record["id"], **r} for r in item.rejected)
                for key, value in zip(("loss", "decision_loss", "jepa_loss"), aggregates.detach().unbind()):
                    stats[key] += value
                accelerator.backward(loss)
                del results, result, loss, values, weights, aggregates
            if world > 1:
                _sum_gradients(model.parameters(), device)
            norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm")
            if world > 1:
                # Stochastic rounding in the optimizer step draws random numbers; the ranks
                # advanced their generators differently, so they share one seed per step.
                with _shared_rng(device, (config.seed * 1_000_003 + step) % 2**63):
                    optimizer.step()
            else:
                optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update()
            if step == first_update:
                parameter_dtypes = Counter(str(p.dtype) for p in model.parameters())
                gradient_dtypes = Counter(str(p.grad.dtype) for p in model.parameters()
                                          if p.grad is not None)
                moment_dtypes = Counter()
                for state in optimizer.state.values():
                    for name in ("exp_avg", "exp_avg_sq", "state1", "state2"):
                        value = state.get(name)
                        if isinstance(value, torch.Tensor):
                            storage = getattr(value, "codes", value)
                            moment_dtypes[str(storage.dtype)] += storage.numel()
                if config.precision == "bfloat16" and (
                    set(parameter_dtypes) != {"torch.bfloat16"}
                    or set(gradient_dtypes) != {"torch.bfloat16"}
                    or any(dtype in moment_dtypes for dtype in ("torch.float32", "torch.float64"))
                ):
                    raise RuntimeError("BF16 training unexpectedly retained high-precision state")
                print(json.dumps({"precision_inventory": {
                    "parameters": dict(parameter_dtypes), "gradients": dict(gradient_dtypes),
                    "optimizer_moment_elements": dict(moment_dtypes),
                }}), flush=True)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize(device)
            completed = step + 1
            # Read three aggregate scalars once per update, not once per record.
            for key in ("loss", "decision_loss", "jepa_loss"):
                stats[key] = float(stats[key])
            update_records = len(batch)
            replica_drift = None
            if world > 1:
                update_records = _gather_stats(stats, update_records, device)
                if completed % 100 == 0:
                    replica_drift = _check_replicas(model, device)
            row = {
                "step": completed,
                "epoch": epoch,
                **stats,
                "records": update_records,
                "weight": total_weight,
                "jepa_coefficient": coefficient,
                "grad_norm": float(norm),
                "seconds": time.monotonic() - started,
                "learning_rates": {g["name"]: float(g["lr"]) for g in optimizer.param_groups},
                "rejected_views": rejections,
                "gradient_diagnostics": gradient_diagnostics,
            }
            if config.jepa.objective == "jepa2":
                from .jepa2_training import training_diagnostics

                row.update(training_diagnostics(processed, torch.cat(jepa2_values).cpu().tolist()))
            if replica_drift is not None:
                row["replica_drift"] = replica_drift
            if world > 1:
                row["processes"] = world
            improved = False
            evaluation_started = time.monotonic()
            evaluated = bool(config.dev) and (
                completed % config.evaluate_every == 0 or completed == len(plan)
                or (stop_after is not None and completed >= stop_after)
            )
            if evaluated and main:
                from .evaluation import evaluate

                evaluation = evaluate(model, splits["dev"], diagnostics=False)
                row["dev"] = evaluation["overall"]
                row["dev_slices"] = evaluation["slices"]
                if ema is not None:
                    with ema.swapped():
                        averaged = evaluate(model, splits["dev"], diagnostics=False)
                    row["dev_ema"] = averaged["overall"]
                    row["dev_ema_slices"] = averaged["slices"]
                # The six-source dev set is a cheap health signal for JEPA2,
                # not the selection rule for this user-requested final epoch.
                improved = (
                    completed == len(plan) if config.jepa.objective == "jepa2"
                    else row["dev"]["nll"] < best_nll
                )
                model.train()
            evaluation_seconds = time.monotonic() - evaluation_started
            stop = stop_after is not None and completed >= stop_after
            checkpoint_started = time.monotonic()
            checkpoint_saved = config.save_checkpoints and (
                completed % config.save_every == 0 or completed == len(plan) or stop or improved
            )
            if checkpoint_saved and not main:
                last_checkpoint = output / f"step-{completed:06d}"
            elif checkpoint_saved:
                last_checkpoint = save_checkpoint(
                    model, optimizer, scheduler, config, manifest, sources, completed, output
                )
                if ema is not None:
                    ema.save(model, last_checkpoint)
                if improved:
                    best_nll = row["dev"]["nll"]
                    best_metadata = {"checkpoint": last_checkpoint.name, "dev_nll": best_nll}
                    if config.jepa.objective == "jepa2":
                        best_metadata["selection"] = "final_epoch_by_user_request"
                    best_file.write_text(json.dumps(best_metadata) + "\n")
                prune_checkpoints(output, config.keep_checkpoints)
            if improved:
                best_nll = row["dev"]["nll"]
            if world > 1 and (evaluated or checkpoint_saved or stop or completed == len(plan)):
                # Rank 0 evaluated or saved alone; the next update starts together. A
                # collective on the training device (barrier() may pick another device).
                _all_reduce_sum(0.0, device)
            step_seconds = time.monotonic() - step_started
            row["timing"] = {
                "data_seconds": data_seconds,
                "train_seconds": row["seconds"],
                "evaluation_seconds": evaluation_seconds,
                "checkpoint_seconds": time.monotonic() - checkpoint_started,
                "step_seconds": step_seconds,
                "compiled_graphs": counters["stats"]["unique_graphs"] - graphs_before,
            }
            row["throughput"] = {
                "tokens_per_second": (stats["encoder_tokens"] + stats["decoder_tokens"]) / step_seconds,
                "records_per_second": update_records / step_seconds,
                "questions_per_second": stats["supervised_questions"] / step_seconds,
            }
            if device.type == "cuda":
                row["gpu"] = {
                    "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
                    "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
                }
            if main:
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if checkpoint_saved:
                        stream.flush()
                        os.fsync(stream.fileno())
                print(json.dumps(row, allow_nan=False), flush=True)
                accelerator.log(
                    scalar_metrics(row), step=completed, log_kwargs={"wandb": {"commit": True}}
                )
            if stop:
                break
        return last_checkpoint
    finally:
        accelerator.unwrap_model(model, keep_fp32_wrapper=False)
