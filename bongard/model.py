"""Native T5Gemma2 hidden states with one shared discriminative head."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from itertools import groupby
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, Gemma3ImageProcessorPil, T5Gemma2Model
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.utils import cached_file

from .attention import QuestionLayout, install_shared_attention
from .compiler import COMPILER_VERSION, Compiler, Request
from .mps_inference import install_mps_inference


def device_for(name: str) -> torch.device:
    if name not in {"cpu", "mps", "cuda"}:
        raise ValueError(
            "Supported devices: cpu, mps, cuda (select a GPU with CUDA_VISIBLE_DEVICES)"
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; select cpu explicitly")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; install a CUDA-enabled PyTorch")
    return torch.device(name)


def question_batches(questions, limit):
    """Complete questions, bounded by row count and a twofold padding ratio."""
    batch = []
    for item in sorted(questions.items(), key=lambda item: len(item[1].tokens)):
        if batch and (len(batch) == limit or len(item[1].tokens) > 2 * len(batch[0][1].tokens)):
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


class JudgmentHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.w = nn.Linear(hidden_size, 1, bias=False)
        self.candidate = nn.Linear(hidden_size, 256, bias=False)
        self.global_readout = nn.Linear(hidden_size, 256, bias=False)

    def forward(self, h, g):
        return self.w(h).squeeze(-1) + (self.candidate(h) * self.global_readout(g)).sum(
            -1
        ) / math.sqrt(256)


def fork_cache(parent: EncoderDecoderCache, batch_size=1) -> EncoderDecoderCache:
    """Private dynamic cache containers, shared differentiable state tensors.

    Pinned Transformers DynamicLayer updates replace tensors with cat/slice;
    they do not mutate parent tensors. Never deepcopy/detach non-leaf tensors.
    Copy both containers before expanding their batch dimension. Prefix tensors
    remain differentiable; expanding a child must never resize its parent.
    """
    child_self = copy.copy(parent.self_attention_cache)
    child_self.layers = [copy.copy(layer) for layer in parent.self_attention_cache.layers]
    child_cross = copy.copy(parent.cross_attention_cache)
    child_cross.layers = [copy.copy(layer) for layer in parent.cross_attention_cache.layers]
    child = copy.copy(parent)
    child.self_attention_cache = child_self
    child.cross_attention_cache = child_cross
    child.is_updated = dict(parent.is_updated)
    if batch_size != 1:
        for layer in [*child_self.layers, *child_cross.layers]:
            if layer.get_seq_length() > 0:
                # A shared state's B=1 K/V need no physical copies. Native
                # dynamic updates replace tensors; these views stay read-only.
                layer.keys = layer.keys.expand(batch_size, -1, -1, -1)
                layer.values = layer.values.expand(batch_size, -1, -1, -1)
    return child


def checkpoint_decoder_layer(function, hidden, positions, mask, ids, cache, use_cache, state):
    """Replay one native layer with private KV containers on each invocation."""
    snapshot = fork_cache(cache) if cache is not None else None

    def run(h):
        private = fork_cache(snapshot) if snapshot is not None else None
        return function(h, positions, mask, ids, private, use_cache, state)

    return checkpoint(run, hidden, use_reentrant=False)


def project_state_kv(attention, state):
    shape = (*state.shape[:-1], -1, attention.head_dim)
    keys = attention.k_norm(attention.k_proj(state).view(shape).transpose(1, 2))
    values = attention.v_proj(state).view(shape).transpose(1, 2)
    return keys, values


class JudgmentModel(nn.Module):
    def __init__(self, backbone: T5Gemma2Model, compiler: Compiler, *, source=None, dtype=None):
        super().__init__()
        # Apply the requested storage dtype to nested text/vision modules too.
        # Autocast alone does not change parameters, accumulated grads or moments.
        dtype = next(backbone.parameters()).dtype if dtype is None else dtype
        self.backbone = backbone.to(dtype=dtype)
        install_shared_attention(backbone)
        install_mps_inference(backbone)
        for config in (
            backbone.config,
            backbone.config.encoder,
            backbone.config.encoder.text_config,
            backbone.config.encoder.vision_config,
            backbone.config.decoder,
        ):
            config.dtype = dtype
        self.head = JudgmentHead(backbone.config.decoder.hidden_size).to(dtype=dtype)
        self.compiler = compiler
        # Tokenizer metadata is fixed by the compiler manifest. Resolve it once,
        # outside tensor execution (AddedToken conversion also breaks Dynamo).
        self.pad_token_id = compiler.tokenizer.pad_token_id
        self.source = source or {}
        self.lora_base = None
        self.lora_settings = None
        self.gradient_checkpointing = False
        self.packing_attention = 'triton'
        self.packing = False
        # Preserve native visual features; learn their projection into the
        # text encoder alongside the instruction-guided judgment task.
        backbone.encoder.vision_tower.requires_grad_(False)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def from_base(
        cls,
        path="google/t5gemma-2-1b-1b",
        *,
        revision=None,
        attention="sdpa",
        local_files_only=False,
        dtype=torch.bfloat16,
        **compiler_options,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            path, revision=revision, local_files_only=local_files_only
        )
        backbone = T5Gemma2Model.from_pretrained(
            path,
            revision=revision,
            dtype=dtype,
            attn_implementation=attention,
            local_files_only=local_files_only,
        )
        # Native local backbone exports may predate processor artifacts. Keep
        # loading those with the model-sized native defaults, as for old bundles.
        has_processor = not Path(path).is_dir() or any(
            (Path(path) / name).is_file()
            for name in ("processor_config.json", "preprocessor_config.json")
        )
        image_processor = (
            Gemma3ImageProcessorPil.from_pretrained(
                path,
                revision=revision,
                local_files_only=local_files_only,
            )
            if has_processor
            else None
        )
        compiler = Compiler(
            tokenizer,
            backbone.config,
            image_processor=image_processor,
            **compiler_options,
        )
        source = {
            "model": str(path),
            "revision": getattr(backbone.config, "_commit_hash", revision),
        }
        for filename in ("config.json", "tokenizer.json"):
            actual = cached_file(str(path), filename, revision=revision, local_files_only=True)
            source[f"{filename}_sha256"] = hashlib.sha256(Path(actual).read_bytes()).hexdigest()
        return cls(
            backbone,
            compiler,
            source=source,
            dtype=dtype,
        )

    def set_gradient_checkpointing(self, enabled: bool):
        self.gradient_checkpointing = enabled
        if enabled:
            self.backbone.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.backbone.decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            for layer in self.backbone.decoder.layers:
                layer._gradient_checkpointing_func = checkpoint_decoder_layer
        else:
            self.backbone.encoder.gradient_checkpointing_disable()
            self.backbone.decoder.gradient_checkpointing_disable()

    def enable_regional_compile(self):
        """Compile repeated text blocks for CUDA training; keep eager evaluation."""
        if hasattr(self, "_compiled_training_regions"):
            return
        from torch._functorch import config as autograd_config
        from torch._inductor import config as inductor_config

        if self.packing_attention == 'flash':
            from . import flash_varlen  # noqa: F401 - register compile-visible FA4 operations

        # Training calls backward outside the forward BF16 autocast context.
        autograd_config.backward_pass_autocast = "off"
        # Periodic gradient diagnostics take two autograd.grad probes with a
        # retained graph before the actual backward. Donated AOT buffers forbid
        # that contract even though ordinary training steps work without it.
        autograd_config.donated_buffer = False
        # This fusion adds length-interval guards to FP8 row/column scaling.
        # Unpadded packs cross those intervals constantly; separate reductions
        # avoid recompiling the entire block for each interval.
        inductor_config.triton.mix_order_reduction = (
            os.environ.get("BONGARD_MIX_ORDER_REDUCTION", "0") == "1"
        )
        torch._dynamo.config.recompile_limit = 32
        regions = [
            *self.backbone.encoder.text_model.layers,
            *(self.backbone.decoder.layers if self.packing else
              (layer.mlp for layer in self.backbone.decoder.layers)),
        ]
        for module in regions:
            module.compile(dynamic=True)
        self._compiled_state_projection = torch.compile(project_state_kv, dynamic=True)
        self._compiled_training_regions = [
            (module, module._compiled_call_impl) for module in regions
        ]
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        for module, compiled in getattr(self, "_compiled_training_regions", ()):
            module._compiled_call_impl = compiled if mode else None
        return self

    def store_frozen_linears_bfloat16(self):
        """Store frozen linear kernels in BF16 while retaining the FP32 base snapshot.

        BF16 autocast already rounds linear operands for CUDA matmul. Keep
        embeddings, normalization, adapters, projector and head at FP32.
        Checkpoint reload starts from the unchanged FP32 frozen backbone.
        """
        kernels = {
            id(module.weight)
            for module in self.backbone.modules()
            if isinstance(module, nn.Linear) and not module.weight.requires_grad
        }
        count = 0
        for parameter in self.backbone.parameters():
            if id(parameter) in kernels:
                count += parameter.numel()
                parameter.data = parameter.data.bfloat16()
        return count

    def ids(self, values):
        return torch.tensor([values], dtype=torch.long, device=self.device)

    def encode(self, tokens, pixel_values=None):
        ids = self.ids(tokens)
        return self.backbone.encoder(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            bongard_encoder_windows={},
            use_cache=False,
            pixel_values=None if pixel_values is None else pixel_values.to(self.device),
        ).last_hidden_state

    def encode_batch(self, requests):
        """Right-padded native encoder batch, including row-ordered image crops."""
        width = max(len(request.state) for request in requests)
        ids = torch.tensor(
            [list(r.state) + [self.pad_token_id] * (width - len(r.state)) for r in requests],
            dtype=torch.long,
            device=self.device,
        )
        lengths = torch.tensor([len(r.state) for r in requests], device=self.device)
        mask = torch.arange(width, device=self.device)[None, :] < lengths[:, None]
        pixels = [r.pixel_values for r in requests if r.pixel_values is not None]
        return self.backbone.encoder(
            input_ids=ids,
            attention_mask=mask.long(),
            bongard_encoder_windows={},
            use_cache=False,
            pixel_values=torch.cat(pixels).to(self.device) if pixels else None,
        ).last_hidden_state

    def state_kv(self, state):
        """Project a shared state once per decoder layer, with native weights.

        Matches pinned T5Gemma2MergedAttention: K is normalized, V is not;
        neither receives RoPE. Shared attention jointly normalizes self and
        cross keys. This cache lives only inside one model forward.
        """
        cross = DynamicCache()
        project = (
            getattr(self, "_compiled_state_projection", project_state_kv)
            if self.training else project_state_kv
        )
        for index, layer in enumerate(self.backbone.decoder.layers):
            keys, values = project(layer.self_attn, state)
            cross.update(keys, values, index)
        cache = EncoderDecoderCache(DynamicCache(config=self.backbone.config.decoder), cross)
        cache.shared_state = True
        return cache

    def decode_batch(self, rows, state, cache=None, *, state_groups=()):
        """Independent complete questions, sharing only the encoded state."""
        width = max(map(len, rows))
        # Construct one rectangular input, avoiding a device upload and padding
        # copy per question. Masks and original token positions stay unchanged.
        ids = torch.tensor(
            [list(row) + [self.pad_token_id] * (width - len(row)) for row in rows],
            dtype=torch.long,
            device=self.device,
        )
        batch_size, width = ids.shape
        lengths = torch.tensor([len(row) for row in rows], device=self.device)
        columns = torch.arange(width, device=self.device)[None, :]
        valid = columns < lengths[:, None]
        attention_mask = valid.long()
        positions = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        shared_attention = bool(state_groups) or (
            cache is not None and getattr(cache, "shared_state", False) and batch_size > 1
        )
        # One question has no inter-question KV reuse. Keep the native kernel
        # in that case; its preprojected state cache is still reused unchanged.
        layout = QuestionLayout(valid, positions) if shared_attention else None
        if layout is not None:
            layout.state_groups = state_groups
        # Native decoder layers, embeddings, RoPE and FFNs remain in charge.
        # Supply empty native mask partitions: SharedMergedAttention constructs
        # only self masks, never a B x T x state_length cross mask.
        empty_mask = torch.empty(0, device=self.device)
        shared_masks = {"full_attention": empty_mask, "sliding_attention": empty_mask}

        def run(h):
            if not state_groups:
                h = h.expand(batch_size, -1, -1)
            # Each batch and checkpoint replay gets empty private self caches.
            # The shared state cache is never extended or mutated.
            working_cache = (
                fork_cache(cache, 1 if state_groups else batch_size) if cache is not None else None
            )
            hidden = self.backbone.decoder(
                input_ids=ids,
                attention_mask=shared_masks if shared_attention else attention_mask,
                position_ids=positions,
                encoder_hidden_states=h,
                encoder_attention_mask=(
                    {"full_attention": empty_mask}
                    if shared_attention
                    else torch.ones(h.shape[:2], dtype=torch.long, device=h.device)
                ),
                past_key_values=working_cache,
                use_cache=cache is not None,
                question_layout=layout,
            ).last_hidden_state
            return hidden

        return run(state)

    def forward(
        self,
        request: Request | list[Request],
        *,
        state_cache=None,
        share_state_kv=True,
        branch_batch_size=8,
        state_owners=None,
    ):
        if (
            isinstance(branch_batch_size, bool)
            or not isinstance(branch_batch_size, int)
            or branch_batch_size < 1
        ):
            raise ValueError("branch_batch_size must be a positive integer")
        if isinstance(request, list):
            if state_cache is not None or not share_state_kv:
                raise ValueError("Record batches own their state cache and require shared K/V")
            return self.forward_batch(request, branch_batch_size, state_owners)
        # Individual validation/inference requests use the native serving path.
        # Packing remains available for an explicit request batch in either mode.
        if self.packing and self.training and state_cache is None:
            return self.forward_batch([request], branch_batch_size)[0]
        # The caller owns a fresh dictionary per record/forward, never across optimizer updates.
        state_cache = {} if state_cache is None else state_cache
        if request.state_key not in state_cache:
            state_cache[request.state_key] = self.encode(request.state, request.pixel_values)
        state = state_cache[request.state_key]
        shared = self.state_kv(state) if share_state_kv else None
        # A long rubric must not pad a batch of short binary questions to its
        # full length. Group similar lengths while retaining the caller's cap.
        # The factor of two bounds padding per row; it is not an input limit.
        readouts = {}
        for group in question_batches(request.questions, branch_batch_size):
            hidden = self.decode_batch([q.tokens for _, q in group], state, shared)
            for (qid, q), row in zip(group, hidden):
                h, g = (
                    row[list(q.candidate_positions)],
                    row[q.decision_position : q.decision_position + 1],
                )
                readouts[qid] = torch.cat((h, g))
        return self.score_readouts(request.questions, readouts)

    def forward_batch(self, requests, branch_batch_size, state_owners=None):
        """Independent records/views in one graph; no persistent computation cache.

        Owners allow a record's JEPA views to reuse identical state, without
        coupling separate records' stochastic encoder evaluations.
        """
        if not requests:
            raise ValueError("A record batch must contain at least one request")
        if self.packing:
            from .packing import forward_packed

            return forward_packed(self, requests, branch_batch_size, state_owners)
        owners = list(range(len(requests))) if state_owners is None else state_owners
        if len(owners) != len(requests):
            raise ValueError("state_owners must match requests")
        states, state_rows, request_rows = [], {}, []
        for owner, request in zip(owners, requests):
            key = (owner, request.state_key)
            if key not in state_rows:
                state_rows[key] = len(states)
                states.append(request)
            request_rows.append(state_rows[key])
        state = self.encode_batch(states)
        # As in serial forward_loss, JEPA shares encoder outputs but each view
        # owns its K/V projection. In particular, nonzero LoRA dropout must not
        # make the two views share a projection's stochastic mask.
        if len(requests) != len(states):
            state = state[request_rows]
        shared = self.state_kv(state)
        questions = {
            (index, qid): q
            for index, request in enumerate(requests)
            for qid, q in request.questions.items()
        }
        readouts = {}
        for batch in question_batches(questions, branch_batch_size):
            # Contiguous state groups let attention read each K/V once without
            # copying it per question. FFNs and projections still run together.
            batch.sort(key=lambda item: item[0][0])
            groups, start = [], 0
            for index, members in groupby(batch, key=lambda item: item[0][0]):
                end = start + len(list(members))
                groups.append((slice(start, end), index, len(requests[index].state)))
                start = end
            hidden = self.decode_batch(
                [q.tokens for _, q in batch], state, shared, state_groups=tuple(groups)
            )
            for (key, q), row in zip(batch, hidden):
                readouts[key] = torch.cat(
                    (
                        row[list(q.candidate_positions)],
                        row[q.decision_position : q.decision_position + 1],
                    )
                )
        scored = self.score_readouts(questions, readouts)
        return [
            {qid: scored[index, qid] for qid in request.questions}
            for index, request in enumerate(requests)
        ]

    def score_readouts(self, questions, readouts):
        # One head and one readout contract for every primitive. Normalization
        # and typed output interpretation remain per logical question.
        candidates = [readouts[qid][:-1] for qid in questions]
        globals_ = [readouts[qid][-1:].expand_as(h) for qid, h in zip(questions, candidates)]
        logits = self.head(torch.cat(candidates), torch.cat(globals_))
        parts = logits.split([len(q.keys) for q in questions.values()])
        return {
            qid: {"logits": part, "readouts": readouts[qid]} for qid, part in zip(questions, parts)
        }

    def save(self, directory):
        from .fused_mlp import hf_state_dict

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(directory / ("adapter" if self.lora_base else "backbone"),
                                      state_dict=hf_state_dict(self.backbone))
        self.compiler.tokenizer.save_pretrained(directory / "tokenizer")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()},
            directory / "head.safetensors",
        )
        import transformers

        manifest = {
            "format": "bongard-v1.3",
            "head_rank": 256,
            "source": self.source,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "compiler": self.compiler.manifest,
            "image_processor": self.compiler.image_processor.to_dict(),
            "limits": {
                "max_sequence_tokens": self.compiler.max_sequence_tokens,
                "max_request_tokens": self.compiler.max_request_tokens,
            },
            "attention": self.backbone.config.decoder._attn_implementation,
            "parameter_dtype": str(next(self.parameters()).dtype).removeprefix("torch."),
            # These non-persistent buffers may have been rounded by the original
            # nested checkpoint dtype before master parameters became FP32.
            # Recomputing sqrt(hidden_size) on reload changes the learned path.
            "embedding_scales": {
                "encoder": float(self.backbone.encoder.text_model.embed_tokens.embed_scale),
                "decoder": float(self.backbone.decoder.embed_tokens.embed_scale),
            },
        }
        if self.lora_base is not None:
            manifest["lora"] = {
                "base": os.path.relpath(self.lora_base, directory.resolve()),
                "settings": self.lora_settings,
            }
        (directory / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, directory, *, device="cpu", attention=None, upgrade_compiler=False, dtype=None):
        directory = Path(directory)
        metadata = json.loads((directory / "bundle.json").read_text())
        if dtype is None:
            dtype = getattr(torch, metadata.get("parameter_dtype", "float32"))
        legacy = metadata["format"] == "bongard-v1.2"
        if legacy and not upgrade_compiler:
            raise ValueError(
                "Independent-Score checkpoint requires a new training course with init_checkpoint; "
                "joint Score cannot silently replace its inference or resume semantics"
            )
        if (
            metadata["format"] not in {"bongard-v1.2", "bongard-v1.3"}
            or metadata["head_rank"] != 256
        ):
            raise ValueError("Unsupported model bundle")
        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
        lora = metadata.get("lora")
        base = (directory / lora["base"]).resolve() if lora else directory / "backbone"
        if lora:
            from .adapters import require_peft

            require_peft()
            if not base.is_dir():
                raise FileNotFoundError(f"Missing frozen backbone: {base}; keep the full LoRA run")
        backbone = T5Gemma2Model.from_pretrained(
            base,
            dtype=dtype,
            attn_implementation=metadata["attention"] if attention is None else attention,
            local_files_only=True,
        )
        manifest = dict(metadata["compiler"])
        if legacy:
            if manifest.get("compiler_version") != "judgment-compiler-v1.2":
                raise ValueError("Unsupported legacy compiler")
            manifest["compiler_version"] = COMPILER_VERSION
        image_processor = (
            Gemma3ImageProcessorPil.from_dict(metadata["image_processor"])
            if "image_processor" in metadata
            else None
        )
        compiler = Compiler(
            tokenizer,
            backbone.config,
            manifest,
            image_processor=image_processor,
            **metadata["limits"],
        )
        model = cls(backbone, compiler, source=metadata["source"], dtype=dtype)
        if "embedding_scales" in metadata:
            for name, embedding in (
                ("encoder", model.backbone.encoder.text_model.embed_tokens),
                ("decoder", model.backbone.decoder.embed_tokens),
            ):
                embedding.embed_scale.fill_(metadata["embedding_scales"][name])
        if lora:
            from peft import PeftConfig, set_peft_model_state_dict

            adapter = directory / "adapter"
            config = PeftConfig.from_pretrained(adapter, local_files_only=True)
            config.inference_mode = False
            model.backbone.add_adapter(config)
            # PEFT restores modules_to_save as well as LoRA matrices. The
            # pinned Transformers loader omits the projector key conversion.
            loaded = set_peft_model_state_dict(
                model.backbone,
                {
                    key.removeprefix("base_model.model."): value
                    for key, value in load_file(adapter / "adapter_model.safetensors").items()
                },
            )
            trainable = {name for name, p in model.backbone.named_parameters() if p.requires_grad}
            missing = trainable.intersection(loaded.missing_keys)
            if missing or loaded.unexpected_keys:
                raise ValueError(
                    f"Incomplete LoRA checkpoint: missing={sorted(missing)}, "
                    f"unexpected={loaded.unexpected_keys}"
                )
            model.lora_base = base
            model.lora_settings = lora["settings"]
        model.head.load_state_dict(load_file(directory / "head.safetensors"))
        return model.to(device_for(device))
