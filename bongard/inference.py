"""Single-view inference and the fixed local System One numeric contract."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import __version__
from .attention import _BACKEND
from .data import DataError, finite_json, outcome_keys, validate_question
from .model import JudgmentModel, checkpoint_config_path

POSTPROCESS_VERSION = "systemone-local-v1.2"


def _temperature_key_ok(key):
    """`choice`, `noul`, `score`, or `<kind>:<option count>` such as `choice:4`."""
    kind, _, count = str(key).partition(":")
    if kind not in {"choice", "noul", "score"}:
        return False
    return count == "" or (count.isdigit() and int(count) >= 2)


def validate_temperatures(values):
    if not isinstance(values, dict) or not all(_temperature_key_ok(k) for k in values):
        raise ValueError("Temperature keys must be choice, noul or score, "
                         "optionally suffixed with :<option count> (e.g. choice:4)")
    normalized = {}
    for kind, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Temperatures must be finite and positive")
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError("Temperatures must be finite and positive") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Temperatures must be finite and positive")
        normalized[kind] = value
    return normalized


def temperature_for(temperatures, kind, option_count):
    """The temperature for one question: the option-count entry (`choice:4`) when fitted,
    else the primitive's entry, else 1.0. Calibration differs by option count because the
    chance level does, which is why the fitted file may carry both."""
    return temperatures.get(f"{kind}:{option_count}", temperatures.get(kind, 1.0))


def probabilities(logits, temperature=1.0):
    """Stable inference probabilities for every finite positive float64 temperature."""
    temperature = validate_temperatures({"choice": temperature})["choice"]
    if logits.ndim != 1 or not logits.numel() or not torch.isfinite(logits).all():
        raise ValueError("Expected a nonempty vector of finite logits")
    # Transfer before casting. MPS has no float64; a combined device/dtype
    # conversion in PyTorch 2.14 can silently produce zeros instead of logits.
    z = logits.detach().cpu().double()
    if temperature >= 1:
        # Scale first so opposite, large finite logits cannot overflow on subtraction.
        z = z / temperature
        z = z - z.max()
    else:
        # Shift first: tiny temperatures can overflow negative gaps to -inf, but the
        # maximum stays exactly zero. Those -inf terms correctly have probability zero.
        z = (z - z.max()) / temperature
    p = torch.softmax(z, dim=-1)
    if not torch.isfinite(p).all() or (p < 0).any() or not p.sum() > 0:
        raise ValueError("Temperature scaling did not produce finite probabilities")
    return p / p.sum()


def answer(question, logits, temperature=1.0):
    validate_question(question, "question")
    kind = question["type"]
    validate_temperatures({kind: temperature})
    keys = outcome_keys(question)
    if logits.ndim != 1 or logits.numel() != len(keys) or not torch.isfinite(logits).all():
        raise ValueError("Expected finite logits for every valid outcome")
    return answer_from_probabilities(question, probabilities(logits, temperature).tolist())


def answer_from_probabilities(question, p):
    """The answer for a validated question from probabilities in outcome_keys order."""
    kind = question["type"]
    keys = outcome_keys(question)
    if kind == "noul":
        return {"type": kind, "noul": p[0]}
    mode = max(range(len(p)), key=p.__getitem__)
    result = {"type": kind, "probabilities": dict(zip(keys, p))}
    if kind == "choice":
        confidence = 1.0 if len(p) == 1 else (p[mode] - 1 / len(p)) / (1 - 1 / len(p))
        result.update(choice=keys[mode], confidence=max(0.0, confidence))
    else:
        center = (len(p) - 1) / 2
        uniform_distance = sum(abs(i - center) for i in range(len(p))) / len(p)
        distance = sum(v * abs(i - mode) for i, v in enumerate(p))
        result.update(
            score=sum(i * v for i, v in enumerate(p)),
            legend=dict(zip(keys, question["criteria"])),
            confidence=max(0.0, 1 - distance / uniform_distance),
        )
    finite_json(result, "answer")
    return result


def rotation_variants(questions):
    """Each multi-option Choice question in every cyclic rotation of its options.

    Returns the questions plus the rotated copies under fresh ids, and for every original
    id its variant ids (the original first). Any position-only preference of the model
    cancels in the average over all rotations: each option takes each position once.
    """
    expanded = dict(questions)
    groups = {}
    for qid, q in questions.items():
        groups[qid] = [qid]
        keys = list(q.get("criteria") or {}) if q["type"] == "choice" else []
        for i in range(1, len(keys)):
            name = f"{qid}#rotation{i}"
            while name in expanded:
                name += "#"
            expanded[name] = {**q, "criteria": {k: q["criteria"][k] for k in keys[i:] + keys[:i]}}
            groups[qid].append(name)
    return expanded, groups


class Predictor:
    def __init__(
        self,
        model: JudgmentModel,
        *,
        model_id=f"bongard-{__version__}",
        temperatures=None,
        calibration_groups=(),
        release_date=None,
        branch_batch_size=8,
        state_cache_mib=0,
        rotations=False,
    ):
        if type(rotations) is not bool:
            raise ValueError("rotations must be a boolean")
        self.rotations = rotations
        if type(branch_batch_size) is not int or branch_batch_size < 1:
            raise ValueError("branch_batch_size must be a positive integer")
        self.branch_batch_size = branch_batch_size
        if type(state_cache_mib) is not int or state_cache_mib < 0:
            raise ValueError("state_cache_mib must be a nonnegative integer")
        self.state_cache_limit = state_cache_mib * 1024 * 1024
        self._state_cache = OrderedDict()
        self._state_cache_bytes = 0
        self._state_cache_signature = None
        self._state_cache_hits = self._state_cache_misses = 0
        self.model = model.eval()
        self.model_id = model_id
        self.temperatures = validate_temperatures(temperatures or {})
        if not isinstance(calibration_groups, (list, tuple)) or any(
            not isinstance(group, str) or not group for group in calibration_groups
        ):
            raise ValueError("Calibration groups must be nonempty strings")
        self.calibration_groups = tuple(calibration_groups)
        self.metadata = {
            "name": model_id,
            "description": (
                f"Local T5Gemma2 judgment model served by Bongard {__version__}; "
                "release_date is the local checkpoint export date or in-memory serving date."
            ),
            "release_date": release_date or datetime.now(timezone.utc).date().isoformat(),
        }

    @classmethod
    def load(
        cls,
        checkpoint,
        *,
        device="cpu",
        temperatures=None,
        model_id=None,
        branch_batch_size=8,
        state_cache_mib=0,
        rotations=False,
    ):
        calibration_groups = ()
        if temperatures is not None:
            values = json.loads(Path(temperatures).read_text())
            if "temperatures" in values:
                from .evaluation import model_fingerprint

                if values.get("model_sha256") != model_fingerprint(checkpoint):
                    raise ValueError("Calibration belongs to a different checkpoint")
                calibration_groups = values.get("calibration_groups", ())
                values = values["temperatures"]
        else:
            values = {}
        return cls(
            JudgmentModel.load(checkpoint, device=device),
            temperatures=values,
            model_id=model_id or f"bongard-{Path(checkpoint).name}",
            calibration_groups=calibration_groups,
            branch_batch_size=branch_batch_size,
            state_cache_mib=state_cache_mib,
            rotations=rotations,
            release_date=datetime.fromtimestamp(
                checkpoint_config_path(checkpoint).stat().st_mtime, timezone.utc
            )
            .date()
            .isoformat(),
        )

    def clear_state_cache(self):
        self._state_cache.clear()
        self._state_cache_bytes = 0
        self._state_cache_signature = None

    def state_cache_info(self):
        return {
            "entries": len(self._state_cache),
            "bytes": self._state_cache_bytes,
            "limit_bytes": self.state_cache_limit,
            "hits": self._state_cache_hits,
            "misses": self._state_cache_misses,
        }

    @staticmethod
    def _state_bytes(key, state):
        # Bound retained tensor + input payload, not allocator/process RSS.
        tokens, images = key
        return state.numel() * state.element_size() + 36 * len(tokens) + sum(map(len, images))

    def _cached_state(self, compiled):
        encoder = self.model.backbone.encoder
        if not self.state_cache_limit or self.model.training or encoder.training:
            if self._state_cache:
                self.clear_state_cache()
            return None
        device = self.model.device.type
        signature = (
            id(encoder),
            id(self.model.compiler),
            encoder.config.text_config._attn_implementation,
            encoder.config.vision_config._attn_implementation,
            _BACKEND.get(),
            json.dumps(self.model.compiler.image_processor.to_dict(), sort_keys=True),
            torch.is_autocast_enabled(device),
            torch.get_autocast_dtype(device),
            tuple(
                (id(p), p._version, p.device, p.dtype)
                for p in (*encoder.parameters(), *encoder.buffers())
            ),
            tuple(
                (tuple(m.active_adapters), m.disable_adapters, tuple(sorted(m.scaling.items())))
                for m in encoder.modules()
                if hasattr(m, "lora_A")
            ),
        )
        if signature != self._state_cache_signature:
            self.clear_state_cache()
            self._state_cache_signature = signature
        key = compiled.state_key
        if key in self._state_cache:
            self._state_cache_hits += 1
            self._state_cache.move_to_end(key)
            return {key: self._state_cache[key]}
        self._state_cache_misses += 1
        return {}

    def _retain_state(self, compiled, state_cache):
        key = compiled.state_key
        if state_cache is None or key in self._state_cache:
            return
        state = state_cache[key]
        size = self._state_bytes(key, state)
        if size > self.state_cache_limit:
            return
        while self._state_cache_bytes + size > self.state_cache_limit:
            old_key, old_state = self._state_cache.popitem(last=False)
            self._state_cache_bytes -= self._state_bytes(old_key, old_state)
        self._state_cache[key] = state
        self._state_cache_bytes += size

    @torch.inference_mode()
    def predict(self, request):
        if request.get("model", self.model_id) != self.model_id:
            raise DataError(f"Unknown model; expected {self.model_id}")
        questions, compiled = request["questions"], None
        groups = {qid: [qid] for qid in questions}
        if self.rotations:
            expanded, rotated = rotation_variants(questions)
            if len(expanded) > len(questions):
                try:
                    compiled = self.model.compiler.compile({**request, "questions": expanded})
                    questions, groups = expanded, rotated
                except DataError:
                    pass  # over the token budget with rotations: answer in the given order
        if compiled is None:
            compiled = self.model.compiler.compile(request)
        # Usage counts the request as sent; rotated copies are an inference detail.
        input_tokens = compiled.input_tokens - sum(
            len(compiled.questions[name].tokens) for names in groups.values() for name in names[1:]
        )
        state_cache = self._cached_state(compiled) if self.state_cache_limit else None
        predictions = self.model(
            compiled,
            state_cache=state_cache,
            # A single complete question has no cross-question reuse and no
            # generation continuation. Use the existing native cache-free path.
            share_state_kv=len(compiled.questions) > 1,
            branch_batch_size=self.branch_batch_size,
        )
        self._retain_state(compiled, state_cache)
        logits = [predictions[qid]["logits"] for qid in questions]
        if logits[0].device.type != "cpu":
            # One readback before any scalar checks or probability processing.
            # Keep the CPU float64 conversion in probabilities(), after transfer.
            sizes = [z.numel() for z in logits]
            packed = logits[0] if len(logits) == 1 else torch.cat(logits)
            logits = packed.cpu().split(sizes)
        logits = dict(zip(questions, logits))
        answers = {}
        for qid, q in request["questions"].items():
            temperature = temperature_for(self.temperatures, q["type"], len(outcome_keys(q)))
            if len(groups[qid]) == 1:
                answers[qid] = answer(q, logits[qid], temperature)
                continue
            average = dict.fromkeys(outcome_keys(q), 0.0)
            for name in groups[qid]:
                p = probabilities(logits[name], temperature).tolist()
                for key, value in zip(outcome_keys(questions[name]), p):
                    average[key] += value / len(groups[qid])
            answers[qid] = answer_from_probabilities(q, [average[k] for k in outcome_keys(q)])
        # The head scores the candidates directly. No token is generated.
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
