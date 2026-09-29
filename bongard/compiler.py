"""One compiler for training, evaluation and serving. No labels or routing IDs in tokens."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from torch import Tensor
from transformers import Gemma3ImageProcessorPil

from .data import DataError, outcome_keys, validate_request
from .token_safety import (
    BoundaryBuilder,
    GuardedTokenizer,
    config_control_ids,
    declared_special_ids,
)
from .vision import compile_images

COMPILER_VERSION = "judgment-compiler-v1.3"


def entry(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Question:
    keys: tuple[str, ...]
    tokens: tuple[int, ...]
    candidate_positions: tuple[int, ...]
    decision_position: int
    instruction_tokens: int
    candidate_starts: tuple[int, ...] = ()


@dataclass(frozen=True)
class Request:
    state: tuple[int, ...]
    questions: dict[str, Question]
    input_tokens: int
    images: tuple[bytes, ...] = field(default=(), repr=False)
    pixel_values: Tensor | None = field(default=None, compare=False, repr=False)

    @property
    def state_key(self):
        return self.state, self.images


class Compiler:
    def __init__(
        self,
        tokenizer,
        config,
        manifest=None,
        *,
        image_processor=None,
        max_request_tokens=64000,
        max_sequence_tokens=32768,
        encoder_questions=None,
    ):
        # Question-aware encoding: the request's question instructions (never the candidates)
        # are appended to the encoder input, so the encoder can read a long state for what the
        # questions ask instead of producing one question-agnostic summary. The mode is part of
        # the saved compiler manifest, so a checkpoint carries its own input format. None keeps
        # the saved mode (False for bundles written before the flag existed); an explicit bool
        # switches it, which a training run may do when starting from an older checkpoint.
        saved_mode = bool(manifest.get("encoder_questions", False)) if manifest else False
        self.encoder_questions = saved_mode if encoder_questions is None else bool(encoder_questions)
        self.tokenizer = tokenizer
        self.vision_config = config.encoder
        # Old text bundles retained the native vision configuration and weights.
        # Their default uses the native Gemma3 PIL normalization and model size;
        # from_base loads the actual pinned processor settings instead.
        self.image_processor = image_processor or Gemma3ImageProcessorPil(
            size={
                "height": config.encoder.vision_config.image_size,
                "width": config.encoder.vision_config.image_size,
            },
        )
        backend = tokenizer.backend_tokenizer
        controls = config_control_ids(config.to_dict()) | set(tokenizer.all_special_ids)
        controls |= declared_special_ids(backend)
        vocab = backend.get_vocab()
        if manifest is None:
            unused = sorted(
                (tid, token)
                for token, tid in vocab.items()
                if re.fullmatch(r"<unused\d+>", token) and tid not in controls
            )
            if len(unused) < 2:
                raise DataError(
                    "Two safe existing unused token IDs are required; no vocabulary resize"
                )
            manifest = {
                "markers": dict(
                    zip(("candidate_end", "decision_end"), [tid for tid, _ in unused[:2]])
                )
            }
        self.guard = GuardedTokenizer(backend, manifest["markers"], controls)
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id
        if self.bos is None or self.eos is None:
            raise DataError("The original tokenizer must declare BOS and EOS")
        fingerprint = hashlib.sha256(backend.to_str().encode()).hexdigest()
        self.manifest = {
            "compiler_version": COMPILER_VERSION,
            "tokenizer_sha256": fingerprint,
            "markers": self.guard.markers,
            "spellings": {k: backend.id_to_token(v) for k, v in self.guard.markers.items()},
            "control_ids": sorted(controls),
            "encoder_questions": self.encoder_questions,
        }
        if "tokenizer_sha256" in manifest:
            expected = {**manifest, "encoder_questions": self.encoder_questions}
            if expected != self.manifest:
                raise DataError("Tokenizer, controls or compiler differ from the saved manifest")
        limit = min(
            config.encoder.text_config.max_position_embeddings,
            config.decoder.max_position_embeddings,
        )
        self.max_sequence_tokens = min(max_sequence_tokens, limit)
        self.max_request_tokens = max_request_tokens
        if self.max_sequence_tokens < 1 or self.max_request_tokens < 1:
            raise DataError("Token budgets must be positive")
        max_id = max(vocab.values())
        if max_id >= min(config.encoder.text_config.vocab_size, config.decoder.vocab_size):
            raise DataError("Tokenizer IDs exceed original model vocabulary")

    def compile(self, request: dict) -> Request:
        validate_request(request)
        questions = {qid: self.question(q) for qid, q in request["questions"].items()}
        preface = self.encoder_preface(request["questions"]) if self.encoder_questions else ""
        return self._compile(request["state"], questions, request.get("images"), preface)

    @staticmethod
    def encoder_preface(questions) -> str:
        """The distinct question instructions, in request order, as one encoder-side block.
        Candidates and criteria stay in the decoder; rotation variants share their source's
        instructions, so duplicates collapse and the state key stays stable."""
        seen, texts = set(), []
        for q in questions.values():
            text = q.get("instructions")
            if isinstance(text, str) and text.strip() and text not in seen:
                seen.add(text)
                texts.append(text)
        return "\n\nquestions: " + entry(texts) if texts else ""

    def compile_representation(self, content, scope: str, *, predict=False, images=None) -> Request:
        """A scoped latent query, with no invented class outcomes or labels.

        The native decoder processes the query against only this view's encoder
        state. Its final marker is a tied-weight predictor when predict=True;
        the observed target is independently encoded, never appended to it.
        """
        if not isinstance(scope, str) or not scope.strip() or type(predict) is not bool:
            raise DataError("A representation requires a nonempty scope and boolean predict")
        instruction = {
            "operation": "predict_scoped_content" if predict else "encode_observed_content",
            "scope": scope,
        }
        builder = BoundaryBuilder(self.guard)
        builder.ids.append(self.bos)
        builder.content("representation: " + entry(instruction) + "\n")
        builder.marker("decision_end")
        piece = builder.finish()
        question = Question((), piece.ids, (), piece.decision_positions[0],
                            len(self.guard.encode_content(entry(instruction))))
        return self._compile(content, {"representation": question}, images)

    def _compile(self, content, questions, image_inputs=None, preface="") -> Request:
        body = (self.bos, *self.guard.encode_content(entry(content)))
        pixel_values, images, tail = None, (), ()
        if image_inputs:
            visual_tokens, pixel_values, images = compile_images(
                image_inputs,
                self.tokenizer,
                self.vision_config,
                self.image_processor,
            )
            tail = (*self.guard.encode_content("\n\n"), *visual_tokens)
        state = (*body, *tail, self.eos)
        if preface:
            # The question preface is an aid, not content: a request that would no longer fit
            # the token budgets with it is compiled without it. The rule is deterministic, so
            # training, evaluation and serving agree; it only affects near-limit states.
            aware = (*body, *self.guard.encode_content(preface), *tail, self.eos)
            longest = max((len(q.tokens) for q in questions.values()), default=0)
            asked = sum(len(q.tokens) for q in questions.values())
            if (len(aware) + longest <= self.max_sequence_tokens
                    and len(aware) + asked <= self.max_request_tokens):
                state = aware
        total = len(state)
        for q in questions.values():
            if len(state) + len(q.tokens) > self.max_sequence_tokens:
                raise DataError("state + question exceeds token budget; input was not truncated")
            total += len(q.tokens)
        if total > self.max_request_tokens:
            raise DataError("Logical request exceeds token budget; candidates were not removed")
        return Request(state, questions, total, images, pixel_values)

    def question(self, q: dict) -> Question:
        builder = BoundaryBuilder(self.guard)
        builder.ids.append(self.bos)
        builder.content(f"type: {q['type']}\n")
        if "instructions" in q:
            builder.content("instructions: " + entry(q["instructions"]) + "\n")
        instruction_tokens = (
            len(self.guard.encode_content(entry(q["instructions"]))) if "instructions" in q else 0
        )
        keys = outcome_keys(q)

        # Primitive semantics stay in the input; all outcomes share one sequence.
        # Score names expose the supplied ordinal positions, not target values.
        criteria = (
            dict(zip(keys, q["criteria"])) if q["type"] == "score" else q.get("criteria") or {}
        )
        starts = []
        for key in keys:
            starts.append(len(builder.ids))
            builder.content("candidate: " + entry({"name": key, "description": criteria.get(key)}))
            builder.marker("candidate_end")
            builder.content("\n")
        builder.marker("decision_end")
        piece = builder.finish()
        return Question(
            tuple(keys),
            piece.ids,
            piece.candidate_positions,
            piece.decision_positions[0],
            instruction_tokens,
            tuple(starts),
        )
