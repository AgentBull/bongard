"""ID-level protection for untrusted text plus compiler-owned readout positions.

Works with a tokenizers.Tokenizer backend (or an equivalent tested interface).
No vocabulary resize, no model weights, and no tokenizer mutation. This is a
compiler component, NOT the complete Jev request compiler.

A protected ID in untrusted text is replaced by existing UTF-8 byte-token IDs
for its literal spelling. If the tokenizer cannot round-trip that replacement,
fail closed; never drop the text or substitute UNK. Exact raw input is retained
outside this component; decode equality is in the tokenizer's normalization
space, not a promise to reverse arbitrary tokenizer normalization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


class TokenSafetyError(ValueError):
    pass


def config_control_ids(config: Any) -> set[int]:
    """Collect declared token-ID controls; include the model's default EOI ID."""
    out: set[int] = {256000}  # T5Gemma2TextScaledWordEmbedding default override.

    def walk(value: Any):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str) and key.endswith(
                    ("_token_id", "_token_index", "_token_ids")
                ):
                    vals = child if isinstance(child, list) else [child]
                    out.update(x for x in vals if type(x) is int and x >= 0)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(config)
    return out


def declared_special_ids(backend: Any) -> set[int]:
    raw = json.loads(backend.to_str())
    return {t["id"] for t in raw.get("added_tokens", []) if t.get("special") is True}


class GuardedTokenizer:
    def __init__(self, backend: Any, markers: dict[str, int], control_ids: Iterable[int]):
        if set(markers) != {"candidate_end", "decision_end"}:
            raise TokenSafetyError("Both logical readout marker IDs must be declared")
        if (
            any(type(v) is not int or v < 0 for v in markers.values())
            or len(set(markers.values())) != 2
        ):
            raise TokenSafetyError("Marker IDs must be distinct nonnegative integers")
        controls = set(control_ids) | declared_special_ids(backend)
        if controls & set(markers.values()):
            raise TokenSafetyError(
                "A marker ID conflicts with an execution control or EOI override"
            )
        vocab = backend.get_vocab()
        for tid in markers.values():
            if tid not in vocab.values():
                raise TokenSafetyError(f"Unknown marker ID {tid}")
        self.backend = backend
        self.markers = dict(markers)
        self.protected = controls | set(markers.values())
        self.byte_ids: dict[int, int] = {}
        for byte in range(256):
            name = f"<0x{byte:02X}>"
            tid = vocab.get(name)
            if tid is None or tid in self.protected:
                raise TokenSafetyError(
                    "Safe existing UTF-8 byte vocabulary unavailable; no silent fallback"
                )
            self.byte_ids[byte] = tid

    def encode_content(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Serialize structured content before encoding")
        encoded = self.backend.encode(text, add_special_tokens=False)
        ids, offsets = list(encoded.ids), list(encoded.offsets)
        if len(ids) != len(offsets):
            raise TokenSafetyError("Token offsets unavailable")
        if not any(tid in self.protected for tid in ids):
            return ids
        safe: list[int] = []
        for tid, (start, end) in zip(ids, offsets):
            if tid not in self.protected:
                safe.append(tid)
                continue
            literal = self.backend.id_to_token(tid)
            # Reject unknown-token substitution, stripping AddedTokens, offset
            # ambiguity, and nonliteral control effects rather than lose input.
            if not literal or not 0 <= start < end <= len(text) or text[start:end] != literal:
                raise TokenSafetyError("Protected token is not a verified literal source span")
            safe.extend(self.byte_ids[b] for b in literal.encode("utf-8"))
        if set(safe) & self.protected:
            raise TokenSafetyError("Protected ID survived neutralization")
        before = self.backend.decode(ids, skip_special_tokens=False)
        after = self.backend.decode(safe, skip_special_tokens=False)
        if before != after:
            raise TokenSafetyError("Byte-token replacement changes tokenizer-decoded text")
        return safe


@dataclass(frozen=True)
class CompiledPiece:
    ids: tuple[int, ...]
    candidate_positions: tuple[int, ...]
    decision_positions: tuple[int, ...]


class BoundaryBuilder:
    """Assemble text and trusted markers without rescanning user strings."""

    def __init__(self, guard: GuardedTokenizer):
        self.guard = guard
        self.ids: list[int] = []
        self.positions: dict[str, list[int]] = {name: [] for name in guard.markers}

    def content(self, text: str) -> None:
        self.ids.extend(self.guard.encode_content(text))

    def marker(self, name: str) -> None:
        if name not in self.positions:
            raise TokenSafetyError("Unknown logical marker")
        self.positions[name].append(len(self.ids))
        self.ids.append(self.guard.markers[name])

    def finish(self) -> CompiledPiece:
        for name, tid in self.guard.markers.items():
            actual = [i for i, x in enumerate(self.ids) if x == tid]
            if actual != self.positions[name]:
                raise TokenSafetyError("Untrusted content forged a readout position")
        return CompiledPiece(
            tuple(self.ids),
            tuple(self.positions["candidate_end"]),
            tuple(self.positions["decision_end"]),
        )
