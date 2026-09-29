"""Probability supervision and conditional two-view JEPA losses.

No text-generation loss, teacher network, reinforcement stage, or trajectory
regularizer. This module does not implement a T5Gemma2 trainer or cache backend.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .data import DataError, outcome_keys, validate_record, validate_target, validate_view


def _working(tensor: Tensor) -> Tensor:
    if not tensor.is_floating_point():
        raise TypeError("Expected a floating-point tensor")
    return tensor


def decision_loss(logits: Tensor, question: dict, target: dict, *, check_finite=True) -> Tensor:
    """Logits contain only real outcomes, ordered by outcome_keys(question)."""
    validate_target(target, question, "target")
    logits = _working(logits)
    keys = outcome_keys(question)
    if logits.ndim != 1 or len(logits) != len(keys):
        raise ValueError("Expected one logit per valid outcome; remove padding first")
    if check_finite and not bool(torch.isfinite(logits).all()):
        raise ValueError("Non-finite logits")
    if question["type"] == "noul":
        truth_logit = logits[0] - logits[1]  # [true, false]
        return F.binary_cross_entropy_with_logits(truth_logit, logits.new_tensor(target["p_true"]))
    lp = F.log_softmax(logits, dim=0)
    if target["kind"] == "class":
        return -lp[keys.index(target["value"])]
    if target["kind"] == "distribution":
        distribution = logits.new_tensor([target["values"][key] for key in keys])
        return -(distribution * lp).sum()
    indices = [keys.index(key) for key in target["values"]]
    return -torch.logsumexp(lp[indices], dim=0)


def _cosine_distances(a: Tensor, b: Tensor, *, check_finite=True) -> Tensor:
    if a.shape != b.shape or a.ndim not in (1, 2, 3) or a.numel() == 0:
        raise ValueError("Readouts must have identical nonempty [...,d] shape")
    if a.device != b.device:
        raise ValueError("Readouts must be on the same device")
    a, b = _working(a), _working(b)
    if check_finite and not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("Non-finite readouts")
    return 1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-8)


def jepa_loss(readouts_a: Tensor, readouts_b: Tensor) -> Tensor:
    """Low-level cosine mean. Both sides retain gradients.

    This primitive alone does not enforce a question's readout contract;
    training uses question_jepa_loss, below.
    """
    return _cosine_distances(readouts_a, readouts_b).mean()


def question_jepa_loss(question: dict, a: Tensor, b: Tensor, *, check_finite=True) -> Tensor:
    """Align every scoring readout with fixed candidate/global group weights.

    Every primitive uses [K+1,d], ordered h_0,...,h_(K-1),g.
    Noul has K=2 with true before false; Score follows the supplied level order.
    No padding or inferred positional pairing is allowed.
    """
    k = len(outcome_keys(question))
    prefix = (k + 1,)
    for x in (a, b):
        if x.ndim != len(prefix) + 1 or tuple(x.shape[:-1]) != prefix or x.shape[-1] == 0:
            raise ValueError(f"Expected {question['type']} scoring readouts shaped {prefix} + (d,)")
    distances = _cosine_distances(a, b, check_finite=check_finite)
    return 0.5 * distances[:-1].mean() + 0.5 * distances[-1]


def _selected_view(record: dict, selected_view_id: str | None) -> dict:
    views = record.get("views", [])
    if not views:
        raise DataError("Alternate predictions require a verified view in the record")
    if selected_view_id is None:
        if len(views) != 1:
            raise DataError("Multiple views require the collator's selected_view_id")
        return views[0]
    matches = [v for v in views if v["view_id"] == selected_view_id]
    if len(matches) != 1:
        raise DataError("selected_view_id must identify exactly one verified view")
    return matches[0]


def record_loss(
    record: dict,
    base: dict[str, dict[str, Tensor]],
    alternate: dict[str, dict[str, Tensor]] | None = None,
    jepa_lambda: float = 0.01,
    selected_view_id: str | None = None,
    *,
    check_finite: bool = True,
) -> dict[str, Any]:
    """Average logical questions and supervised views, never branch counts.

    Predictions: qid -> {'logits': [outcomes], 'readouts': question-specific
    full readout tensor}. Readouts are required only for selected views explicitly
    audited with equivalence.readout_scope='all_readouts'. Old question_only
    views remain paired SFT examples; they are NOT silently promoted to prefix
    equivalence. Counts report this skip; a trainer must track them.

    These checks cannot prove natural-language equivalence or that a collator
    actually used the selected view. It must pass the proper inputs and routing.
    """
    validate_record(record)
    if not math.isfinite(jepa_lambda) or jepa_lambda < 0:
        raise ValueError("jepa_lambda must be finite and nonnegative")
    selected = _selected_view(record, selected_view_id) if alternate is not None else None
    if alternate is None and selected_view_id is not None:
        raise DataError("selected_view_id supplied without alternate predictions")
    paired_ids = set()
    if selected is not None:
        validate_view(record, selected)
        paired_ids = set(selected["request"]["questions"]) & set(record["targets"])
        if set(alternate) != paired_ids:
            raise DataError(
                "Alternate predictions must exactly match selected supervised questions"
            )
    eligible = (
        selected is not None and selected["equivalence"].get("readout_scope") == "all_readouts"
    )
    losses, aux_losses, weights = [], [], []
    questions = record["request"]["questions"]
    for qid, target in record["targets"].items():
        q = questions[qid]
        p = base[qid]
        supervised = decision_loss(p["logits"], q, target, check_finite=check_finite)
        auxiliary = supervised.new_zeros(())
        if qid in paired_ids:
            alt = alternate[qid]
            supervised = 0.5 * (supervised + decision_loss(
                alt["logits"], q, target, check_finite=check_finite))
            if eligible:
                if "readouts" not in p or "readouts" not in alt:
                    raise ValueError(
                        "Full readouts required; g-only predictions are not the v2 contract"
                    )
                auxiliary = question_jepa_loss(q, p["readouts"], alt["readouts"],
                                              check_finite=check_finite)
        losses.append(supervised)
        aux_losses.append(auxiliary)
        weights.append(record.get("question_weights", {}).get(qid, 1.0))
    if not losses:
        raise ValueError("No supervised questions")
    w = losses[0].new_tensor(weights)
    if not all(math.isfinite(weight) and weight > 0 for weight in weights):
        raise ValueError("Invalid question weights")
    d = (torch.stack(losses) * w).sum() / w.sum()
    j = (torch.stack(aux_losses) * w).sum() / w.sum()
    return {
        "loss": d + jepa_lambda * j,
        "decision_loss": d,
        "jepa_loss": j,
        "paired_questions": len(paired_ids),
        "jepa_eligible_questions": len(paired_ids) if eligible else 0,
        "jepa_scope_skipped_questions": len(paired_ids)
        if alternate is not None and not eligible
        else 0,
    }


def batch_loss(records: list[dict], outputs: list[dict[str, Tensor]]) -> Tensor:
    if not records or len(records) != len(outputs):
        raise ValueError("Nonempty matching record/output lists required")
    values = torch.stack([o["loss"] for o in outputs])
    weights = values.new_tensor([r.get("weight", 1.0) for r in records])
    if not bool(torch.isfinite(weights).all() and (weights > 0).all()):
        raise ValueError("Invalid sample weights")
    return (values * weights).sum() / weights.sum()


def record_losses(entries, jepa_lambda):
    """Batch the reference objectives by outcome shape, without padding outcomes.

    Each entry is (record, base, alternate, selected_view_id). The caller checks
    prediction finiteness once for the physical batch before entering here.
    Sample weights remain the trainer's responsibility, as with record_loss.
    """
    if not entries or not math.isfinite(jepa_lambda) or jepa_lambda < 0:
        raise ValueError("Nonempty entries and a finite nonnegative JEPA coefficient required")
    groups, metadata = defaultdict(list), []
    readouts_a, readouts_b, readout_weights, readout_owners = [], [], [], []
    for owner, (record, base, alternate, view_id) in enumerate(entries):
        validate_record(record)
        selected = _selected_view(record, view_id) if alternate is not None else None
        if alternate is None and view_id is not None:
            raise DataError("selected_view_id supplied without alternate predictions")
        paired = set()
        if selected is not None:
            validate_view(record, selected)
            paired = set(selected["request"]["questions"]) & set(record["targets"])
            if set(alternate) != paired:
                raise DataError("Alternate predictions must exactly match selected supervised questions")
        eligible = selected is not None and selected["equivalence"].get("readout_scope") == "all_readouts"
        weights = {qid: record.get("question_weights", {}).get(qid, 1.0)
                   for qid in record["targets"]}
        total = sum(weights.values())
        if not weights or not all(math.isfinite(w) and w > 0 for w in weights.values()):
            raise ValueError("Invalid question weights")
        for qid, target in record["targets"].items():
            question = record["request"]["questions"][qid]
            keys = outcome_keys(question)
            weight = weights[qid] / total
            views = [base[qid]] + ([alternate[qid]] if qid in paired else [])
            for prediction in views:
                logits = _working(prediction["logits"])
                if logits.ndim != 1 or len(logits) != len(keys):
                    raise ValueError("Expected one logit per valid outcome; remove padding first")
                groups[(target["kind"], len(keys))].append(
                    (logits, target, keys, owner, weight / len(views)))
            if qid in paired and eligible:
                if any("readouts" not in view for view in views):
                    raise ValueError("Full readouts required; g-only predictions are not the v2 contract")
                a, b = (view["readouts"] for view in views)
                if a.shape != b.shape or a.ndim != 2 or a.shape[0] != len(keys) + 1 or not a.shape[1]:
                    raise ValueError("Expected matching full scoring readouts shaped [K+1,d]")
                readouts_a.append(a)
                readouts_b.append(b)
                readout_weights.extend([weight * .5 / len(keys)] * len(keys) + [weight * .5])
                readout_owners.extend([owner] * (len(keys) + 1))
        metadata.append({
            "paired_questions": len(paired),
            "jepa_eligible_questions": len(paired) if eligible else 0,
            "jepa_scope_skipped_questions": len(paired) if selected is not None and not eligible else 0,
        })
    parts = []
    for (kind, _), group in groups.items():
        logits = torch.stack([entry[0] for entry in group])
        targets, keys, owners, weights = zip(*(entry[1:] for entry in group))
        if kind == "bernoulli":
            values = F.binary_cross_entropy_with_logits(
                logits[:, 0] - logits[:, 1], logits.new_tensor([t["p_true"] for t in targets]),
                reduction="none")
        else:
            lp = F.log_softmax(logits, dim=-1)
            if kind == "class":
                indices = torch.tensor([k.index(t["value"]) for t, k in zip(targets, keys)],
                                       device=logits.device)
                values = -lp.gather(1, indices[:, None]).squeeze(1)
            elif kind == "distribution":
                distribution = logits.new_tensor([[t["values"][key] for key in k]
                                                  for t, k in zip(targets, keys)])
                values = -(distribution * lp).sum(-1)
            else:
                selected = torch.tensor([[key in t["values"] for key in k]
                                         for t, k in zip(targets, keys)], device=logits.device)
                values = -torch.logsumexp(lp.masked_fill(~selected, -torch.inf), dim=-1)
        parts.append(values.new_zeros(len(entries)).index_add(
            0, torch.tensor(owners, device=logits.device), values * logits.new_tensor(weights)))
    decision = torch.stack(parts).sum(0)
    auxiliary = torch.zeros_like(decision)
    if readouts_a:
        distances = _cosine_distances(torch.cat(readouts_a), torch.cat(readouts_b), check_finite=False)
        auxiliary = distances.new_zeros(len(entries)).index_add(
            0, torch.tensor(readout_owners, device=decision.device),
            distances * distances.new_tensor(readout_weights))
    losses = decision + jepa_lambda * auxiliary
    return [dict(info, loss=loss, decision_loss=d, jepa_loss=j)
            for info, loss, d, j in zip(metadata, losses.unbind(), decision.unbind(), auxiliary.unbind())]
