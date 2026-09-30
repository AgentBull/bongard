"""Probability metrics, representation diagnostics and held-out temperature fitting."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import DataError, check_split_groups, eligible_views, outcome_keys, validate_record
from .inference import temperature_for, validate_temperatures
from .losses import decision_loss
from .model import checkpoint_config_path
from .provenance import check_independent_data, checkpoint_sources


def target_distribution(question, target):
    keys = outcome_keys(question)
    if target["kind"] == "allowed_set":
        return None
    if target["kind"] == "bernoulli":
        return torch.tensor([target["p_true"], 1 - target["p_true"]], dtype=torch.float64)
    if target["kind"] == "distribution":
        return torch.tensor([target["values"][key] for key in keys], dtype=torch.float64)
    return torch.tensor([float(key == target["value"]) for key in keys], dtype=torch.float64)


def metrics_for(logits, question, target, *, temperature=1.0):
    temperature = validate_temperatures({question["type"]: temperature})[question["type"]]
    # Convert before division: tiny temperatures overflow FP32, and MPS lacks float64.
    raw = logits.detach().cpu().double()
    if not torch.isfinite(raw).all():
        # A non-finite model logit (seen once from the fp8-wrapped training model at an
        # in-trainer dev evaluation, 2026-09-28) must not kill a 16-hour run: count it, treat
        # the outcome as impossible, and let the stage checks on the saved bf16 bundle judge.
        metrics_for.non_finite = getattr(metrics_for, "non_finite", 0) + 1
        raw = torch.nan_to_num(raw, nan=-1e6, posinf=1e6, neginf=-1e6)
    z = raw / temperature
    if not torch.isfinite(z).all():  # a misconfigured temperature, not a model fault
        raise DataError("Temperature-scaled evaluation logits exceed finite float64 range")
    p = z.softmax(-1)
    keys = outcome_keys(question)
    chosen = int(p.argmax())
    t = target_distribution(question, target)
    result = {"nll": float(decision_loss(z, question, target))}
    if t is None:
        result["allowed_accuracy"] = float(keys[chosen] in target["values"])
    else:
        # Expected correctness supports soft targets without pretending they are observations.
        result["accuracy"] = float(t[chosen])
        result["brier"] = float(p.square().sum() - 2 * (p * t).sum() + 1)
        result["distribution_brier"] = float(((p - t) ** 2).sum())
        result["top_probability"] = float(p[chosen])
        if question["type"] == "score":
            fp, ft = p.cumsum(0)[:-1], t.cumsum(0)[:-1]
            result["rps"] = float((fp.square() - 2 * fp * ft + ft).mean())
            result["distribution_rps"] = float(((fp - ft) ** 2).mean())
    if any(not math.isfinite(value) for value in result.values()):
        raise DataError("Temperature-scaled evaluation metrics exceed finite float64 range")
    return result


def summarize(rows):
    totals, weights = defaultdict(float), defaultdict(float)
    bins = [[0.0, 0.0, 0.0] for _ in range(10)]
    risk_groups = defaultdict(list)
    for row in rows:
        weight = row["weight"]
        for name, value in row["metrics"].items():
            totals[name] += value * weight
            weights[name] += weight
        if "top_probability" in row["metrics"]:
            p, correctness = row["metrics"]["top_probability"], row["metrics"]["accuracy"]
            b = bins[min(9, int(p * 10))]
            b[0] += weight
            b[1] += p * weight
            b[2] += correctness * weight
            risk_groups[p].append((weight, weight * (1 - correctness)))
    result = {name: totals[name] / weights[name] for name in totals}
    mass = sum(b[0] for b in bins)
    if mass:
        result["ece"] = sum(abs(b[1] - b[2]) for b in bins) / mass
        result["reliability_bins"] = [
            dict(weight=b[0], probability=b[1] / b[0], correctness=b[2] / b[0])
            for b in bins
            if b[0]
        ]
        # A threshold accepts every question with the same predicted score together.
        # Correctness contributes only to the resulting risk, never to selection.
        groups = [
            (score, math.fsum(w for w, _ in items), math.fsum(e for _, e in items))
            for score, items in sorted(risk_groups.items(), key=lambda item: item[0], reverse=True)
        ]
        total_weight = math.fsum(weight for _, weight, _ in groups)
        selected = {round((len(groups) - 1) * i / 10) for i in range(11)}
        curve, accepted_weights, accepted_errors = [], [], []
        for index, (threshold, weight, errors) in enumerate(groups):
            accepted_weights.append(weight)
            accepted_errors.append(errors)
            if index in selected:
                used = math.fsum(accepted_weights)
                curve.append(
                    {
                        "threshold": threshold,
                        "coverage": used / total_weight,
                        "risk": math.fsum(accepted_errors) / used,
                    }
                )
        result["risk_coverage_score"] = "top_probability"
        result["risk_coverage"] = curve
    result.update(questions=len(rows), effective_weight=sum(r["weight"] for r in rows))
    return result


def representation_stats(rows):
    if not rows:
        return {}
    x = torch.stack(rows).cpu().double()
    direction = F.normalize(x, dim=-1)
    singular = torch.linalg.svdvals(direction - direction.mean(0))
    energy = singular.square()
    if float(energy.sum()) > 0:
        p = energy / energy.sum()
        rank = float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
    else:
        rank = 0.0
    return {
        "count": len(rows),
        "mean_norm": float(x.norm(dim=-1).mean()),
        "direction_variance": float(direction.var(0, unbiased=False).mean()),
        "effective_rank": rank,
    }


@torch.inference_mode()
def evaluate(model, records, *, temperatures=None, diagnostics=True):
    temperatures = validate_temperatures(temperatures or {})
    was_training = model.training
    model.eval()
    non_finite_before = getattr(metrics_for, "non_finite", 0)
    rows, groups, h_samples, g_samples = [], defaultdict(list), [], []
    try:
        for record in records:
            saved = record.get("_tokenized")
            settings = record.get("_token_settings")
            record = {k: v for k, v in record.items() if k not in {"_tokenized", "_token_settings"}}
            validate_record(record)
            if saved is None:
                compiled = model.compiler.compile(record["request"])
            else:
                from .tokenized_data import restore_request, verify_compiler

                verify_compiler(settings, model.compiler)
                compiled = restore_request(saved["base"], record["request"], model.compiler)
            outputs = model(compiled)
            qweights = record.get("question_weights", {})
            denominator = sum(qweights.get(q, 1.0) for q in record["targets"])
            for qid, target in record["targets"].items():
                question = record["request"]["questions"][qid]
                result = outputs[qid]
                weight = record.get("weight", 1.0) * qweights.get(qid, 1.0) / denominator
                temperature = temperature_for(temperatures, question["type"],
                                              len(outcome_keys(question)))
                row = {
                    "id": record["id"],
                    "question": qid,
                    "weight": weight,
                    "metrics": metrics_for(
                        result["logits"], question, target, temperature=temperature
                    ),
                }
                rows.append(row)
                cq = compiled.questions[qid]
                s, i = len(compiled.state), cq.instruction_tokens
                longest = len(cq.tokens)
                k = len(outcome_keys(question))
                labels = [
                    f"primitive:{question['type']}",
                    f"source:{record['task_id']}",
                    f"S:{_bucket(s)}",
                    f"I:{_bucket(i)}",
                    f"T:{_bucket(longest)}",
                    f"K:{k}",
                ]
                for tag in record.get("metadata", {}).get("slices", []):
                    labels.append(f"slice:{tag}")
                for label in labels:
                    groups[label].append(row)
                if diagnostics:
                    r = result["readouts"]
                    h, g = r[:-1], r[-1:]
                    h_samples.extend(v.detach().cpu() for v in h[: max(0, 256 - len(h_samples))])
                    g_samples.extend(v.detach().cpu() for v in g[: max(0, 256 - len(g_samples))])
                    zero_logits = model.head(h, torch.zeros_like(g))
                    row["metrics"]["zero_g_nll"] = metrics_for(
                        zero_logits, question, target, temperature=temperature
                    )["nll"]
        overall = summarize(rows)
        # Non-finite model logits are sanitised in metrics_for; report how many this pass saw
        # so a training log shows the fault instead of hiding it.
        overall["non_finite_logits"] = getattr(metrics_for, "non_finite", 0) - non_finite_before
        return {
            "overall": overall,
            "slices": {k: summarize(v) for k, v in groups.items()},
            "representations": {
                "h": representation_stats(h_samples),
                "g": representation_stats(g_samples),
            },
            "records": rows,
        }
    finally:
        model.train(was_training)


def _bucket(n):
    return next(
        (str(limit) for limit in (128, 256, 512, 1024, 2048, 4096, 32768) if n <= limit), ">32768"
    )


@torch.inference_mode()
def evaluate_pairs(model, records):
    """Optional offline consistency diagnostic; serving still performs one view only."""
    was_training = model.training
    model.eval()
    rows, rejected = [], []
    try:
        for record in records:
            base = model(model.compiler.compile(record["request"]))
            views, bad = eligible_views(record)
            rejected.extend({"id": record["id"], **reason} for reason in bad)
            for view in views:
                alternate = model(model.compiler.compile(view["request"]))
                for qid in (q for q in view["request"]["questions"] if q in record["targets"]):
                    p, q = (base[qid]["logits"].softmax(-1), alternate[qid]["logits"].softmax(-1))
                    question, target = record["request"]["questions"][qid], record["targets"][qid]
                    rows.append(
                        {
                            "id": record["id"],
                            "view_id": view["view_id"],
                            "question": qid,
                            "method": view["equivalence"]["method"],
                            "total_variation": float((p - q).abs().sum() / 2),
                            "argmax_agreement": int(p.argmax() == q.argmax()),
                            "base_nll": float(decision_loss(base[qid]["logits"], question, target)),
                            "view_nll": float(
                                decision_loss(alternate[qid]["logits"], question, target)
                            ),
                        }
                    )
        return {"pairs": rows, "rejected_views": rejected}
    finally:
        model.train(was_training)


def model_fingerprint(checkpoint):
    """Bind calibration to the actual exported weights, not a mutable directory name."""
    digest = hashlib.sha256()
    root = Path(checkpoint)
    config = checkpoint_config_path(root)
    metadata = json.loads(config.read_text())
    base = root / metadata["lora"]["base"] if "lora" in metadata else root / "backbone"
    files = [
        # Keep existing calibration fingerprints valid after the config-file rename.
        ("bundle.json", config),
        ("backbone/config.json", base / "config.json"),
        ("head.safetensors", root / "head.safetensors"),
        *[(f"backbone/{p.name}", p) for p in sorted(base.glob("*.safetensors"))],
    ]
    if "lora" in metadata:
        files.extend(
            (f"adapter/{p.name}", p)
            for p in sorted((root / "adapter").iterdir())
            if p.suffix in {".json", ".safetensors"}
        )
    for name, file in files:
        digest.update(name.encode())
        with file.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def calibration_logits(model, records):
    groups = defaultdict(list)
    model.eval()
    for record in records:
        outputs = model(model.compiler.compile(record["request"]))
        weights = record.get("question_weights", {})
        denominator = sum(weights.get(q, 1.0) for q in record["targets"])
        for qid, target in record["targets"].items():
            q = record["request"]["questions"][qid]
            if target["kind"] == "allowed_set" or len(outcome_keys(q)) == 1:
                continue
            weight = record.get("weight", 1.0) * weights.get(qid, 1.0) / denominator
            groups[q["type"]].append((outputs[qid]["logits"].cpu().double(), q, target, weight))
    return groups


def calibrate(model, records, *, excluded_splits, checkpoint, output):
    # A trained checkpoint owns its full course history. Low-level model bundles
    # without training metadata require the caller's explicit training split.
    sources = checkpoint_sources(checkpoint, required=not bool(excluded_splits.get("train")))
    check_independent_data(records, sources, "calibration")
    check_split_groups({**excluded_splits, "calibration": records})
    groups = calibration_logits(model, records)
    temperatures, metrics = {}, {}
    for kind, items in groups.items():
        log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.2, max_iter=60, line_search_fn="strong_wolfe")
        mass = sum(item[3] for item in items)

        def objective(t):
            return (
                sum(decision_loss(z / t, q, target) * weight for z, q, target, weight in items)
                / mass
            )

        initial = float(objective(torch.tensor(1.0, dtype=torch.float64)))

        def closure():
            optimizer.zero_grad()
            value = objective(log_t.exp())
            value.backward()
            return value

        optimizer.step(closure)
        temperature = float(log_t.detach().exp())
        validate_temperatures({kind: temperature})
        final = float(objective(log_t.detach().exp()))
        if final > initial:
            temperature, final = 1.0, initial
        temperatures[kind] = temperature
        metrics[kind] = {"questions": len(items), "nll_before": initial, "nll_after": final}
    if not temperatures:
        raise DataError("No complete probability targets available for calibration")
    report = {
        "temperatures": temperatures,
        "metrics": metrics,
        "model_sha256": model_fingerprint(checkpoint),
        "calibration_groups": sorted({r["metadata"]["split_group"] for r in records}),
    }
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report
