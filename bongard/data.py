"""Strict dataset adapters for a conditional-judgment model.

This module converts data and validates its semantics. It is not a model trainer,
not a Jev server, and cannot prove natural-language equivalence between views.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

VERSION = "judgment-sft-v1"
ENTRY_TYPES = (str, dict, list, type(None))


class DataError(ValueError):
    pass


def fail(message: str) -> None:
    raise DataError(message)


def exact_keys(obj: dict, allowed: set[str], where: str) -> None:
    extra = set(obj) - allowed
    if extra:
        fail(f"{where}: unknown fields {sorted(extra)}")


def finite_json(value: Any, where: str = "record") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            fail(f"{where}: NaN/Infinity is not valid data")
        return
    if isinstance(value, list):
        for i, child in enumerate(value):
            finite_json(child, f"{where}[{i}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                fail(f"{where}: JSON object keys must be strings")
            finite_json(child, f"{where}.{key}")
        return
    fail(f"{where}: unsupported value type {type(value).__name__}")


def number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{where}: expected a number, not a boolean/string")
    value = float(value)
    if not math.isfinite(value):
        fail(f"{where}: non-finite number")
    return value


def probability(value: Any, where: str) -> float:
    p = number(value, where)
    if not 0 <= p <= 1:
        fail(f"{where}: expected probability in [0, 1]")
    return p


def outcome_keys(question: dict) -> list[str]:
    kind = question.get("type")
    if kind == "choice":
        return list(question["criteria"])
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    if kind == "noul":
        return ["true", "false"]
    fail(f"Unknown primitive: {kind!r}")


def validate_question(question: Any, where: str) -> None:
    if not isinstance(question, dict):
        fail(f"{where}: expected an object")
    exact_keys(question, {"type", "instructions", "criteria"}, where)
    if "instructions" in question and not isinstance(question["instructions"], ENTRY_TYPES):
        fail(f"{where}.instructions: expected string/object/array/null")
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
            fail(f"{where}: Choice requires 1..255 named options in training IR")
        for key, description in criteria.items():
            if not isinstance(key, str) or not key or not isinstance(description, ENTRY_TYPES):
                fail(f"{where}: invalid option key or description")
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            fail(f"{where}: Score requires 2..10 descriptions")
        if any(not isinstance(v, (str, dict, list)) for v in criteria):
            fail(
                f"{where}: Score levels must be string/object/array descriptions, not null/numbers"
            )
    elif kind == "noul":
        if criteria is not None:
            if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
                fail(f"{where}: Noul criteria keys must be true/false")
            if any(not isinstance(v, ENTRY_TYPES) for v in criteria.values()):
                fail(f"{where}: invalid Noul description")
    else:
        fail(f"{where}: unknown primitive {kind!r}")


def validate_request(request: Any, where: str = "request") -> None:
    finite_json(request, where)
    if not isinstance(request, dict):
        fail(f"{where}: expected an object")
    exact_keys(request, {"model", "state", "questions", "images"}, where)
    if "state" not in request or not isinstance(request["state"], (str, dict, list)):
        fail(f"{where}.state: root must be string/object/array")
    if "model" in request and not isinstance(request["model"], str):
        fail(f"{where}.model: expected a string")
    if "images" in request:
        images = request["images"]
        if not isinstance(images, list) or any(
            not isinstance(image, str)
            or re.fullmatch(r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}", image)
            is None
            for image in images
        ):
            fail(f"{where}.images: expected a list of PNG/JPEG/WebP base64 data URLs")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        fail(f"{where}.questions: expected a nonempty object")
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid:
            fail(f"{where}: question ids must be nonempty strings")
        validate_question(q, f"{where}.questions.{qid}")


def validate_target(target: Any, question: dict, where: str) -> None:
    if not isinstance(target, dict):
        fail(f"{where}: expected an object")
    kind = target.get("kind")
    if question["type"] == "noul":
        exact_keys(target, {"kind", "p_true"}, where)
        if kind != "bernoulli" or "p_true" not in target:
            fail(f"{where}: Noul requires a bernoulli target")
        probability(target["p_true"], f"{where}.p_true")
        return
    allowed = set(outcome_keys(question))
    if kind == "class":
        exact_keys(target, {"kind", "value"}, where)
        if not isinstance(target.get("value"), str) or target["value"] not in allowed:
            fail(f"{where}: class value must be an existing string key")
    elif kind == "distribution":
        exact_keys(target, {"kind", "values"}, where)
        values = target.get("values")
        if not isinstance(values, dict) or set(values) != allowed:
            fail(f"{where}: distribution keys must exactly match all outcomes")
        ps = [probability(v, where) for v in values.values()]
        if not math.isclose(sum(ps), 1.0, rel_tol=0, abs_tol=1e-6):
            fail(f"{where}: distribution must sum to 1; no silent renormalization")
    elif kind == "allowed_set":
        exact_keys(target, {"kind", "values"}, where)
        values = target.get("values")
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(x, str) for x in values)
        ):
            fail(f"{where}: allowed_set must be a nonempty list of keys")
        if len(values) != len(set(values)) or not set(values) <= allowed:
            fail(f"{where}: duplicate or nonexistent allowed outcome")
    else:
        fail(f"{where}: unsupported target kind {kind!r}")


def validate_record(record: dict, *, validate_views: bool = False) -> dict:
    if not isinstance(record, dict):
        fail("record must be an object")
    finite_json({k: v for k, v in record.items() if k != "views"})
    exact_keys(
        record,
        {
            "schema_version",
            "id",
            "task_id",
            "request",
            "targets",
            "views",
            "metadata",
            "weight",
            "question_weights",
        },
        "record",
    )
    if record.get("schema_version") != VERSION:
        fail("record: unknown schema_version")
    for key in ("id", "task_id"):
        if not isinstance(record.get(key), str) or not record[key]:
            fail(f"record.{key}: expected a nonempty string")
    if "metadata" in record and not isinstance(record["metadata"], dict):
        fail("record.metadata: expected an object")
    if number(record.get("weight", 1.0), "weight") <= 0:
        fail("weight must be positive")
    validate_request(record.get("request"))
    qs = record["request"]["questions"]
    targets = record.get("targets")
    if not isinstance(targets, dict) or not targets or not set(targets) <= set(qs):
        fail("targets: nonempty subset of question ids required")
    for qid, target in targets.items():
        validate_target(target, qs[qid], f"targets.{qid}")
    weights = record.get("question_weights", {})
    if not isinstance(weights, dict) or not set(weights) <= set(targets):
        fail("question_weights: keys must be supervised question ids")
    for qid, value in weights.items():
        if number(value, f"question_weights.{qid}") <= 0:
            fail("question weights must be positive")
    if validate_views:
        views = record.get("views", [])
        if not isinstance(views, list):
            fail("views must be a list")
        ids = set()
        for view in views:
            validate_view(record, view)
            if view["view_id"] in ids:
                fail("duplicate view_id")
            ids.add(view["view_id"])
    return record


def validate_view(record: dict, view: Any) -> dict:
    """Check only the declared question subset, never promote whole-record scope."""
    finite_json(view, "view")
    if not isinstance(view, dict):
        fail("view must be an object")
    exact_keys(view, {"view_id", "request", "equivalence"}, "view")
    if not isinstance(view.get("view_id"), str) or not view["view_id"]:
        fail("view_id must be nonempty")
    eq = view.get("equivalence")
    if not isinstance(eq, dict) or eq.get("verified") is not True:
        fail("view requires verified equivalence")
    if eq.get("relation") != "same_conditional_task":
        fail("view requires same_conditional_task, not same_label")
    if eq.get("readout_scope", "question_only") not in {"question_only", "all_readouts"}:
        fail("invalid readout_scope")
    for key in ("method", "reference"):
        if not isinstance(eq.get(key), str) or not eq[key]:
            fail(f"equivalence.{key} is required")
    validate_request(view.get("request"), "view.request")
    base = record["request"]["questions"]
    other = view["request"]["questions"]
    if not set(other) <= set(base):
        fail("view questions must be a subset of the base request")
    for qid, q in other.items():
        if q["type"] != base[qid]["type"] or outcome_keys(q) != outcome_keys(base[qid]):
            fail("view must retain primitive and ordered outcome keys")
        if qid in record["targets"]:
            validate_target(record["targets"][qid], q, f"view.targets.{qid}")
    if not set(other) & set(record["targets"]):
        fail("view has no supervised questions")
    return view


def eligible_views(record: dict) -> tuple[list[dict], list[dict]]:
    """Invalid relations never discard reliable primary supervision."""
    views = record.get("views", [])
    if not isinstance(views, list):
        return [], [{"reason": "views must be a list"}]
    counts = Counter(
        v.get("view_id") for v in views if isinstance(v, dict) and isinstance(v.get("view_id"), str)
    )
    accepted, rejected = [], []
    for index, view in enumerate(views):
        try:
            validate_view(record, view)
            if counts[view["view_id"]] > 1:
                fail("duplicate view_id")
            accepted.append(view)
        except DataError as exc:
            rejected.append({"index": index, "reason": str(exc)})
    return accepted, rejected


def _field(row: dict, fields: dict, name: str, default_name: str | None = None) -> Any:
    key = fields.get(name, default_name or name)
    if key not in row:
        fail(f"missing source field {key!r} (role: {name})")
    return copy.deepcopy(row[key])


def _source_label(value: Any, config: dict) -> Any:
    kind = config.get("label_type", "string")
    if kind == "string":
        if not isinstance(value, str):
            fail("label_type=string requires string labels; declare numeric mapping explicitly")
    elif kind == "integer":
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            fail("label_type=integer requires an integer")
    elif kind == "boolean":
        if value in ("true", "false"):
            value = value == "true"
        if not isinstance(value, bool):
            fail("label_type=boolean requires true/false")
    else:
        fail(f"unknown label_type {kind}")
    return value


def map_label(value: Any, config: dict, keys: list[str]) -> str:
    value = _source_label(value, config)
    for mapping in config.get("label_map", []):
        source = mapping["source"]
        if type(source) is type(value) and source == value:
            key = mapping["target"]
            if key not in keys:
                fail("label_map points outside the outcome space")
            return key
    if isinstance(value, str) and value in keys:
        return value
    fail(f"unmapped label {value!r}; do not guess label semantics")


def make_target(row: dict, config: dict, q: dict) -> dict:
    fields = config.get("fields", {})
    kind = config.get("target_kind", "class")
    if kind == "bernoulli":
        value = _field(row, fields, "p_true")
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                fail("p_true column must contain a number")
        return {"kind": "bernoulli", "p_true": probability(value, "p_true")}
    if q["type"] == "noul":
        value = _source_label(_field(row, fields, "label"), config)
        for mapping in config.get("label_map", []):
            if type(mapping["source"]) is type(value) and mapping["source"] == value:
                return {
                    "kind": "bernoulli",
                    "p_true": probability(mapping["target"], "mapped p_true"),
                }
        if type(value) is bool:
            return {"kind": "bernoulli", "p_true": float(value)}
        fail("binary labels require an explicit label_map or boolean label_type")
    keys = outcome_keys(q)
    if kind == "class":
        key = map_label(_field(row, fields, "label"), config, keys)
        return {"kind": "class", "value": key}
    if kind == "allowed_set":
        values = _field(row, fields, "allowed")
        if not isinstance(values, list):
            fail("allowed_set source must be a list")
        return {"kind": "allowed_set", "values": [map_label(x, config, keys) for x in values]}
    if kind == "distribution":
        values = _field(row, fields, "probabilities")
        if isinstance(values, list):
            order = config.get("probability_order")
            if (
                not isinstance(order, list)
                or len(order) != len(set(order))
                or set(order) != set(keys)
                or len(order) != len(values)
            ):
                fail("probability arrays require explicit probability_order matching every outcome")
            values = dict(zip(order, values))
        return {"kind": "distribution", "values": values}
    fail(f"unknown target_kind {kind!r}")


def normalize(row: dict, config: dict, task_id: str) -> dict:
    """Normalize one source row. Semantic equivalence cannot be inferred here."""
    finite_json(row, "source row")
    adapter, fields = config["adapter"], config.get("fields", {})
    if adapter == "native":
        if row.get("schema_version"):
            return validate_record(copy.deepcopy(row))
        request, targets = copy.deepcopy(row["request"]), copy.deepcopy(row["targets"])
    elif adapter in {"classification", "pair", "instruction", "binary", "ordinal", "chat_label"}:
        q = copy.deepcopy(config["question"])
        if adapter == "pair":
            state = {
                name: _field(row, {"value": source}, "value")
                for name, source in config["state_fields"].items()
            }
        elif adapter == "instruction":
            state = _field(row, fields, "input")
            q["instructions"] = _field(row, fields, "instruction")
        elif adapter == "chat_label":
            messages = _field(row, fields, "messages")
            roles = [x.get("role") for x in messages] if isinstance(messages, list) else []
            if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
                fail(
                    "chat_label supports only [system?, user, assistant-label]; no implicit multi-turn conversion"
                )
            if any(not isinstance(m.get("content"), str) for m in messages):
                fail("chat_label content must be text")
            if len(messages) == 3:
                q["instructions"] = messages[0]["content"]
            state = messages[-2]["content"]
            row = copy.deepcopy(row)
            # Only an exact label is accepted; rationales are never substring-parsed.
            row[fields.get("label", "label")] = messages[-1]["content"]
        else:
            state = _field(row, fields, "text")
        request = {"state": state, "questions": {"q": q}}
        targets = {"q": make_target(row, config, q)}
    elif adapter == "dynamic_choice":
        criteria = _field(row, fields, "choices")
        local = copy.deepcopy(config)
        if isinstance(criteria, list):
            criteria = {str(i): text for i, text in enumerate(criteria)}
            if local.get("answer_encoding") != "index":
                fail("list choices require answer_encoding=index")
            local["label_type"] = "integer"
            local["label_map"] = [{"source": i, "target": str(i)} for i in range(len(criteria))]
        elif not isinstance(criteria, dict):
            fail("choices must be an object or list")
        q = {
            "type": "choice",
            "instructions": _field(row, fields, "instructions"),
            "criteria": criteria,
        }
        request = {"state": _field(row, fields, "state"), "questions": {"q": q}}
        targets = {"q": make_target(row, local, q)}
    elif adapter in {"multilabel_complete", "multilabel_partial"}:
        label_criteria = config["label_criteria"]
        qs = {
            key: {
                "type": "noul",
                "instructions": {"task": config["instructions"], "label": key, "definition": desc},
            }
            for key, desc in label_criteria.items()
        }
        request = {"state": _field(row, fields, "text"), "questions": qs}
        if adapter == "multilabel_complete":
            labels = _field(row, fields, "labels")
            if not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
                fail("complete multilabel requires a list of string keys")
            if len(labels) != len(set(labels)) or not set(labels) <= set(qs):
                fail("unknown or duplicated multilabel keys")
            targets = {key: {"kind": "bernoulli", "p_true": float(key in labels)} for key in qs}
        else:
            statuses = _field(row, fields, "label_states")
            if not isinstance(statuses, dict) or not set(statuses) <= set(qs):
                fail("partial multilabel requires an object of known labels")
            targets = {}
            for key, value in statuses.items():
                if value is None:
                    continue  # Unknown: no target, never 0 or 0.5.
                if type(value) is not bool:
                    fail("partial label states must be boolean or null")
                targets[key] = {"kind": "bernoulli", "p_true": float(value)}
    else:
        fail(f"unsupported adapter {adapter!r}")
    raw_digest = hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:20]
    state_digest = hashlib.sha256(
        json.dumps(request["state"], ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:20]
    metadata = copy.deepcopy(row.get("metadata", {}))
    metadata.setdefault("source", task_id)
    if "split_group" not in metadata:
        group_field = fields.get("split_group", "split_group")
        if group_field in row:
            metadata["split_group"] = row[group_field]
        else:
            metadata["split_group"] = f"{task_id}:{state_digest}"
            metadata["split_group_inferred"] = True
    result = {
        "schema_version": VERSION,
        "id": str(row.get("id", f"{task_id}:{raw_digest}")),
        "task_id": task_id,
        "request": request,
        "targets": targets,
        "metadata": metadata,
    }
    for name in ("weight", "question_weights", "views"):
        if name in row:
            result[name] = copy.deepcopy(row[name])
    if "view_state_field" in config and config["view_state_field"] in row:
        if "views" in result:
            fail("use either full views or view_state_field, not both")
        alternative = copy.deepcopy(request)
        alternative["state"] = copy.deepcopy(row[config["view_state_field"]])
        result["views"] = [
            {
                "view_id": "alternative",
                "request": alternative,
                "equivalence": copy.deepcopy(row.get("view_verification", {})),
            }
        ]
    return validate_record(result)


def model_visible_inputs(record: dict, view_index: int | None = None) -> dict:
    """Reference boundary only, not tokenization or model forward.

    Training metadata, routing ids, and target values are not model text.
    Each element of questions must still run as an isolated logical question.
    """
    request = record["request"] if view_index is None else record["views"][view_index]["request"]
    return {
        "state": copy.deepcopy(request["state"]),
        "questions": [copy.deepcopy(q) for q in request["questions"].values()],
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for k, v in pairs:
        if k in result:
            fail(f"duplicate JSON key {k!r}")
        result[k] = v
    return result


def strict_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=lambda s: fail(f"invalid JSON constant {s}"),
    )


def iter_rows(path: Path, *, format: str | None = None) -> Iterator[dict]:
    ext = f".{format.lower()}" if format else path.suffix.lower()
    if ext == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if line.strip():
                    try:
                        row = strict_loads(line)
                        if not isinstance(row, dict):
                            fail("source row must be an object")
                        yield row
                    except (ValueError, KeyError, TypeError) as e:
                        raise DataError(f"{path}:{line_number}: {e}") from e
    elif ext == ".json":
        rows = strict_loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            fail(".json input must be an array of row objects")
        yield from rows
    elif ext in {".csv", ".tsv"}:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ",")
            if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                fail("CSV/TSV requires unique header names")
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    fail("malformed CSV/TSV row")
                yield dict(row)
    elif ext == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise DataError("Parquet requires pyarrow; install the project dependencies") from e
        source = pq.ParquetFile(path)
        from .parquet_data import FORMAT, iter_records

        if (source.schema_arrow.metadata or {}).get(b"bongard_format") == FORMAT:
            yield from iter_records(path)
            return
        for batch in source.iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:
        fail("supported containers: jsonl/json/csv/tsv/parquet")


def metadata_records(records):
    """Avoid reading images/token arrays when only identity and grouping are needed."""
    return records.metadata_records() if hasattr(records, "metadata_records") else iter(records)


def check_split_groups(splits: dict[str, Iterable[dict]]) -> None:
    seen_groups: dict[str, str] = {}
    seen_ids: dict[str, str] = {}
    for split, records in splits.items():
        for r in metadata_records(records):
            if r["id"] in seen_ids:
                fail(f"duplicate record id: {r['id']}")
            group = r.get("metadata", {}).get("split_group")
            if not isinstance(group, str) or not group:
                fail("audited split_group required before training")
            for key, store in ((group, seen_groups), (r["id"], seen_ids)):
                if key in store and store[key] != split:
                    fail(f"cross-split leakage: {key!r} occurs in {store[key]} and {split}")
                store[key] = split
