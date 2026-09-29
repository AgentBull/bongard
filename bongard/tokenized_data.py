"""Offline token IDs stored alongside the original rows in the two training Parquets."""

from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .compiler import Compiler, Question, Request, entry
from .data import DataError, eligible_views, outcome_keys
from .parquet_data import FORMAT, unpack
from .predictive_data import ADAPTER_VERSION, training_record
from .token_safety import TokenSafetyError
from .vision import image_pixels

METADATA_KEY = b"bongard_tokens"
TOKEN_COLUMNS = ("_tokenized", "_token_costs", "_split_group", "_group_inferred")


def compiler_settings(compiler):
    return json.loads(
        entry(
            {
                "version": 1,
                "compiler": compiler.manifest,
                "adapter": ADAPTER_VERSION,
                "image_processor": compiler.image_processor.to_dict(),
                "image_tokens": compiler.vision_config.mm_tokens_per_image,
            }
        )
    )


def verify_compiler(settings, compiler):
    if settings != compiler_settings(compiler):
        raise DataError(
            "Stored token IDs use different tokenizer/processor settings; run tokenize again"
        )


def _request_dict(request):
    return {
        "state": request.state,
        "questions": {
            name: {
                "keys": question.keys,
                "tokens": question.tokens,
                "candidate_positions": question.candidate_positions,
                "decision_position": question.decision_position,
                "instruction_tokens": question.instruction_tokens,
                "candidate_starts": question.candidate_starts,
            }
            for name, question in request.questions.items()
        },
        "input_tokens": request.input_tokens,
    }


def compile_record(record, compiler):
    def supervised(request, *, base=False):
        order = record["targets"] if base else request["questions"]
        return compiler.compile(
            {
                **request,
                "questions": {
                    q: request["questions"][q] for q in order if q in record["targets"]
                },
            }
        )

    base = supervised(record["request"], base=True)
    candidates, rejected = eligible_views(record)
    views, costs = [], [base.input_tokens]
    for view in candidates:
        try:
            alternate = supervised(view["request"])
        except (DataError, TokenSafetyError) as exc:
            views.append({"view_id": view["view_id"], "request": None, "reason": str(exc)})
            costs.append(base.input_tokens)
        else:
            cost = base.input_tokens + alternate.input_tokens
            if alternate.state_key == base.state_key:
                cost -= len(alternate.state)
            views.append({"view_id": view["view_id"], "request": _request_dict(alternate)})
            costs.append(cost)
    return {
        "_tokenized": entry({"base": _request_dict(base), "views": views, "rejected": rejected}),
        "_token_costs": costs,
        "_split_group": record["metadata"]["split_group"],
        "_group_inferred": bool(record["metadata"].get("split_group_inferred")),
    }


def restore_request(saved, request, compiler, *, pixel_cache=None):
    """Candidate chunks can be reordered without changing or retokenizing their content."""
    questions = {}
    for qid in request["questions"]:
        value = saved["questions"][qid]
        question = Question(
            tuple(value["keys"]),
            tuple(value["tokens"]),
            tuple(value["candidate_positions"]),
            value["decision_position"],
            value["instruction_tokens"],
            tuple(value["candidate_starts"]),
        )
        keys = tuple(outcome_keys(request["questions"][qid]))
        if keys != question.keys:
            if set(keys) != set(question.keys):
                raise DataError("Stored candidates differ from the request")
            starts, positions = [], []
            tokens = list(question.tokens[: question.candidate_starts[0]])
            ends = (*question.candidate_starts[1:], question.decision_position)
            for key in keys:
                i = question.keys.index(key)
                start = question.candidate_starts[i]
                starts.append(len(tokens))
                positions.append(len(tokens) + question.candidate_positions[i] - start)
                tokens.extend(question.tokens[start : ends[i]])
            tokens.extend(question.tokens[question.decision_position :])
            question = Question(
                keys,
                tuple(tokens),
                tuple(positions),
                question.decision_position,
                question.instruction_tokens,
                tuple(starts),
            )
        questions[qid] = question
    state = tuple(saved["state"])
    if saved["input_tokens"] > compiler.max_request_tokens or any(
        len(state) + len(q.tokens) > compiler.max_sequence_tokens for q in questions.values()
    ):
        raise DataError("Stored token IDs exceed configured token budget; input was not truncated")
    pixels, images = None, ()
    if request.get("images"):
        cache = {} if pixel_cache is None else pixel_cache
        key = tuple(request["images"])
        if key not in cache:
            cache[key] = image_pixels(key, compiler.image_processor)
        pixels, images = cache[key]
    return Request(state, questions, saved["input_tokens"], images, pixels)


def restore_compiled_request(saved, compiler, *, images=(), pixel_cache=None):
    """Restore an exact internal query, including candidate-free JEPA2 queries.

    Unlike restore_request this does not permute candidates or infer a primitive.
    Its caller verifies the objective/compiler version before reading the cache.
    """
    questions = {
        qid: Question(tuple(q["keys"]), tuple(q["tokens"]), tuple(q["candidate_positions"]),
                      q["decision_position"], q["instruction_tokens"],
                      tuple(q["candidate_starts"]))
        for qid, q in saved["questions"].items()
    }
    state = tuple(saved["state"])
    total = len(state) + sum(len(q.tokens) for q in questions.values())
    if not questions or total != saved["input_tokens"]:
        raise DataError("Cached internal request has inconsistent token counts")
    if total > compiler.max_request_tokens or any(
        len(state) + len(q.tokens) > compiler.max_sequence_tokens for q in questions.values()
    ):
        raise DataError("Stored token IDs exceed configured token budget; input was not truncated")
    for q in questions.values():
        if (len(q.keys) != len(q.candidate_positions)
                or not 0 <= q.decision_position < len(q.tokens)
                or q.tokens[q.decision_position] != compiler.guard.markers["decision_end"]
                or any(not 0 <= p < q.decision_position
                       or q.tokens[p] != compiler.guard.markers["candidate_end"]
                       for p in q.candidate_positions)):
            raise DataError("Cached internal request has invalid readout positions")
    pixels, contents = None, ()
    if images:
        cache = {} if pixel_cache is None else pixel_cache
        key = tuple(images)
        if key not in cache:
            cache[key] = image_pixels(key, compiler.image_processor)
        pixels, contents = cache[key]
    return Request(state, questions, total, contents, pixels)


def _schema(original, compiler):
    base = pa.schema([field for field in original if field.name not in TOKEN_COLUMNS])
    result = base.append(pa.field("_tokenized", pa.large_string()))
    result = result.append(pa.field("_token_costs", pa.list_(pa.int32())))
    result = result.append(pa.field("_split_group", pa.string()))
    result = result.append(pa.field("_group_inferred", pa.bool_()))
    from datasets import Features

    return result.with_metadata(
        {
            **(original.metadata or {}),
            **Features.from_arrow_schema(result).arrow_schema.metadata,
            METADATA_KEY: entry(compiler_settings(compiler)).encode(),
        }
    )


def load_compiler(base, *, revision=None, max_request_tokens=64000, max_sequence_tokens=16384,
                  encoder_questions=False):
    from transformers import AutoConfig, AutoTokenizer, Gemma3ImageProcessorPil

    local = Path(base).is_dir()
    options = {"revision": revision, "local_files_only": local}
    compiler = Compiler(
        AutoTokenizer.from_pretrained(base, **options),
        AutoConfig.from_pretrained(base, **options),
        image_processor=Gemma3ImageProcessorPil.from_pretrained(base, **options),
        max_request_tokens=max_request_tokens,
        max_sequence_tokens=max_sequence_tokens,
        encoder_questions=encoder_questions,
    )
    # Repeated question templates dominate the large dataset. This changes no IDs.
    question = compiler.question
    cached = lru_cache(maxsize=8192)(lambda text: question(json.loads(text)))
    compiler.question = lambda value: cached(entry(value))
    return compiler


_worker_source = None
_worker_compiler = None


def _init_worker(path, base, options):
    global _worker_source, _worker_compiler
    import torch

    torch.set_num_threads(1)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _worker_source = pq.ParquetFile(path)
    _worker_compiler = load_compiler(base, **options)


def _compile_group(number, source=None, compiler=None):
    source = source or _worker_source
    compiler = compiler or _worker_compiler
    columns = [name for name in source.schema_arrow.names if name not in TOKEN_COLUMNS]
    table = source.read_row_group(number, columns=columns)
    prepared = []
    for row in table.to_pylist():
        try:
            prepared.append(compile_record(training_record(unpack(row)), compiler))
        except Exception as exc:
            raise DataError(f"Tokenization failed for {row['id']}: {exc}") from exc
    extra = pa.Table.from_pylist(
        prepared, schema=pa.schema(list(_schema(source.schema_arrow, compiler))[-4:])
    )
    for name in TOKEN_COLUMNS:
        table = table.append_column(name, extra[name])
    return table


def _signature(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def tokenize_parquet(path, compiler, *, workers=1, base=None, options=None):
    """Keep the canonical path and every original row; atomically add compiled columns."""
    path = Path(path).resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    signature = _signature(path)
    manifest_path = path.parent / "manifest.json"
    manifest_bytes = manifest_path.read_bytes() if manifest_path.exists() else None
    source = pq.ParquetFile(path)
    if (source.schema_arrow.metadata or {}).get(b"bongard_format") != FORMAT:
        raise DataError("tokenize requires a canonical embedded-image Parquet")
    schema = _schema(source.schema_arrow, compiler)
    rows = tokens = paired = 0
    started = time.monotonic()
    last_report = started
    with tempfile.TemporaryDirectory(prefix=".tokenize-", dir=path.parent) as directory:
        staged = Path(directory) / path.name
        with pq.ParquetWriter(
            staged,
            schema,
            compression="zstd",
            compression_level=3,
            use_dictionary=["dataset", "schema_version"],
        ) as writer:

            def write(table):
                nonlocal rows, tokens, paired, last_report
                for costs in table["_token_costs"].to_pylist():
                    tokens += costs[0]
                    paired += len(costs) > 1
                rows += len(table)
                writer.write_table(table.replace_schema_metadata(schema.metadata))
                if time.monotonic() - last_report > 20:
                    print(
                        json.dumps(
                            {
                                "file": path.name,
                                "tokenized_rows": rows,
                                "total_rows": source.metadata.num_rows,
                                "seconds": round(time.monotonic() - started, 1),
                            }
                        ),
                        flush=True,
                    )
                    last_report = time.monotonic()

            if workers == 1:
                for group in range(source.num_row_groups):
                    write(_compile_group(group, source, compiler))
            else:
                if base is None:
                    raise ValueError("Parallel tokenization requires a tokenizer directory/model")
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(str(path), str(base), options or {}),
                ) as pool:
                    groups = iter(range(source.num_row_groups))
                    pending = deque(
                        pool.submit(_compile_group, group)
                        for group in list(range(min(2 * workers, source.num_row_groups)))
                    )
                    for _ in range(len(pending)):
                        next(groups)
                    while pending:
                        write(pending.popleft().result())
                        group = next(groups, None)
                        if group is not None:
                            pending.append(pool.submit(_compile_group, group))
        if rows != source.metadata.num_rows or pq.ParquetFile(staged).metadata.num_rows != rows:
            raise DataError("Tokenization changed the number of records")
        if _signature(path) != signature or (
            (manifest_path.read_bytes() if manifest_path.exists() else None) != manifest_bytes
        ):
            raise DataError(
                "Dataset or manifest changed during tokenization; original was not replaced"
            )
        source.close()
        os.replace(staged, path)
        if manifest_bytes is not None:
            manifest = json.loads(manifest_bytes)
            category = (schema.metadata or {}).get(b"training_category", b"").decode()
            if category in manifest.get("categories", {}):
                manifest["categories"][category]["bytes"] = path.stat().st_size
                temporary_manifest = Path(directory) / "manifest.json"
                temporary_manifest.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
                )
                os.replace(temporary_manifest, manifest_path)
    report = {
        "file": str(path),
        "records": rows,
        "base_tokens": tokens,
        "records_with_views": paired,
        "seconds": round(time.monotonic() - started, 1),
    }
    print(json.dumps(report), flush=True)
    return report
