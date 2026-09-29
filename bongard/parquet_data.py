"""Portable canonical/predictive records with image bytes embedded in Parquet."""

from __future__ import annotations

import base64
import copy
import json
import mimetypes
from pathlib import Path

from .data import DataError, strict_loads

FORMAT = b"bongard-embedded-images-v1"


def schema():
    from datasets import Features, Image, List, Value

    result = Features(
        {
            "id": Value("string"),
            "dataset": Value("string"),
            "schema_version": Value("string"),
            "record_json": Value("large_string"),
            "images": List(Image(decode=False)),
        }
    ).arrow_schema
    return result.with_metadata({**result.metadata, b"bongard_format": FORMAT})


def requests(record):
    if isinstance(record, dict):
        for key, value in record.items():
            if key == "request" and isinstance(value, dict):
                yield value
            else:
                yield from requests(value)
    elif isinstance(record, list):
        for value in record:
            yield from requests(value)


def dataset_name(record):
    return record.get("metadata", {}).get("dataset") or record["id"].split(":")[0]


def pack(record):
    record = copy.deepcopy(record)
    images = []
    for request in requests(record):
        embedded = []
        for value in request.get("images", []):
            if value.startswith("data:image/") and ";base64," in value:
                header, payload = value.split(",", 1)
                mime = header[5:].split(";", 1)[0]
                content = base64.b64decode(payload, validate=True)
            else:
                path = Path(value)
                if not path.is_file():
                    raise DataError("image_must_be_local_or_data_url:" + value[:120])
                mime = mimetypes.guess_type(path.name)[0]
                if not mime or not mime.startswith("image/"):
                    raise DataError("unknown_local_image_mime:" + str(path))
                content = path.read_bytes()
            if not content:
                raise DataError("empty_image_bytes")
            embedded.append({"image_index": len(images), "mime_type": mime})
            images.append({"bytes": content, "path": None})
        if "images" in request:
            request["images"] = embedded
    return {
        "id": record["id"],
        "dataset": dataset_name(record),
        "schema_version": record["schema_version"],
        "record_json": json.dumps(
            record, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ),
        "images": images,
    }


def unpack(row):
    record = strict_loads(row["record_json"])
    images, used = row["images"], []
    if record["id"] != row["id"] or record["schema_version"] != row["schema_version"]:
        raise DataError("packed_record_identity_mismatch")
    for request in requests(record):
        restored = []
        for ref in request.get("images", []):
            index = ref["image_index"]
            if type(index) is not int or not 0 <= index < len(images):
                raise DataError("invalid_embedded_image_index")
            item = images[index]
            if (
                not isinstance(item["bytes"], bytes)
                or not item["bytes"]
                or item["path"] is not None
            ):
                raise DataError("image_must_be_embedded_bytes_without_path")
            used.append(index)
            restored.append(
                "data:"
                + ref["mime_type"]
                + ";base64,"
                + base64.b64encode(item["bytes"]).decode("ascii")
            )
        if "images" in request:
            request["images"] = restored
    if sorted(used) != list(range(len(images))):
        raise DataError("unreferenced_or_reused_embedded_image")
    return record


def iter_records(path, *, batch_size=128):
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    if (source.schema_arrow.metadata or {}).get(b"bongard_format") != FORMAT:
        raise DataError("not_bongard_embedded_parquet")
    for batch in source.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield unpack(row)


class Writer:
    def __init__(self, path, *, category, max_rows=1024, max_bytes=16 * 1024**2):
        import pyarrow.parquet as pq

        self.path = Path(path)
        self.schema = schema()
        self.schema = self.schema.with_metadata(
            {**self.schema.metadata, b"training_category": category.encode()}
        )
        self.writer = pq.ParquetWriter(
            self.path,
            self.schema,
            compression="zstd",
            compression_level=3,
            use_dictionary=["dataset", "schema_version"],
            write_statistics=["id", "dataset", "schema_version"],
        )
        self.rows, self.bytes = [], 0
        self.max_rows, self.max_bytes = max_rows, max_bytes

    def add(self, record):
        row = pack(record)
        self.rows.append(row)
        self.bytes += len(row["record_json"].encode()) + sum(len(i["bytes"]) for i in row["images"])
        if len(self.rows) >= self.max_rows or self.bytes >= self.max_bytes:
            self.flush()

    def flush(self):
        if self.rows:
            import pyarrow as pa

            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows, self.bytes = [], 0

    def close(self):
        self.flush()
        self.writer.close()
