"""Disk-backed HF Datasets without imposing an Arrow schema on dynamic questions."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path

from datasets import Dataset, Features, Value
from datasets.exceptions import DatasetGenerationError

from .data import DataError, iter_rows, strict_loads
from .predictive_data import ADAPTER_VERSION, training_record


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_rows(path):
    found = False
    for row in iter_rows(Path(path)):
        found = True
        row = training_record(row)
        # Dynamic keys, heterogeneous state/criteria, and option order are part
        # of the model input. Arrow struct inference would insert null fields
        # or reject mixed types. Keep the validated JSON losslessly in a column.
        yield {"record_json": json.dumps(row, ensure_ascii=False, allow_nan=False)}
    if not found:
        raise DataError(f"Empty dataset: {path}")


class CanonicalDataset(Sequence):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        if isinstance(index, slice):
            return [strict_loads(value) for value in row["record_json"]]
        return strict_loads(row["record_json"])

    def __iter__(self):
        for row in self.dataset:
            yield strict_loads(row["record_json"])


class TokenizedDataset(Sequence):
    """Memory-mapped packed rows; planning reads only small ID/group/cost columns."""

    tokens_ready = True

    def __init__(self, dataset, settings):
        self.dataset = dataset
        self.settings = settings
        self.ids = dataset.data.column("id")
        self.costs = dataset.data.column("_token_costs")
        if dataset.data.column("_tokenized").null_count or self.costs.null_count:
            raise DataError("Some rows have no token IDs; run tokenize after appending data")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        from .parquet_data import unpack

        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        row = self.dataset[index]
        record = training_record(unpack(row))
        record["_tokenized"] = json.loads(row["_tokenized"])
        record["_token_settings"] = self.settings
        return record

    def metadata_records(self):
        columns = self.dataset.select_columns(["id", "_split_group", "_group_inferred"])
        for batch in columns.iter(batch_size=4096):
            for record_id, group, inferred in zip(
                batch["id"], batch["_split_group"], batch["_group_inferred"], strict=True
            ):
                yield {"id": record_id, "metadata": {
                    "split_group": group, "split_group_inferred": inferred,
                }}

    def token_cost(self, index, config, epoch):
        costs = self.costs[index].as_py()
        if config.jepa.enabled and len(costs) > 1:
            rng = random.Random(f"{config.seed}:{epoch}:{self.ids[index].as_py()}")
            return costs[1 + rng.randrange(len(costs) - 1)]
        return costs[0]

    def verify_compiler(self, compiler):
        from .tokenized_data import verify_compiler

        verify_compiler(self.settings, compiler)


def read_dataset(path, *, cache_dir=None):
    path = Path(path).resolve()
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        from .tokenized_data import METADATA_KEY

        settings = (pq.read_schema(path).metadata or {}).get(METADATA_KEY)
        if settings is not None:
            dataset = Dataset.from_parquet(str(path), cache_dir=cache_dir, keep_in_memory=False)
            return TokenizedDataset(dataset, json.loads(settings))
    # Include validation/compiler-independent IR version in the cache identity.
    fingerprint = f"canonical-v2-{ADAPTER_VERSION}-{path.suffix.lower()[1:]}-" + file_sha256(path)
    try:
        dataset = Dataset.from_generator(
            _canonical_rows,
            gen_kwargs={"path": str(path)},
            features=Features({"record_json": Value("large_string")}),
            fingerprint=fingerprint,
            cache_dir=cache_dir,
            keep_in_memory=False,
        )
    except DatasetGenerationError as exc:
        if isinstance(exc.__cause__, (DataError, ValueError, TypeError)):
            raise exc.__cause__ from exc
        raise
    if not len(dataset):
        raise DataError(f"Empty dataset: {path}")
    return CanonicalDataset(dataset)
