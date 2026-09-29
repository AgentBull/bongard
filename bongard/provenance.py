"""Preserve split membership across training courses and independent evaluations."""

from __future__ import annotations

from pathlib import Path

from .data import DataError, check_split_groups, metadata_records, strict_loads
from .dataset import file_sha256, read_dataset

SPLITS = ("train", "dev", "calibration", "test")


def membership(records):
    check_split_groups({"dataset": records})
    return {
        "split_groups": sorted({r["metadata"]["split_group"] for r in metadata_records(records)}),
        "record_ids": sorted({r["id"] for r in metadata_records(records)}),
    }


def source_snapshot(split, info):
    """Upgrade legacy path/hash metadata while the original dataset is available."""
    if split not in SPLITS:
        raise DataError(f"Unknown provenance split: {split}")
    result = {**info, "split": split}
    if "split_groups" not in result or "record_ids" not in result:
        path = Path(info["path"])
        if not path.is_file():
            raise DataError(f"Cannot recover split provenance: dataset is missing: {path}")
        if file_sha256(path) != info["sha256"]:
            raise DataError(f"Dataset changed since training: {path}")
        records = read_dataset(path)
        result.update(membership(records))
    for key in ("split_groups", "record_ids"):
        values = result[key]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise DataError(f"Invalid split provenance: {key}")
    return result


def checkpoint_sources(checkpoint, *, required=False, _seen=None):
    """Modern checkpoints are self-contained; older ones must recover every ancestor."""
    checkpoint = Path(checkpoint).resolve()
    seen = set() if _seen is None else _seen
    if checkpoint in seen:
        raise DataError(f"Cyclic checkpoint provenance: {checkpoint}")
    seen.add(checkpoint)
    metadata_path = checkpoint / "training.json"
    if not metadata_path.is_file():
        if required:
            raise DataError(f"Training split provenance is missing: {metadata_path}")
        return []
    metadata = strict_loads(metadata_path.read_text())
    if "sources" in metadata:
        sources = [source_snapshot(info["split"], info) for info in metadata["sources"]]
    else:
        parent = metadata.get("config", {}).get("init_checkpoint")
        sources = checkpoint_sources(parent, required=True, _seen=seen) if parent else []
        sources.extend(source_snapshot(split, info) for split, info in metadata["data"].items())
    if not sources or not any(info["split"] == "train" for info in sources):
        raise DataError(f"Training split provenance is incomplete: {metadata_path}")
    return sources


def check_source_overlap(candidate, excluded, purpose):
    for source in excluded:
        for key in ("split_groups", "record_ids"):
            overlap = set(candidate[key]) & set(source[key])
            if overlap:
                value = sorted(overlap)[0]
                raise DataError(
                    f"cross-split leakage: {purpose} {key} contains {value!r} "
                    f"from prior {source['split']} data ({source.get('path', 'temperature file')})"
                )


def course_sources(manifest, inherited):
    current = [source_snapshot(split, info) for split, info in manifest.items()]
    for source in current:
        # Reusing a split in a later course is valid, changing its purpose is not.
        check_source_overlap(
            source, [old for old in inherited if old["split"] != source["split"]], source["split"]
        )
    unique = {}
    for source in [*inherited, *current]:
        unique[(source["split"], source["sha256"])] = source
    return list(unique.values())


def check_independent_data(records, sources, purpose, *, calibration_groups=()):
    candidate = membership(records)
    allowed_split = "test" if purpose == "evaluation" else "calibration"
    excluded = [source for source in sources if source["split"] != allowed_split]
    if calibration_groups:
        excluded.append(
            {"split": "calibration", "split_groups": calibration_groups, "record_ids": []}
        )
    check_source_overlap(candidate, excluded, purpose)
