"""Record adapter hook used by the dataset readers.

Canonical records pass through validation unchanged. The adapter version is part of the
tokenized-data fingerprint, so it stays fixed.
"""

from __future__ import annotations

from .data import validate_record

ADAPTER_VERSION = "execution-readouts-v1"


def training_record(row):
    return validate_record(row)
