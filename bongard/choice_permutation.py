"""Training-only candidate reordering for reviewed Choice rubrics.

The catalogue of reviewed rubrics is not part of this release, so no question is eligible and
records keep their option order. Supply your own reordering here to train order invariance.
"""

from __future__ import annotations


def permute(record, seed):
    return record
