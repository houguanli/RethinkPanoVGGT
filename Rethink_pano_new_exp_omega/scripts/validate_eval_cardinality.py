#!/usr/bin/env python3
"""Fail unless a formal mixed4 evaluation contains all 8,833 anchor sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_FULL_ANCHOR_SETS = {
    "Stanford2D3DS": 216,
    "Matterport3D": 891,
    "Structured3D": 1662,
    "Panocity": 6064,
}
EXPECTED_FULL_ANCHOR_TOTAL = sum(EXPECTED_FULL_ANCHOR_SETS.values())


def evaluated_sets_by_dataset(summary: dict[str, Any]) -> dict[str, int]:
    actual: dict[str, int] = {}
    for run in summary.get("runs", []):
        name = str(run.get("name", ""))
        for dataset in EXPECTED_FULL_ANCHOR_SETS:
            if name.startswith(f"{dataset}_"):
                actual[dataset] = actual.get(dataset, 0) + int(run.get("evaluated_samples", -1))
                break
    return actual


def validate_full_anchor_summary(summary: dict[str, Any]) -> dict[str, int]:
    policy = summary.get("sample_policy")
    if policy != "anchor":
        raise ValueError(f"formal eval requires sample_policy='anchor', got {policy!r}")

    actual = evaluated_sets_by_dataset(summary)
    if actual != EXPECTED_FULL_ANCHOR_SETS:
        raise ValueError(
            "formal eval cardinality mismatch: "
            f"expected={EXPECTED_FULL_ANCHOR_SETS} total={EXPECTED_FULL_ANCHOR_TOTAL}; "
            f"actual={actual} total={sum(actual.values())}"
        )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    actual = validate_full_anchor_summary(summary)
    print(f"[CHECK] formal eval cardinality verified: {actual} total={sum(actual.values())}")


if __name__ == "__main__":
    main()
