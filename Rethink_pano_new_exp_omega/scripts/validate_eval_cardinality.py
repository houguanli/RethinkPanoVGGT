#!/usr/bin/env python3
"""Fail unless a formal mixed4 evaluation contains all 8,833 anchor sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
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
    parser.add_argument("summary", type=Path, nargs="?")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--index-audit", type=Path, help="Verify all test indices and persist/check their fingerprint before training/eval.")
    args = parser.parse_args()

    if args.index_audit:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from training.train_pano_omega import parse_args, build_dataset
        from scripts.evaluate_mixed4_depth_checkpoint import DATASETS, apply_exact_eval_pano_count
        records = {}
        for display_name, minimal_name, split in DATASETS:
            config = parse_args(["--config", str(args.config)])
            config.dataset_root = args.dataset_root
            config.dataset_split, config.minimal_datasets = split, minimal_name
            config.dataset_max_samples = None
            count = 10 if minimal_name == "panocity" else 3
            apply_exact_eval_pano_count(config, count, minimal_name)
            dataset = build_dataset(config, (512, 1024))
            assert len(dataset) == EXPECTED_FULL_ANCHOR_SETS[display_name], (display_name, len(dataset))
            assert all(len(group) == count for group in dataset.groups), display_name
            fingerprint = hashlib.sha256(json.dumps({"items": dataset.items, "groups": dataset.groups},
                                                      sort_keys=True, default=str).encode()).hexdigest()
            records[display_name] = {"anchor_sets": len(dataset), "input_panos": count, "index_sha256": fingerprint}
            print(f"[INDEX] {display_name}: {records[display_name]}", flush=True)
        if args.index_audit.exists():
            assert json.loads(args.index_audit.read_text()) == records, "Test indices changed since initial audit"
        else:
            args.index_audit.parent.mkdir(parents=True, exist_ok=True)
            args.index_audit.write_text(json.dumps(records, indent=2))
        return

    if args.summary is None:
        parser.error("summary or --index-audit is required")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    actual = validate_full_anchor_summary(summary)
    print(f"[CHECK] formal eval cardinality verified: {actual} total={sum(actual.values())}")


if __name__ == "__main__":
    main()
