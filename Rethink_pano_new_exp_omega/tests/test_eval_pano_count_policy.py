import unittest
from types import SimpleNamespace

from scripts.evaluate_mixed4_depth_checkpoint import (
    PANOVGGT_DATASET_PANO_COUNTS,
    build_parser,
    resolve_dataset_pano_counts,
    apply_exact_eval_pano_count,
    validate_evaluated_pano_counts,
)


class EvalPanoCountPolicyTest(unittest.TestCase):
    def test_exact_eval_policy_overrides_training_dataset_cap(self) -> None:
        for training_count in (2, 6):
            args = SimpleNamespace(pano_min_count=2, pano_max_count=training_count,
                                   dataset_pano_max_counts=f"panocity:{training_count},matterport3d:2")
            for dataset, count in PANOVGGT_DATASET_PANO_COUNTS.items():
                apply_exact_eval_pano_count(args, count, dataset)
                self.assertEqual((args.pano_min_count, args.pano_max_count), (count, count))
                self.assertEqual(args.dataset_pano_max_counts, f"{dataset}:{count}")

    def test_rejects_silent_clamping_and_incompatible_resume_rows(self) -> None:
        validate_evaluated_pano_counts(
            [{"dataset": "Panocity", "input_pano_count": "10"}], PANOVGGT_DATASET_PANO_COUNTS)
        for row in ({"dataset": "Panocity", "input_pano_count": "6"},
                    {"run": "Panocity_test_0", "input_pano_count": "6"}):
            with self.assertRaisesRegex(ValueError, "expected 10, got 6"):
                validate_evaluated_pano_counts([row], PANOVGGT_DATASET_PANO_COUNTS)

    def test_eval_defaults_to_panovggt_multi_pano_counts(self) -> None:
        args = build_parser().parse_args(
            ["--config", "config.yaml", "--checkpoint", "model.pt", "--output", "summary.json"]
        )

        self.assertEqual(args.pano_count_policy, "panovggt")
        self.assertEqual(args.sample_policy, "anchor")
        self.assertEqual(
            resolve_dataset_pano_counts(args.pano_count_policy, None),
            {
                "panocity": 10,
                "matterport3d": 3,
                "stanford2d3ds": 3,
                "structured3d": 3,
            },
        )

    def test_single_pano_policy_forces_one_pano_per_dataset(self) -> None:
        counts = resolve_dataset_pano_counts("single", None)

        self.assertEqual(counts, {name: 1 for name in PANOVGGT_DATASET_PANO_COUNTS})

    def test_explicit_dataset_count_overrides_policy(self) -> None:
        counts = resolve_dataset_pano_counts(
            "single",
            "panocity:2,stanford2d3ds:4",
        )

        self.assertEqual(counts["panocity"], 2)
        self.assertEqual(counts["stanford2d3ds"], 4)
        self.assertEqual(counts["matterport3d"], 1)


if __name__ == "__main__":
    unittest.main()
