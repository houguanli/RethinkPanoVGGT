import unittest

from scripts.validate_eval_cardinality import (
    EXPECTED_FULL_ANCHOR_SETS,
    validate_full_anchor_summary,
)


def make_summary(policy="anchor", counts=None):
    counts = counts or EXPECTED_FULL_ANCHOR_SETS
    return {
        "sample_policy": policy,
        "runs": [
            {"name": f"{dataset}_test_0", "evaluated_samples": count}
            for dataset, count in counts.items()
        ],
    }


class EvalCardinalityTest(unittest.TestCase):
    def test_accepts_exact_full_anchor_eval(self) -> None:
        self.assertEqual(
            validate_full_anchor_summary(make_summary()),
            EXPECTED_FULL_ANCHOR_SETS,
        )

    def test_rejects_scene_neighborhood_eval(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_policy"):
            validate_full_anchor_summary(make_summary(policy="scene_neighborhood"))

    def test_rejects_incomplete_anchor_eval(self) -> None:
        counts = dict(EXPECTED_FULL_ANCHOR_SETS)
        counts["Panocity"] = 253
        with self.assertRaisesRegex(ValueError, "cardinality mismatch"):
            validate_full_anchor_summary(make_summary(counts=counts))


if __name__ == "__main__":
    unittest.main()
