"""Exhaustive boundary and truth-table tests for the registered verdicts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp205_verdict", HERE / "verdict.py")
assert SPEC is not None and SPEC.loader is not None
VERDICT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERDICT
SPEC.loader.exec_module(VERDICT)


class PrimaryTruthTable(unittest.TestCase):
    def test_every_reachable_state(self) -> None:
        cases = (
            ((0.85, 0.72, 0.55), "OPERATIONAL_CROSSMIC_CONFIRMATION"),
            ((0.85, 0.72, 0.50), "PRIMARY_ONLY"),
            ((0.79, 0.60, 0.55), "REPLICATED_WEAK_POSITIVE"),
            ((0.79, 0.60, 0.50), "WEAK_PRIMARY_ONLY"),
            ((0.70, 0.50, 0.55), "SECONDARY_ONLY_NOT_CONFIRMED"),
            ((0.70, 0.50, 0.50), "NOT_CONFIRMED"),
        )
        observed = set()
        for (point, lcb, wavlm_lcb), expected in cases:
            with self.subTest(expected=expected):
                result = VERDICT.primary_verdict(
                    ecapa_point=point, ecapa_lcb=lcb, wavlm_lcb=wavlm_lcb
                )
                self.assertEqual(result.verdict, expected)
                observed.add(result.verdict)
        self.assertEqual(len(observed), 6)

    def test_strict_interval_boundaries_and_closed_point_bar(self) -> None:
        self.assertFalse(
            VERDICT.primary_verdict(
                ecapa_point=0.80, ecapa_lcb=0.50, wavlm_lcb=0.50
            ).existence
        )
        self.assertFalse(
            VERDICT.primary_verdict(
                ecapa_point=0.80, ecapa_lcb=0.70, wavlm_lcb=0.50
            ).operational
        )
        self.assertTrue(
            VERDICT.primary_verdict(
                ecapa_point=0.80, ecapa_lcb=0.7000001, wavlm_lcb=0.50
            ).operational
        )
        self.assertFalse(
            VERDICT.primary_verdict(
                ecapa_point=0.90, ecapa_lcb=0.80, wavlm_lcb=0.50
            ).representation_replication
        )

    def test_invalid_numeric_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            VERDICT.primary_verdict(
                ecapa_point=0.60, ecapa_lcb=0.70, wavlm_lcb=0.55
            )
        with self.assertRaises(ValueError):
            VERDICT.primary_verdict(
                ecapa_point=1.01, ecapa_lcb=0.70, wavlm_lcb=0.55
            )


class ReverseAndHeadline(unittest.TestCase):
    def test_all_reverse_states(self) -> None:
        cases = (
            ((0.51, 0.51), "BIDIRECTIONAL_REPLICATION"),
            ((0.51, 0.50), "REVERSE_REPRESENTATION_DEPENDENT"),
            ((0.50, 0.51), "REVERSE_REPRESENTATION_DEPENDENT"),
            ((0.50, 0.50), "REVERSE_NOT_CONFIRMED"),
        )
        for (ecapa_lcb, wavlm_lcb), expected in cases:
            with self.subTest(expected=expected):
                result = VERDICT.reverse_verdict(
                    ecapa_lcb=ecapa_lcb, wavlm_lcb=wavlm_lcb
                )
                self.assertEqual(result.verdict, expected)

    def test_reverse_cannot_rescue_primary(self) -> None:
        reverse = VERDICT.reverse_verdict(ecapa_lcb=0.60, wavlm_lcb=0.60)
        weak = VERDICT.primary_verdict(
            ecapa_point=0.79, ecapa_lcb=0.60, wavlm_lcb=0.60
        )
        self.assertEqual(
            VERDICT.headline_permission(weak, reverse), "NO_OPERATIONAL_HEADLINE"
        )

    def test_only_full_intersection_permits_bidirectional_headline(self) -> None:
        primary = VERDICT.primary_verdict(
            ecapa_point=0.80, ecapa_lcb=0.71, wavlm_lcb=0.51
        )
        reverse = VERDICT.reverse_verdict(ecapa_lcb=0.51, wavlm_lcb=0.51)
        self.assertEqual(
            VERDICT.headline_permission(primary, reverse),
            "BIDIRECTIONAL_HEADLINE_PERMITTED",
        )


if __name__ == "__main__":
    unittest.main()
