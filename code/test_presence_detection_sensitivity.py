#!/usr/bin/env python3
"""Failure-injection checks for the presence-detection estimators."""

from __future__ import annotations

import unittest

import numpy as np

import presence_detection_sensitivity as pds


class EstimatorTests(unittest.TestCase):
    def test_separable_scores_give_zero_error_and_zero_cost(self) -> None:
        rng = np.random.default_rng(1)
        target = rng.uniform(1.0, 2.0, 500)
        nontarget = rng.uniform(-2.0, -1.0, 500)
        self.assertEqual(pds.equal_error_rate(target, nontarget), 0.0)
        self.assertEqual(pds.normalized_min_dcf(target, nontarget), 0.0)

    def test_identical_distributions_give_chance_and_reject_all(self) -> None:
        scores = np.random.default_rng(2).normal(0.0, 1.0, 500)
        self.assertAlmostEqual(pds.equal_error_rate(scores, scores), 0.5, places=12)
        self.assertAlmostEqual(pds.normalized_min_dcf(scores, scores), 1.0, places=12)

    def test_reject_all_threshold_is_representable(self) -> None:
        # Non-targets above every target: the cheapest decision is to reject everything.
        target = np.array([0.1, 0.2, 0.3])
        nontarget = np.array([0.4, 0.5, 0.6])
        self.assertEqual(pds.normalized_min_dcf(target, nontarget), 1.0)
        self.assertEqual(pds.equal_error_rate(target, nontarget), 1.0)

    def test_hand_computed_small_case(self) -> None:
        # Threshold .35 misses one of four targets and accepts one of four non-targets.
        target = np.array([0.3, 0.4, 0.5, 0.6])
        nontarget = np.array([0.0, 0.1, 0.2, 0.4])
        self.assertAlmostEqual(pds.equal_error_rate(target, nontarget), 0.25, places=12)
        cost = min(0.01 * 0.25 + 0.99 * 0.25, 0.01 * 0.5 + 0.99 * 0.0, 0.01 * 1.0)
        self.assertAlmostEqual(pds.normalized_min_dcf(target, nontarget), cost / 0.01, places=12)


if __name__ == "__main__":
    unittest.main()
