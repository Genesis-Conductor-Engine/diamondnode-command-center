#!/usr/bin/env python3
"""Unit tests for procedural_truth_verifier."""

import json
import unittest
from unittest.mock import patch

from procedural_truth_verifier import (
    ProceduralTruthVerifier,
    VerifierConfig,
    automaton_emergence_score,
    rule30_vdf,
    seed_from_traces,
    build_laplacian,
    eigen_project,
    LoopbackTrace,
    compute_crystal_score,
    _default_plan_graph,
)


class TestRule30(unittest.TestCase):
    def test_vdf_deterministic(self):
        final1, _ = rule30_vdf(12345, 32)
        final2, _ = rule30_vdf(12345, 32)
        self.assertEqual(final1, final2)

    def test_vdf_changes_with_steps(self):
        f8, _ = rule30_vdf(42, 8)
        f64, _ = rule30_vdf(42, 64)
        self.assertNotEqual(f8, f64)


class TestSeedFromTraces(unittest.TestCase):
    def test_seed_from_hashes(self):
        traces = [
            LoopbackTrace("a", "http://x", 200, 1.0, "abc123", "ok", True),
            LoopbackTrace("b", "http://y", 200, 2.0, "def456", "ok", True),
        ]
        seed = seed_from_traces(traces)
        self.assertIsInstance(seed, int)
        self.assertGreater(seed, 0)


class TestEigenProjection(unittest.TestCase):
    def test_laplacian_eigen(self):
        nodes, edges = _default_plan_graph()
        lap = build_laplacian(len(nodes), edges)
        signal = [1.0] * len(nodes)
        result = eigen_project(lap, signal, k=3)
        self.assertIn("loopback_delta_eigen", result)
        self.assertIn("coefficients", result)


class TestCrystalScore(unittest.TestCase):
    def test_pass_high_score(self):
        cs = compute_crystal_score(True, 0.05, 0.95, target=0.85)
        self.assertTrue(cs["passed"])
        self.assertGreaterEqual(cs["value"], 0.85)

    def test_fail_low_loopback(self):
        cs = compute_crystal_score(False, 0.05, 0.95, target=0.85)
        self.assertFalse(cs["passed"])


class TestAutomatonEmergence(unittest.TestCase):
    def test_identical_vectors_high_score(self):
        v = [1.0, 0.5, 0.3, 0.8]
        score = automaton_emergence_score(v, v, 0xABCD)
        self.assertGreater(score, 0.5)


class TestVerifierIntegration(unittest.TestCase):
    @patch("procedural_truth_verifier.probe_endpoint")
    def test_verify_mock_loopback(self, mock_probe):
        mock_probe.side_effect = [
            LoopbackTrace("q-mem", "http://127.0.0.1:8082/health", 200, 5.0, "a" * 16, "{}", True),
            LoopbackTrace("ollama_v1", "http://127.0.0.1:11434/api/tags", 200, 8.0, "b" * 16, "{}", True),
        ]
        cfg = VerifierConfig(crystal_target=0.50)  # lower threshold for mock
        result = ProceduralTruthVerifier(cfg).verify(source="test")
        self.assertIn("evt_id", result.evt)
        self.assertEqual(result.evt["record_type"], "procedural_truth")
        self.assertIn("crystal_score", result.evt)
        json.dumps(result.evt)  # serializable


if __name__ == "__main__":
    unittest.main()