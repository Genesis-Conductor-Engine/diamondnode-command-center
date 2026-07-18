#!/usr/bin/env python3
"""Structural regression tests for the credential-blind WebGPU HUD."""
from pathlib import Path
import re
import unittest


HUD_PATH = Path("/home/diamondnode/bin/webgpu-self-observers.html")


class WebGpuHudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = HUD_PATH.read_text()

    def test_renders_only_the_allowlisted_inference_fields_as_text(self):
        self.assertIn("const inference = state.inference || {};", self.source)
        self.assertIn("function renderInferenceHud(inference)", self.source)
        self.assertIn("textContent", self.source)
        self.assertNotIn("innerHTML", self.source)

        for field in (
            "status",
            "engine",
            "age_s",
            "gate",
            "samples_per_s",
            "min_energy",
        ):
            self.assertIn(f"inference.{field}", self.source)

    def test_labels_stale_and_unavailable_inference_without_key_or_ingest_patterns(self):
        self.assertIn("inference.status === 'stale'", self.source)
        self.assertIn("inference.status === 'unavailable'", self.source)
        self.assertNotRegex(
            self.source,
            re.compile(r"stream[_ -]?key|ingest(?:[_ -]?(?:id|url))?|rtmp://", re.IGNORECASE),
        )


if __name__ == "__main__":
    unittest.main()
