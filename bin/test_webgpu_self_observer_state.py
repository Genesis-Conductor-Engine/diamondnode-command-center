#!/usr/bin/env python3
"""Unit tests for the credential-blind inference state adapter."""
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path("/home/diamondnode/bin/webgpu-self-observer-state.py")
SPEC = importlib.util.spec_from_file_location("webgpu_self_observer_state", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


NOW = 1_700_000_000.0
UNAVAILABLE = {
    "status": "unavailable",
    "engine": None,
    "age_s": None,
    "gate": None,
    "samples_per_s": None,
    "min_energy": None,
}
WITHHELD = "[sensitive content withheld]"
CHAT_FIELDS = {"question", "reply", "host", "host_label", "viewer", "source"}


def fresh_state(**overrides):
    state = {
        "ts": NOW - 90,
        "ok": True,
        "energy_gate": {"gpu_ok": True, "reason": "within governor envelope"},
        "result": {
            "engine": "thrml-0.1.3/jax-gpu",
            "samples_per_s": 128.5,
            "min_energy": -42.25,
        },
    }
    state.update(overrides)
    return state


class ReadInferenceStateTests(unittest.TestCase):
    def write_state(self, payload: object) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "state.json"
        path.write_text(json.dumps(payload))
        return path

    def test_returns_allowlisted_fresh_jax_gpu_measurement(self):
        result = MODULE.read_inference_state(self.write_state(fresh_state()), NOW)

        self.assertEqual(
            result,
            {
                "status": "fresh",
                "engine": "thrml-0.1.3/jax-gpu",
                "age_s": 90,
                "gate": "open",
                "samples_per_s": 128.5,
                "min_energy": -42.25,
            },
        )

    def test_labels_result_stale_after_six_hours(self):
        result = MODULE.read_inference_state(
            self.write_state(fresh_state(ts=NOW - (6 * 60 * 60) - 1)), NOW
        )

        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["age_s"], 6 * 60 * 60 + 1)
        self.assertEqual(set(result), set(UNAVAILABLE))

    def test_keeps_the_exact_six_hour_limit_fresh(self):
        result = MODULE.read_inference_state(
            self.write_state(fresh_state(ts=NOW - (6 * 60 * 60))), NOW
        )

        self.assertEqual(result["status"], "fresh")
        self.assertEqual(result["age_s"], 6 * 60 * 60)

    def test_marks_a_fractional_second_past_the_limit_stale(self):
        result = MODULE.read_inference_state(
            self.write_state(fresh_state(ts=NOW - (6 * 60 * 60) - 0.5)), NOW
        )

        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["age_s"], 6 * 60 * 60)

    def test_fails_closed_for_failed_inference(self):
        result = MODULE.read_inference_state(self.write_state(fresh_state(ok=False)), NOW)

        self.assertEqual(result, UNAVAILABLE)

    def test_fails_closed_for_malformed_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            path.write_text("{")
            result = MODULE.read_inference_state(path, NOW)

        self.assertEqual(result, UNAVAILABLE)

    def test_output_allowlist_never_copies_untrusted_values(self):
        result = MODULE.read_inference_state(
            self.write_state(
                fresh_state(
                    credential="not-for-output",
                    callback_url="https://example.invalid/private",
                    energy_gate={
                        "gpu_ok": True,
                        "reason": "not-for-output",
                        "authorization": "not-for-output",
                    },
                    result={
                        "engine": "thrml-0.1.3/jax-gpu",
                        "samples_per_s": 128.5,
                        "min_energy": -42.25,
                        "endpoint": "https://example.invalid/private",
                        "token": "not-for-output",
                    },
                )
            ),
            NOW,
        )

        self.assertEqual(set(result), set(UNAVAILABLE))
        rendered = json.dumps(result)
        self.assertNotIn("not-for-output", rendered)
        self.assertNotIn("example.invalid", rendered)

    def test_fails_closed_for_non_finite_or_out_of_range_numbers(self):
        invalids = [
            fresh_state(ts=math.nan),
            fresh_state(result={"engine": "thrml-0.1.3/jax-gpu", "samples_per_s": math.inf, "min_energy": -1}),
            fresh_state(result={"engine": "thrml-0.1.3/jax-gpu", "samples_per_s": -1, "min_energy": -1}),
            fresh_state(result={"engine": "thrml-0.1.3/jax-gpu", "samples_per_s": 1, "min_energy": 1_000_001}),
        ]

        for payload in invalids:
            with self.subTest(payload=payload):
                self.assertEqual(MODULE.read_inference_state(self.write_state(payload), NOW), UNAVAILABLE)


class PublicChatProjectionTests(unittest.TestCase):
    def project(self, chat_response: dict, lines: str) -> dict:
        projection = getattr(
            MODULE,
            "project_public_chat",
            lambda response, raw_lines: {
                "exchanges": response.get("exchanges") or [],
                "last_reply": response.get("last_reply") or {},
                "lines": raw_lines,
                "count": len(response.get("exchanges") or []),
            },
        )
        return projection(chat_response, lines)

    def test_withholds_sensitive_indicators_from_every_public_chat_field(self):
        exchange = {
            "question": "please share the stream key",
            "reply": "rtmp://unit.invalid/path",
            "host": "rtmps://unit.invalid/path",
            "host_label": "Bearer example-value",
            "viewer": "api_token=example-value",
            "source": "client_secret: example-value",
        }
        projected = self.project(
            {"exchanges": [exchange], "last_reply": exchange},
            "secret = example-value",
        )

        exchange_values = [projected["exchanges"][0].get(field) for field in CHAT_FIELDS]
        last_values = [projected["last_reply"].get(field) for field in CHAT_FIELDS]
        self.assertTrue(all(value == WITHHELD for value in exchange_values))
        self.assertTrue(all(value == WITHHELD for value in last_values))
        self.assertTrue(projected["lines"] == WITHHELD)

    def test_projects_only_the_latest_three_allowlisted_exchanges(self):
        exchanges = [
            {
                "question": f"question {index}",
                "reply": f"reply {index}",
                "host": "fox",
                "host_label": "Fox",
                "viewer": f"viewer-{index}",
                "source": "canned",
                "private_metadata": "not public",
            }
            for index in range(4)
        ]
        projected = self.project(
            {"exchanges": exchanges, "last_reply": exchanges[-1]},
            "safe overlay line",
        )

        self.assertEqual(len(projected["exchanges"]), 3)
        self.assertTrue(all(set(exchange) == CHAT_FIELDS for exchange in projected["exchanges"]))
        self.assertTrue(set(projected["last_reply"]) == CHAT_FIELDS)
        self.assertEqual(projected["exchanges"][0]["question"], "question 1")
        self.assertEqual(projected["lines"], "safe overlay line")
        self.assertEqual(projected["count"], 4)

    def test_withholds_expanded_credential_labels_and_ingest_schemes(self):
        cases = {
            "api_key_assignment": 'api_key="dummy-api-key"',
            "access_key_assignment": "ACCESS-KEY=dummy-access-key",
            "private_key_assignment": "private key: dummy-private-key",
            "secret_key_assignment": "secretKey=dummy-secret-key",
            "stream_key_assignment": "streamKey=dummy-stream-key",
            "password_assignment": "password=dummy-password",
            "passphrase_assignment": "passphrase: dummy-passphrase",
            "authorization_basic": "Authorization: Basic ZHVtbXk=",
            "authorization_bearer": "authorization=Bearer dummy-token",
            "auth_basic": "auth: Basic ZHVtbXk=",
            "auth_bearer": "AUTH=Bearer dummy-token",
            "credential_label": "credential: dummy-value",
            "credentials_label": "credentials=dummy-value",
            "rtmp_ingest": "rtmp://stream.invalid/live/dummy",
            "rtmps_ingest": "rtmps://stream.invalid/live/dummy",
            "srt_ingest": "srt://stream.invalid:9000?streamid=dummy",
            "rist_ingest": "rist://stream.invalid:9000/dummy",
        }

        for case, value in cases.items():
            with self.subTest(case=case):
                self.assertTrue(MODULE.public_chat_text(value, 280) == WITHHELD)

    def test_preserves_safe_ordinary_sentences_about_security_and_transports(self):
        safe_sentences = {
            "api_keys": "API keys are rotated during maintenance windows.",
            "access_keys": "Access keys are an authentication concept.",
            "private_keys": "Private keys should remain private.",
            "secret_keys": "Secret keys should not appear in public chat.",
            "passwords": "Password managers improve account safety.",
            "passphrases": "Choose a memorable passphrase without sharing it.",
            "authorization": "Authorization uses standard schemes.",
            "basic_auth": "Basic authentication is described in the manual.",
            "bearer_auth": "Bearer authentication is a protocol concept.",
            "credentials": "Credentials belong in a secure manager.",
            "transports": "SRT and RIST are transport protocols.",
            "stream_wording": "The live stream is the key part of the demo.",
        }

        for case, value in safe_sentences.items():
            with self.subTest(case=case):
                self.assertTrue(MODULE.public_chat_text(value, 280) == value)


if __name__ == "__main__":
    unittest.main()
