#!/home/diamondnode/venv312/bin/python
"""Focused tests for the credential-blind SOTA stream watchdog."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "sota-stream-watchdog.py"
SPEC = importlib.util.spec_from_file_location("sota_stream_watchdog", MODULE_PATH)
watchdog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watchdog
SPEC.loader.exec_module(watchdog)


class FakeSystemctlSource:
    def __init__(self, pid: int, active: bool = True) -> None:
        self.pid = pid
        self.active = active
        self.control_group = (
            "/user.slice/user-1000.slice/app.slice/fake-encoder.service"
        )
        self.probed_services: list[str] = []
        self.restarted_services: list[str] = []

    def snapshot(self, service_name: str):
        self.probed_services.append(service_name)
        return watchdog.ServiceSnapshot(
            pid=self.pid,
            active=self.active,
            control_group=self.control_group,
        )

    def restart(self, service_name: str) -> None:
        self.restarted_services.append(service_name)


class FakeSocketProgressSource:
    def __init__(self, counters: dict[str, int] | None) -> None:
        self.counters = counters
        self.probed_control_groups: list[str] = []

    def counters_for_cgroup(self, control_group: str) -> dict[str, int] | None:
        self.probed_control_groups.append(control_group)
        return None if self.counters is None else dict(self.counters)


class WatchdogTestCase(unittest.TestCase):
    service_name = "fake-encoder.service"

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.proc_root = self.root / "proc"
        self.state_path = self.root / "runtime" / "watchdog.json"
        self.systemctl = FakeSystemctlSource(pid=41001)
        self.sockets = FakeSocketProgressSource({"7001": 100})
        self.proc = watchdog.ProcStatusSource(self.proc_root)
        self.store = watchdog.FileStateStore(self.state_path)
        self.now = 1_800_000_000
        self._write_proc_status(41001)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_proc_status(self, pid: int, state: str = "S (sleeping)") -> None:
        proc_dir = self.proc_root / str(pid)
        proc_dir.mkdir(parents=True, exist_ok=True)
        (proc_dir / "status").write_text(
            f"Name:\tfake-encoder\nState:\t{state}\nPid:\t{pid}\n",
            encoding="utf-8",
        )

    def _probe(self, *, restart_on_stall: bool = True) -> int:
        self.now += 30
        return watchdog.run_watchdog(
            service_name=self.service_name,
            state_store=self.store,
            service_source=self.systemctl,
            proc_source=self.proc,
            socket_source=self.sockets,
            clock=lambda: self.now,
            restart_on_stall=restart_on_stall,
        )

    def _state_json(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def test_healthy_progress_resets_failure_count(self) -> None:
        self.assertEqual(self._probe(), 0)
        self.sockets.counters = {"7001": 125}

        self.assertEqual(self._probe(), 0)

        self.assertEqual(self._state_json()["consecutive_no_progress"], 0)
        self.assertEqual(self._state_json()["status"], "healthy")
        self.assertEqual(self.systemctl.restarted_services, [])

    def test_one_missed_sample_stays_successful(self) -> None:
        self.assertEqual(self._probe(), 0)

        self.assertEqual(self._probe(), 0)

        self.assertEqual(self._state_json()["consecutive_no_progress"], 1)
        self.assertEqual(self._state_json()["status"], "waiting")
        self.assertEqual(self.systemctl.restarted_services, [])

    def test_unavailable_socket_measurement_never_counts_as_a_stall(self) -> None:
        self.assertEqual(self._probe(), 0)
        self.sockets.counters = None

        for _ in range(5):
            self.assertEqual(self._probe(), 0)

        state = self._state_json()
        self.assertEqual(state["consecutive_no_progress"], 0)
        self.assertEqual(state["status"], "unavailable")
        self.assertEqual(self.systemctl.restarted_services, [])

    def test_third_consecutive_failure_restarts_fake_service_and_exits_nonzero(self) -> None:
        self.assertEqual(self._probe(), 0)  # establish the first counter baseline
        self.assertEqual(self._probe(), 0)
        self.assertEqual(self._probe(), 0)

        self.assertEqual(self._probe(), 1)

        self.assertEqual(self._state_json()["consecutive_no_progress"], 3)
        self.assertEqual(self._state_json()["status"], "stalled")
        self.assertEqual(self.systemctl.restarted_services, [self.service_name])

        self.assertEqual(self._probe(), 1)
        self.assertEqual(self.systemctl.restarted_services, [self.service_name])

    def test_restart_action_is_disabled_without_explicit_flag(self) -> None:
        self.assertEqual(self._probe(restart_on_stall=False), 0)
        self.assertEqual(self._probe(restart_on_stall=False), 0)
        self.assertEqual(self._probe(restart_on_stall=False), 0)

        self.assertEqual(self._probe(restart_on_stall=False), 1)
        self.assertEqual(self.systemctl.restarted_services, [])

    def test_pid_change_resets_failures_and_establishes_new_baseline(self) -> None:
        self.assertEqual(self._probe(), 0)
        self.assertEqual(self._probe(), 0)
        self.assertEqual(self._probe(), 0)
        self.systemctl.pid = 41002
        self._write_proc_status(41002)
        self.sockets.counters = {"8002": 1}

        self.assertEqual(self._probe(), 0)

        state = self._state_json()
        self.assertEqual(state["pid"], 41002)
        self.assertEqual(state["consecutive_no_progress"], 0)
        self.assertEqual(state["status"], "baseline")
        self.assertEqual(self.systemctl.restarted_services, [])

    def test_state_output_has_an_exact_allowlist_and_discards_unknown_input(self) -> None:
        self.state_path.parent.mkdir(parents=True)
        forbidden_value = "rtmp://user:credential@remote.example/live/private"
        self.state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "sampled_at_unix": 1,
                    "pid": 41001,
                    "active": True,
                    "socket_counters": {"7001": 100},
                    "consecutive_no_progress": 2,
                    "status": "waiting",
                    "remote_address": forbidden_value,
                }
            ),
            encoding="utf-8",
        )

        self.assertEqual(self._probe(), 0)

        raw_state = self.state_path.read_text(encoding="utf-8")
        state = json.loads(raw_state)
        self.assertEqual(set(state), watchdog.STATE_KEYS)
        self.assertNotIn(forbidden_value, raw_state)
        self.assertNotIn("remote_address", raw_state)
        self.assertEqual(self.state_path.stat().st_mode & 0o777, 0o600)

    def test_proc_probe_uses_status_and_rejects_pid_mismatch_or_zombie(self) -> None:
        self.assertTrue(self.proc.is_alive(41001))
        self._write_proc_status(41001, state="Z (zombie)")
        self.assertFalse(self.proc.is_alive(41001))
        (self.proc_root / "41001" / "status").write_text(
            "State:\tS (sleeping)\nPid:\t99999\n", encoding="utf-8"
        )
        self.assertFalse(self.proc.is_alive(41001))

    def test_cgroup_parser_requires_an_exact_service_cgroup(self) -> None:
        target = "/user.slice/user-1000.slice/app.slice/fake-encoder.service"
        raw = (
            "ESTAB local remote "
            f"cgroup:{target} ino:7001 bytes_sent:321 sealed=must-not-escape\n"
            "ESTAB local remote "
            f"cgroup:{target}-shadow ino:8001 bytes_sent:654\n"
        )

        counters = watchdog.parse_ss_cgroup_output(raw, target)

        self.assertEqual(counters, {"7001": 321})
        self.assertNotIn("must-not-escape", repr(counters))

    def test_socket_probe_uses_cgroup_metadata_without_pid_ownership(self) -> None:
        target = "/user.slice/user-1000.slice/app.slice/fake-encoder.service"
        calls: list[tuple[list[str], dict]] = []

        class Result:
            returncode = 0
            stdout = f"ESTAB local remote ino:7001 cgroup:{target} bytes_sent:321\n"

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return Result()

        with patch.object(watchdog.subprocess, "run", side_effect=run):
            counters = watchdog.SocketProgressSource().counters_for_cgroup(target)

        self.assertEqual(counters, {"7001": 321})
        self.assertEqual(calls[0][0], ["ss", "--cgroup", "-tinoeOH"])
        self.assertNotIn("-p", calls[0][0])
        self.assertEqual(calls[0][1]["env"], watchdog.MINIMAL_ENV)

    def test_socket_probe_failure_is_reported_as_unavailable(self) -> None:
        target = "/user.slice/user-1000.slice/app.slice/fake-encoder.service"

        class Result:
            returncode = 1
            stdout = ""

        with patch.object(watchdog.subprocess, "run", return_value=Result()):
            counters = watchdog.SocketProgressSource().counters_for_cgroup(target)

        self.assertIsNone(counters)

    def test_systemctl_show_command_requests_only_safe_properties(self) -> None:
        command = watchdog.systemctl_show_command(self.service_name)

        self.assertEqual(
            command,
            [
                "systemctl",
                "--user",
                "show",
                "--no-pager",
                "--property=MainPID",
                "--property=ActiveState",
                "--property=ControlGroup",
                self.service_name,
            ],
        )
        with self.assertRaises(ValueError):
            watchdog.systemctl_show_command("../../unsafe.service")

    def test_systemctl_user_manager_commands_use_fixed_runtime_environment(self) -> None:
        calls: list[dict] = []

        class Result:
            returncode = 0
            stdout = (
                "MainPID=41001\n"
                "ActiveState=active\n"
                "ControlGroup=/user.slice/user-1000.slice/app.slice/"
                "fake-encoder.service\n"
            )

        def run(*args, **kwargs):
            calls.append(kwargs)
            return Result()

        with (
            patch.object(watchdog.os, "getuid", return_value=1000),
            patch.object(watchdog.subprocess, "run", side_effect=run),
        ):
            source = watchdog.SystemctlSource()
            self.assertEqual(
                source.snapshot(self.service_name),
                watchdog.ServiceSnapshot(
                    pid=41001,
                    active=True,
                    control_group=(
                        "/user.slice/user-1000.slice/app.slice/"
                        "fake-encoder.service"
                    ),
                ),
            )
            source.restart(self.service_name)

        expected_env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "XDG_RUNTIME_DIR": "/run/user/1000",
        }
        self.assertEqual([call["env"] for call in calls], [expected_env, expected_env])


class UnitFileTestCase(unittest.TestCase):
    unit_dir = Path("/home/diamondnode/.config/systemd/user")

    def test_service_runs_credential_blind_watchdog_with_guarded_restart(self) -> None:
        service = (self.unit_dir / "sota-stream-watchdog.service").read_text(
            encoding="utf-8"
        )
        self.assertIn("Type=oneshot", service)
        self.assertIn("--service sota-webgpu-stream.service", service)
        self.assertIn("--restart-on-stall", service)
        self.assertNotIn("ExecStartPost", service)
        self.assertNotIn("EnvironmentFile", service)
        for hardening in (
            "NoNewPrivileges=yes",
            "PrivateDevices=yes",
            "ProtectSystem=full",
            "ProtectHome=read-only",
            "RestrictAddressFamilies=AF_UNIX AF_NETLINK",
            "LockPersonality=yes",
            "MemoryDenyWriteExecute=yes",
        ):
            self.assertIn(hardening, service)

    def test_timer_runs_every_thirty_seconds(self) -> None:
        timer = (self.unit_dir / "sota-stream-watchdog.timer").read_text(
            encoding="utf-8"
        )
        self.assertIn("OnBootSec=30s", timer)
        self.assertIn("OnUnitActiveSec=30s", timer)
        self.assertIn("Unit=sota-stream-watchdog.service", timer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
