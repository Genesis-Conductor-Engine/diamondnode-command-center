#!/home/diamondnode/venv312/bin/python
"""Credential-blind progress watchdog for the SOTA WebGPU stream encoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Callable, Mapping, Sequence


STATE_VERSION = 1
STALL_THRESHOLD = 3
DEFAULT_SERVICE = "sota-webgpu-stream.service"
DEFAULT_STATE_PATH = Path("/tmp/sota-livestream/watchdog.json")
STATE_KEYS = {
    "version",
    "sampled_at_unix",
    "pid",
    "active",
    "socket_counters",
    "consecutive_no_progress",
    "status",
}
VALID_STATUSES = {"baseline", "healthy", "waiting", "stalled"}
SERVICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@:-]*\.service$")
PID_PATTERN = re.compile(r"\bpid=(\d+)\b")
INODE_PATTERN = re.compile(r"\bino:(\d+)\b")
BYTES_SENT_PATTERN = re.compile(r"\bbytes_sent:(\d+)\b")
MINIMAL_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}


@dataclass(frozen=True)
class ServiceSnapshot:
    pid: int
    active: bool


@dataclass(frozen=True)
class WatchdogState:
    version: int
    sampled_at_unix: int
    pid: int
    active: bool
    socket_counters: dict[str, int]
    consecutive_no_progress: int
    status: str


@dataclass(frozen=True)
class Evaluation:
    state: WatchdogState
    restart_required: bool
    exit_code: int


def _is_plain_int(value: object) -> bool:
    return type(value) is int


def _valid_counters(value: object) -> bool:
    if not isinstance(value, dict) or len(value) > 4096:
        return False
    return all(
        isinstance(inode, str)
        and inode.isdigit()
        and _is_plain_int(count)
        and count >= 0
        for inode, count in value.items()
    )


def state_from_json(value: object) -> WatchdogState | None:
    """Accept only the exact, non-sensitive state schema."""
    if not isinstance(value, dict) or set(value) != STATE_KEYS:
        return None
    if value.get("version") != STATE_VERSION:
        return None
    if not _is_plain_int(value.get("sampled_at_unix")):
        return None
    if not _is_plain_int(value.get("pid")) or value["pid"] < 0:
        return None
    if type(value.get("active")) is not bool:
        return None
    if not _valid_counters(value.get("socket_counters")):
        return None
    strikes = value.get("consecutive_no_progress")
    if not _is_plain_int(strikes) or not 0 <= strikes <= STALL_THRESHOLD:
        return None
    if value.get("status") not in VALID_STATUSES:
        return None
    return WatchdogState(
        version=STATE_VERSION,
        sampled_at_unix=value["sampled_at_unix"],
        pid=value["pid"],
        active=value["active"],
        socket_counters=dict(value["socket_counters"]),
        consecutive_no_progress=strikes,
        status=value["status"],
    )


class FileStateStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> WatchdogState | None:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                return state_from_json(json.load(handle))
        except (OSError, ValueError, TypeError):
            return None

    def write(self, state: WatchdogState) -> None:
        payload = asdict(state)
        if set(payload) != STATE_KEYS:
            raise ValueError("unsafe watchdog state schema")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".watchdog.", dir=self.path.parent, text=True
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise


class ProcStatusSource:
    """Reads only Pid and State from a configurable proc status tree."""

    def __init__(self, root: Path = Path("/proc")) -> None:
        self.root = Path(root)

    def is_alive(self, pid: int) -> bool:
        if type(pid) is not int or pid <= 0:
            return False
        found_pid: int | None = None
        state_code: str | None = None
        try:
            with (self.root / str(pid) / "status").open(
                "r", encoding="utf-8", errors="replace"
            ) as handle:
                for line in handle:
                    if line.startswith("Pid:"):
                        found_pid = int(line.partition(":")[2].strip())
                    elif line.startswith("State:"):
                        state_code = line.partition(":")[2].strip()[:1]
        except (OSError, ValueError):
            return False
        return found_pid == pid and state_code not in {None, "X", "Z"}


def validate_service_name(service_name: str) -> str:
    if not SERVICE_NAME_PATTERN.fullmatch(service_name):
        raise ValueError("invalid service name")
    return service_name


def systemctl_show_command(service_name: str) -> list[str]:
    service_name = validate_service_name(service_name)
    return [
        "systemctl",
        "--user",
        "show",
        "--no-pager",
        "--property=MainPID",
        "--property=ActiveState",
        service_name,
    ]


def systemctl_user_manager_env() -> dict[str, str]:
    """Build a fixed user-manager environment without copying process variables."""
    return {**MINIMAL_ENV, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"}


def _parse_systemctl_show(output: str) -> ServiceSnapshot:
    safe_values: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"MainPID", "ActiveState"}:
            safe_values[key] = value.strip()
    try:
        pid = int(safe_values.get("MainPID", "0"))
    except ValueError:
        pid = 0
    return ServiceSnapshot(
        pid=max(pid, 0), active=safe_values.get("ActiveState") == "active"
    )


class SystemctlSource:
    """Runs fixed, safe-property systemctl probes and a validated restart."""

    def snapshot(self, service_name: str) -> ServiceSnapshot:
        try:
            result = subprocess.run(
                systemctl_show_command(service_name),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                env=systemctl_user_manager_env(),
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ServiceSnapshot(pid=0, active=False)
        if result.returncode != 0:
            return ServiceSnapshot(pid=0, active=False)
        return _parse_systemctl_show(result.stdout)

    def restart(self, service_name: str) -> None:
        service_name = validate_service_name(service_name)
        try:
            subprocess.run(
                ["systemctl", "--user", "restart", service_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                env=systemctl_user_manager_env(),
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return


def parse_ss_output(output: str, pid: int) -> dict[str, int]:
    """Reduce transient ss output to numeric counters without retaining endpoints."""
    counters: dict[str, int] = {}
    for line in output.splitlines():
        if pid not in {int(match) for match in PID_PATTERN.findall(line)}:
            continue
        inode_match = INODE_PATTERN.search(line)
        bytes_match = BYTES_SENT_PATTERN.search(line)
        if inode_match is None or bytes_match is None:
            continue
        inode = inode_match.group(1)
        bytes_sent = int(bytes_match.group(1))
        counters[inode] = max(counters.get(inode, 0), bytes_sent)
    return counters


class SocketProgressSource:
    """Internally reduces ss TCP_INFO data to inode/bytes_sent integers."""

    def counters_for_pid(self, pid: int) -> dict[str, int]:
        if type(pid) is not int or pid <= 0:
            return {}
        try:
            result = subprocess.run(
                ["ss", "-tinpeOH"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                env=MINIMAL_ENV,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        if result.returncode != 0:
            return {}
        return parse_ss_output(result.stdout, pid)


def _has_progress(
    previous: Mapping[str, int], current: Mapping[str, int]
) -> bool:
    return any(
        count > 0 if inode not in previous else count > previous[inode]
        for inode, count in current.items()
    )


def evaluate_state(
    previous: WatchdogState | None,
    snapshot: ServiceSnapshot,
    process_alive: bool,
    socket_counters: Mapping[str, int],
    sampled_at_unix: int,
) -> Evaluation:
    """Pure state transition: require three same-PID no-progress samples."""
    usable = snapshot.active and snapshot.pid > 0 and process_alive
    counters = dict(socket_counters) if usable else {}
    previous_strikes = previous.consecutive_no_progress if previous else 0

    if previous is None:
        strikes = 0 if usable else 1
        status = "baseline" if usable else "waiting"
    elif usable and previous.pid != snapshot.pid:
        strikes = 0
        status = "baseline"
    elif usable and _has_progress(previous.socket_counters, counters):
        strikes = 0
        status = "healthy"
    else:
        strikes = min(previous_strikes + 1, STALL_THRESHOLD)
        status = "stalled" if strikes == STALL_THRESHOLD else "waiting"

    state = WatchdogState(
        version=STATE_VERSION,
        sampled_at_unix=int(sampled_at_unix),
        pid=snapshot.pid if snapshot.pid > 0 else 0,
        active=usable,
        socket_counters=counters,
        consecutive_no_progress=strikes,
        status=status,
    )
    return Evaluation(
        state=state,
        restart_required=(
            strikes == STALL_THRESHOLD and previous_strikes < STALL_THRESHOLD
        ),
        exit_code=1 if strikes == STALL_THRESHOLD else 0,
    )


def run_watchdog(
    *,
    service_name: str,
    state_store: FileStateStore,
    service_source,
    proc_source,
    socket_source,
    clock: Callable[[], float] = time.time,
    restart_on_stall: bool = False,
) -> int:
    """Run one probe. Dependencies are injectable so production is never needed in tests."""
    service_name = validate_service_name(service_name)
    previous = state_store.load()
    snapshot = service_source.snapshot(service_name)
    process_alive = proc_source.is_alive(snapshot.pid)
    counters = (
        socket_source.counters_for_pid(snapshot.pid) if process_alive else {}
    )
    evaluation = evaluate_state(
        previous,
        snapshot,
        process_alive,
        counters,
        int(clock()),
    )
    state_store.write(evaluation.state)
    if evaluation.restart_required and restart_on_stall:
        service_source.restart(service_name)
    return evaluation.exit_code


@dataclass(frozen=True)
class CliOptions:
    service_name: str = DEFAULT_SERVICE
    state_path: Path = DEFAULT_STATE_PATH
    restart_on_stall: bool = False


def parse_cli(argv: Sequence[str]) -> CliOptions | None:
    service_name = DEFAULT_SERVICE
    state_path = DEFAULT_STATE_PATH
    restart_on_stall = False
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--restart-on-stall":
            restart_on_stall = True
            index += 1
        elif argument in {"--service", "--state-file"} and index + 1 < len(argv):
            value = argv[index + 1]
            if argument == "--service":
                service_name = value
            else:
                state_path = Path(value)
            index += 2
        else:
            return None
    try:
        validate_service_name(service_name)
    except ValueError:
        return None
    return CliOptions(service_name, state_path, restart_on_stall)


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_cli(list(sys.argv[1:] if argv is None else argv))
    if options is None:
        return 0
    try:
        return run_watchdog(
            service_name=options.service_name,
            state_store=FileStateStore(options.state_path),
            service_source=SystemctlSource(),
            proc_source=ProcStatusSource(),
            socket_source=SocketProgressSource(),
            restart_on_stall=options.restart_on_stall,
        )
    except Exception:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
