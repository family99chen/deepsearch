"""
Worker subprocess lifecycle helpers.

Keep worker execution isolated in its own process group so timeout cleanup can
reap the whole browser tree instead of only the Python worker process.
"""

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union


@dataclass
class ManagedSubprocessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _terminate_process_group(process: subprocess.Popen, grace_period: float = 5.0):
    """Terminate the worker process group and escalate if needed."""
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=grace_period)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def run_worker_subprocess(
    cmd: Sequence[str],
    cwd: Union[str, Path],
    timeout: int,
) -> ManagedSubprocessResult:
    """
    Run a worker in its own session so timeout cleanup removes child browsers.
    """
    process = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
        start_new_session=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return ManagedSubprocessResult(
            returncode=process.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
        )
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        stdout, stderr = process.communicate()
        return ManagedSubprocessResult(
            returncode=process.returncode if process.returncode is not None else -signal.SIGKILL,
            stdout=stdout or exc.stdout or "",
            stderr=stderr or exc.stderr or "",
            timed_out=True,
        )
    except Exception:
        _terminate_process_group(process)
        raise
