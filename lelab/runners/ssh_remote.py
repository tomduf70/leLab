# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SSH remote runner — runs a training on `robotic-ai` (RTX 5060 Ti PC) over SSH.

Modeled on HfCloudJobRunner: a detached out-of-process runner monitored by two
worker threads (a log tail + a status poller). The training itself runs detached
on the remote host (`setsid nohup`), so it survives an SSH drop or an Orin
restart; we reconnect to tail its log and poll its liveness.

Sync is entirely Hub-based, exactly like the cloud runner: the dataset is pulled
by the remote trainer from `--dataset.repo_id`, and checkpoints are pushed back
to the Hub (`--policy.push_to_hub`) so the LeLab UI lists them the same way it
lists cloud-job checkpoints. SSH only launches and monitors the process.

MVP scope: the remote host, project dir and HF user are hard-coded constants
below. Making them configurable (env / settings) is a follow-up.
"""

from __future__ import annotations

import contextlib
import logging
import shlex
import subprocess
import threading
import time
from base64 import b64encode
from pathlib import Path
from queue import Empty, Queue

from ..jobs import LogLine, TrainingMetrics, extract_wandb_run_url, parse_metrics_into
from ..train import TrainingRequest, build_training_command
from .hf_cloud import WRAPPER_SOURCE

logger = logging.getLogger(__name__)

# --- Remote layout (hard-coded for the MVP) ---------------------------------
# `robotic-ai` is the Host alias in the Orin's ~/.ssh/config (cloudflared
# ProxyCommand, user tomduf). $HOME there is /home/tomduf; we use absolute
# paths everywhere so nothing depends on remote shell tilde/$HOME expansion
# (the trainer argv is shlex-quoted and would not expand $HOME anyway).
SSH_HOST = "robotic-ai"
REMOTE_HOME = "/home/tomduf"
REMOTE_PROJECT_DIR = f"{REMOTE_HOME}/projects/lerobot-test"
# Direct venv console script — same invocation the user's dashboard used. More
# robust than `uv run` (which isn't on the non-interactive SSH PATH).
REMOTE_TRAIN_BIN = f"{REMOTE_PROJECT_DIR}/.venv/bin/lerobot-train"
# venv interpreter, used to run the checkpoint-uploader wrapper (needs the
# venv's huggingface_hub). The remote host's HF token (~/.cache/huggingface)
# authenticates the uploads.
REMOTE_PYTHON = f"{REMOTE_PROJECT_DIR}/.venv/bin/python"
REMOTE_JOBS_ROOT = f"{REMOTE_HOME}/.cache/lelab/jobs"
# HF account the remote host pushes checkpoints to; the LeLab UI lists them
# from f"{REMOTE_HF_USER}/{job_id}".
REMOTE_HF_USER = "tomduf70"

# SSH option sets. BatchMode prevents any interactive prompt from hanging a
# worker thread. The tail connection adds keepalives so a dead link is noticed
# (and the loop reconnects) rather than blocking forever on a half-open socket.
_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
_SSH_TAIL_OPTS = _SSH_OPTS + ["-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"]

# Cadence of the liveness poller.
_STATUS_POLL_INTERVAL_S = 5.0
# Backoff before the tail loop reconnects after the stream ends or errors.
_TAIL_RECONNECT_BACKOFF_S = 5.0
# How long the launcher may sit in the STARTING phase (pidfile not yet written)
# before we treat it as a failed launch. The bootstrap already confirmed the
# detached process forked, so this only guards against it dying instantly.
_STARTING_GRACE_S = 30.0


def _remote_job_dir(job_id: str) -> str:
    return f"{REMOTE_JOBS_ROOT}/{job_id}"


class SshJobRunner:
    """Run a training on the remote GPU host over SSH. Single-shot — instantiate
    per job. Satisfies the JobRunner Protocol declared in lelab/jobs.py."""

    def __init__(
        self,
        metrics: TrainingMetrics,
        log_file_path: Path,
    ) -> None:
        self._metrics = metrics
        self._log_file_path = log_file_path
        self._job_id: str | None = None
        self._remote_dir: str | None = None
        self._remote_log: str | None = None
        self._remote_rc: str | None = None
        self._remote_pid: str | None = None

        self._log_queue: Queue[LogLine] = Queue()
        self._tail_thread: threading.Thread | None = None
        self._status_thread: threading.Thread | None = None
        self._tail_proc: subprocess.Popen | None = None
        self._stop_event = threading.Event()
        self._log_file = None  # type: ignore[assignment]

        # Set once the run reaches a terminal state. None while live.
        self._terminal_returncode: int | None = None
        self._terminal_message: str | None = None
        self._wandb_run_url: str | None = None
        self._lines_processed: int = 0
        self._started_at: float = 0.0

    # -- lifecycle -----------------------------------------------------------

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None:
        # output_dir is the host-local path the registry pins; it doesn't exist
        # on the remote host, so we ignore it and write to a remote-local path
        # (checkpoints reach the UI via the Hub, never that path directly).
        del output_dir
        if self._job_id is not None:
            raise RuntimeError("SshJobRunner already started")
        self._init_paths(job_id)

        # Hub-based sync: force a push so the UI can browse checkpoints. Mutating
        # config here means the persisted JobRecord.config reflects what ran.
        config.policy_push_to_hub = True
        config.policy_repo_id = f"{REMOTE_HF_USER}/{job_id}"

        # lerobot's EvalConfig (>=0.5) rejects eval.batch_size > eval.n_episodes
        # at parse time, even when eval is disabled (eval_freq=0). The form's
        # defaults are 50 > 10, which would abort the remote trainer before it
        # starts. Clamp — harmless since we never evaluate here.
        if config.eval_batch_size > config.eval_n_episodes:
            config.eval_batch_size = config.eval_n_episodes

        # Reuse build_training_command for the flag list, then swap its
        # `python -m lerobot.scripts.lerobot_train` prefix for the remote venv
        # console script (stable across the remote's lerobot version).
        trainer_argv = [REMOTE_TRAIN_BIN, *build_training_command(config, f"{self._remote_dir}/run")[3:]]

        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_file_path.open("a", buffering=1)

        logger.info("Launching SSH job %s on %s: %s", job_id, SSH_HOST, " ".join(trainer_argv))
        self._launch_remote(trainer_argv)
        self._started_at = time.time()
        self._start_worker_threads(job_id)

    def reattach(self, job_id: str) -> None:
        """Take over an already-running remote job after an Orin restart.

        All remote paths derive from job_id, so there's nothing to persist —
        we just re-open the log file and restart the worker threads."""
        if self._job_id is not None:
            raise RuntimeError("SshJobRunner already started")
        self._init_paths(job_id)
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_file_path.open("a", buffering=1)
        self._started_at = time.time()
        self._start_worker_threads(f"{job_id}-reattach")

    def _init_paths(self, job_id: str) -> None:
        self._job_id = job_id
        self._remote_dir = _remote_job_dir(job_id)
        self._remote_log = f"{self._remote_dir}/log"
        self._remote_rc = f"{self._remote_dir}/rc"
        self._remote_pid = f"{self._remote_dir}/pid"

    def _launch_remote(self, trainer_argv: list[str]) -> None:
        """Write a launcher script to the remote host and start it detached.

        The launcher records its own PID (for liveness checks) and the trainer's
        exit code (for the final status). The trainer runs under the inlined
        checkpoint-uploader wrapper (shared with the HF cloud runner): the
        wrapper spawns the trainer and, while it runs, uploads each new
        <output_dir>/checkpoints/<step>/ to the Hub model repo so the LeLab UI
        can browse them live — same `checkpoints/<step>/pretrained_model/`
        layout the cloud runner produces.

        Both files (launcher + wrapper) ship as base64 over SSH stdin, so no
        argument or Python source ever has to survive a layer of shell quoting.
        """
        assert self._remote_dir and self._remote_log and self._remote_rc and self._remote_pid
        wrapper_path = f"{self._remote_dir}/wrapper.py"
        script_path = f"{self._remote_dir}/launch.sh"
        # `python wrapper.py -- <trainer argv>`: the wrapper forwards everything
        # after `--` to the trainer (see WRAPPER_SOURCE).
        run_line = (
            f"{shlex.quote(REMOTE_PYTHON)} {shlex.quote(wrapper_path)} -- "
            f"{' '.join(shlex.quote(a) for a in trainer_argv)}"
        )
        launch_sh = (
            "#!/bin/bash\n"
            f"echo $$ > {shlex.quote(self._remote_pid)}\n"
            f"cd {shlex.quote(REMOTE_PROJECT_DIR)}\n"
            f"{run_line}\n"
            f"echo $? > {shlex.quote(self._remote_rc)}\n"
        )
        launch_b64 = b64encode(launch_sh.encode()).decode()
        wrapper_b64 = b64encode(WRAPPER_SOURCE.encode()).decode()
        bootstrap = "\n".join(
            [
                "set -e",
                f"mkdir -p {shlex.quote(self._remote_dir)}",
                f"printf %s {shlex.quote(wrapper_b64)} | base64 -d > {shlex.quote(wrapper_path)}",
                f"printf %s {shlex.quote(launch_b64)} | base64 -d > {shlex.quote(script_path)}",
                # Detached: setsid makes it a new session leader (so we can later
                # signal the whole process group), nohup + full redirect frees the
                # SSH channel so this command returns immediately.
                f"setsid nohup bash {shlex.quote(script_path)} "
                f"> {shlex.quote(self._remote_log)} 2>&1 < /dev/null &",
                "echo STARTED",
            ]
        )
        result = self._run_ssh(["bash", "-s"], input_text=bootstrap, timeout=60)
        if result.returncode != 0 or "STARTED" not in result.stdout:
            raise RuntimeError(
                f"Failed to launch remote job on {SSH_HOST} "
                f"(rc={result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
            )

    def _start_worker_threads(self, label: str) -> None:
        self._tail_thread = threading.Thread(target=self._tail_loop, name=f"ssh-job-{label}-logs", daemon=True)
        self._tail_thread.start()
        self._status_thread = threading.Thread(
            target=self._status_poll_loop, name=f"ssh-job-{label}-status", daemon=True
        )
        self._status_thread.start()

    # -- ssh helpers ---------------------------------------------------------

    def _run_ssh(
        self, remote_argv: list[str], input_text: str | None = None, timeout: float = 30.0
    ) -> subprocess.CompletedProcess:
        """Run a one-shot SSH command and capture its output."""
        cmd = ["ssh", *_SSH_OPTS, SSH_HOST, *remote_argv]
        return subprocess.run(  # noqa: S603 — fixed host alias, no shell
            cmd, input=input_text, capture_output=True, text=True, timeout=timeout
        )

    def _set_terminal(self, returncode: int, message: str | None = None) -> None:
        """Record the run's terminal state. Idempotent. Wakes the workers."""
        if self._terminal_returncode is not None:
            return
        self._terminal_returncode = returncode
        if message:
            self._terminal_message = message
        self._stop_event.set()

    # -- worker loops --------------------------------------------------------

    def _tail_loop(self) -> None:
        """`ssh tail -F` the remote log, teeing each line to disk + the queue.

        Reconnects on stream end/error while the status poller still says the
        job is alive. `tail -n +1 -F` re-reads from the top on reconnect, so we
        skip the already-processed prefix via _lines_processed (same scheme as
        the HF cloud runner). Exits when _stop_event is set.
        """
        assert self._remote_log is not None
        try:
            while not self._stop_event.is_set():
                try:
                    self._tail_proc = subprocess.Popen(  # noqa: S603 — fixed host alias, no shell
                        ["ssh", *_SSH_TAIL_OPTS, SSH_HOST, f"tail -n +1 -F {shlex.quote(self._remote_log)}"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        bufsize=1,
                    )
                    seen = 0
                    assert self._tail_proc.stdout is not None
                    for raw in self._tail_proc.stdout:
                        if self._stop_event.is_set():
                            break
                        seen += 1
                        if seen <= self._lines_processed:
                            continue  # replayed prefix from a reconnect
                        self._lines_processed = seen
                        self._handle_log_line(raw.rstrip())
                except Exception as exc:
                    logger.info("SSH log tail disconnected, will reconnect: %s", exc)
                finally:
                    self._kill_tail_proc()

                if self._stop_event.wait(_TAIL_RECONNECT_BACKOFF_S):
                    return
        finally:
            if self._log_file is not None:
                with contextlib.suppress(Exception):
                    self._log_file.close()
                self._log_file = None

    def _handle_log_line(self, stripped: str) -> None:
        if not stripped:
            return
        parse_metrics_into(stripped, self._metrics)
        if self._wandb_run_url is None:
            url = extract_wandb_run_url(stripped)
            if url is not None:
                self._wandb_run_url = url
        log_line = LogLine(timestamp=time.time(), message=stripped)
        if self._log_file is not None:
            try:
                self._log_file.write(log_line.model_dump_json() + "\n")
            except Exception as exc:  # pragma: no cover
                logger.exception("Error writing SSH log: %s", exc)
        if self._log_queue.qsize() >= 1000:
            with contextlib.suppress(Empty):
                self._log_queue.get_nowait()
        self._log_queue.put(log_line)

    def _kill_tail_proc(self) -> None:
        proc, self._tail_proc = self._tail_proc, None
        if proc is None:
            return
        with contextlib.suppress(Exception):
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)

    def _status_poll_loop(self) -> None:
        """Poll remote liveness until the run reaches a terminal state.

        One SSH round-trip per tick inspects three remote files: `rc` (exit code,
        authoritative once present), `pid` (+ `kill -0` for liveness), else the
        run is still STARTING. Sole writer of _terminal_returncode under normal
        operation (stop() pre-sets it on cancel)."""
        assert self._remote_rc and self._remote_pid
        probe = (
            f"if [ -f {shlex.quote(self._remote_rc)} ]; then echo \"RC:$(cat {shlex.quote(self._remote_rc)})\"; "
            f"elif [ -f {shlex.quote(self._remote_pid)} ] && kill -0 \"$(cat {shlex.quote(self._remote_pid)})\" "
            "2>/dev/null; then echo ALIVE; "
            f"elif [ -f {shlex.quote(self._remote_pid)} ]; then echo CRASHED; "
            "else echo STARTING; fi"
        )
        while not self._stop_event.is_set():
            try:
                result = self._run_ssh([probe], timeout=30)
                out = result.stdout.strip()
                if out.startswith("RC:"):
                    self._set_terminal(self._parse_rc(out[3:]))
                    return
                if out == "CRASHED":
                    self._set_terminal(1, "Remote process died without writing an exit code")
                    return
                if out == "STARTING" and (time.time() - self._started_at) > _STARTING_GRACE_S:
                    self._set_terminal(1, "Remote launcher never started")
                    return
                # ALIVE / STARTING(within grace): keep polling.
            except Exception as exc:
                logger.warning("SSH status poll failed for %s: %s", self._job_id, exc)
            if self._stop_event.wait(_STATUS_POLL_INTERVAL_S):
                return

    @staticmethod
    def _parse_rc(raw: str) -> int:
        try:
            return int(raw.strip())
        except ValueError:
            return 1

    # -- JobRunner Protocol --------------------------------------------------

    def stop(self) -> None:
        if self._job_id is None:
            return
        # Pre-set so finalisation treats this as a clean cancel regardless of
        # what the poller observes next.
        self._set_terminal(1, "Canceled by user")
        if self._remote_pid is None:
            return
        # Signal the whole process group (setsid made the launcher a session
        # leader, so its PID is the PGID): TERM, then KILL after a grace.
        kill = (
            f"P=$(cat {shlex.quote(self._remote_pid)} 2>/dev/null); "
            'if [ -n "$P" ]; then kill -TERM -"$P" 2>/dev/null; '
            'sleep 3; kill -KILL -"$P" 2>/dev/null; fi; true'
        )
        try:
            self._run_ssh([kill], timeout=30)
        except Exception as exc:
            logger.info("Remote kill for %s ignored: %s", self._job_id, exc)

    def is_running(self) -> bool:
        if self._job_id is None:
            return False
        return self._terminal_returncode is None

    def returncode(self) -> int | None:
        return self._terminal_returncode

    def stream_log_lines(self) -> list[LogLine]:
        out: list[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            pass
        return out

    def wandb_run_url(self) -> str | None:
        return self._wandb_run_url

    def terminal_message(self) -> str | None:
        return self._terminal_message
