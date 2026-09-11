"""
Human-readable progress reporting for VoxShield training and evaluation.

Design goals:

    1. At any moment you can see how fast the run is going and when it ends.
    2. Scrollback stays useful - the live line is transient, but a permanent
       heartbeat line is printed every ``log_every`` steps.
    3. Everything printed is also appended to a JSONL file so a run can be
       plotted or compared later without re-running it.
    4. No hard dependency on a TTY. When stdout is redirected to a file the
       live line is suppressed and only the heartbeats survive.

Nothing here touches the model or the optimizer; it is pure instrumentation.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

try:  # torch is optional so this module stays importable by plain tooling
    import torch
except Exception:  # pragma: no cover - torch is a hard dep of training itself
    torch = None


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def format_duration(seconds: float | None) -> str:
    """Render a duration the way a human reads a clock, not as 8213.4s."""

    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--"

    # Keep a decimal below 10 s, otherwise a fast phase reads as a flat "0s"
    # and you cannot tell 40 ms from 900 ms.
    if seconds < 10:
        return f"{seconds:.1f}s"

    seconds = int(round(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"

    if minutes:
        return f"{minutes}m{secs:02d}s"

    return f"{secs}s"


def format_count(value: int) -> str:
    """1234567 -> 1.23M"""

    if value < 1_000:
        return str(value)

    if value < 1_000_000:
        return f"{value / 1_000:.1f}K"

    if value < 1_000_000_000:
        return f"{value / 1_000_000:.2f}M"

    return f"{value / 1_000_000_000:.2f}B"


def _terminal_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default


def _is_tty() -> bool:
    if os.environ.get("VOXSHIELD_NO_LIVE_PROGRESS"):
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def gpu_memory_summary() -> str:
    """Peak allocated / reserved VRAM this epoch, or empty string on CPU."""

    if torch is None or not torch.cuda.is_available():
        return ""

    allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

    return f"vram {allocated:.2f}/{reserved:.2f}G"


# ---------------------------------------------------------------------------
# running statistics
# ---------------------------------------------------------------------------


class RunningMean:
    """Streaming mean that never stores the whole history."""

    __slots__ = ("total", "count")

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0.0

    def update(self, value: float, weight: float = 1.0) -> None:
        if value is None or not math.isfinite(float(value)):
            return
        self.total += float(value) * weight
        self.count += weight

    @property
    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0.0


class RateTracker:
    """
    Two throughput estimates:

        ``recent_rate``  - windowed over the last N steps. Reacts quickly and
                           is what ETA should use once the run has warmed up.
        ``overall_rate`` - whole-run average. The honest number to quote
                           afterwards, because it includes stalls.
    """

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self.timestamps: deque = deque(maxlen=window + 1)
        self.start = time.perf_counter()
        self.steps = 0

    def tick(self) -> None:
        self.steps += 1
        self.timestamps.append(time.perf_counter())

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    @property
    def overall_rate(self) -> float:
        elapsed = self.elapsed
        if elapsed <= 0 or self.steps == 0:
            return 0.0
        return self.steps / elapsed

    @property
    def recent_rate(self) -> float:
        if len(self.timestamps) < 2:
            return self.overall_rate
        span = self.timestamps[-1] - self.timestamps[0]
        if span <= 0:
            return self.overall_rate
        return (len(self.timestamps) - 1) / span

    @property
    def seconds_per_step(self) -> float:
        rate = self.recent_rate
        return 1.0 / rate if rate > 0 else 0.0


# ---------------------------------------------------------------------------
# the reporter
# ---------------------------------------------------------------------------


@dataclass
class EpochResult:
    epoch: int
    loss: float
    accuracy: float
    seconds: float
    steps: int
    samples: int
    extra: dict = field(default_factory=dict)


class TrainingProgress:
    """
    Progress reporter for a multi-epoch training run.

    Usage::

        progress = TrainingProgress(
            total_epochs=5,
            steps_per_epoch=len(loader),
            batch_size=8,
            run_dir=Path("experiments/exp001"),
        )
        progress.run_header({"model": "wav2vec2-base"})

        for epoch in range(1, 6):
            progress.start_epoch(epoch)
            for batch in loader:
                ...
                progress.step(loss=..., accuracy=..., lr=...)
            progress.end_epoch(val_eer=0.031)

        progress.run_footer(checkpoint="models/best.pt")
    """

    def __init__(
        self,
        total_epochs: int,
        steps_per_epoch: int,
        batch_size: int,
        run_dir: Path | str | None = None,
        log_every: int = 25,
        rate_window: int = 50,
        stream=None,
    ) -> None:
        self.total_epochs = max(int(total_epochs), 1)
        self.steps_per_epoch = max(int(steps_per_epoch), 1)
        self.batch_size = batch_size
        self.log_every = max(int(log_every), 1)
        self.rate_window = rate_window

        self.stream = stream or sys.stdout
        self.live = _is_tty()

        self.run_start = time.perf_counter()
        self.epoch = 0
        self.epoch_rate: RateTracker | None = None
        self.epoch_loss = RunningMean()
        self.epoch_accuracy = RunningMean()
        self.epoch_start = 0.0
        self.epoch_step = 0
        self.global_step = 0
        self.last_lr = 0.0
        self.history: list[EpochResult] = []
        self._live_line_length = 0

        self.run_dir = Path(run_dir) if run_dir else None
        self.jsonl_path: Path | None = None

        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.jsonl_path = self.run_dir / "progress.jsonl"
            # Truncate so re-running the same experiment id starts clean.
            self.jsonl_path.write_text("", encoding="utf-8")

    # -- output plumbing ---------------------------------------------------

    def _write_permanent(self, text: str) -> None:
        self._clear_live()
        self.stream.write(text + "\n")
        self.stream.flush()

    def _write_live(self, text: str) -> None:
        if not self.live:
            return
        width = _terminal_width()
        if len(text) > width - 1:
            text = text[: width - 2] + "~"
        padding = max(self._live_line_length - len(text), 0)
        self.stream.write("\r" + text + " " * padding)
        self.stream.flush()
        self._live_line_length = len(text)

    def _clear_live(self) -> None:
        if self.live and self._live_line_length:
            self.stream.write("\r" + " " * self._live_line_length + "\r")
            self.stream.flush()
            self._live_line_length = 0

    def _record(self, payload: dict) -> None:
        if self.jsonl_path is None:
            return
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def note(self, message: str) -> None:
        """
        Print a permanent line without disturbing the live progress line.

        Use this for anything the run wants to announce mid-epoch - a stage
        change, a new best checkpoint, an early-stopping decision.
        """

        self._write_permanent("  " + message)
        self._record({"event": "note", "epoch": self.epoch, "message": message})

    # -- headers -----------------------------------------------------------

    def run_header(self, config: dict) -> None:
        """Print the run configuration as an aligned block."""

        self._write_permanent("")
        self._write_permanent("=" * 78)
        self._write_permanent("  VoxShield anti-spoof training")
        self._write_permanent("=" * 78)

        width = max((len(str(k)) for k in config), default=0)
        width = max(width, len("steps / epoch"))

        for key, value in config.items():
            self._write_permanent(f"  {str(key).ljust(width)}  {value}")

        total_steps = self.steps_per_epoch * self.total_epochs

        self._write_permanent("-" * 78)
        self._write_permanent(f"  {'steps / epoch'.ljust(width)}  {self.steps_per_epoch}")
        self._write_permanent(f"  {'total steps'.ljust(width)}  {total_steps}")
        self._write_permanent("=" * 78)
        self._write_permanent("")

        self._record(
            {
                "event": "run_start",
                "config": {str(k): str(v) for k, v in config.items()},
                "steps_per_epoch": self.steps_per_epoch,
                "total_epochs": self.total_epochs,
            }
        )

    def estimate(self, seconds_per_step: float, note: str = "") -> dict:
        """
        Print a projected runtime from a measured per-step cost.

        Call this after a short warm-up probe so the run can be sized
        *before* committing hours to it.
        """

        per_epoch = seconds_per_step * self.steps_per_epoch
        total = per_epoch * self.total_epochs

        self._write_permanent("  ---- projected runtime " + "-" * 40)
        if note:
            self._write_permanent(f"  {note}")
        self._write_permanent(
            f"  measured     {seconds_per_step * 1000:8.0f} ms/step"
            f"   ({self.batch_size / max(seconds_per_step, 1e-9):.1f} samples/s)"
        )
        self._write_permanent(
            f"  per epoch    {format_duration(per_epoch):>8}"
            f"   ({self.steps_per_epoch} steps)"
        )
        self._write_permanent(
            f"  whole run    {format_duration(total):>8}"
            f"   ({self.total_epochs} epochs)"
        )
        memory = gpu_memory_summary()
        if memory:
            self._write_permanent(f"  peak {memory}")
        self._write_permanent("  " + "-" * 62)
        self._write_permanent("")

        payload = {
            "event": "estimate",
            "seconds_per_step": seconds_per_step,
            "seconds_per_epoch": per_epoch,
            "seconds_total": total,
        }
        self._record(payload)
        return payload

    # -- epoch lifecycle ---------------------------------------------------

    def start_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.epoch_step = 0
        self.epoch_start = time.perf_counter()
        self.epoch_rate = RateTracker(window=self.rate_window)
        self.epoch_loss.reset()
        self.epoch_accuracy.reset()

        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # The blank separator goes here rather than at the end of the previous
        # epoch, so any note() the caller emits after end_epoch - "new best
        # EER", "early stopping" - stays visually attached to the epoch it
        # belongs to instead of floating above the next one.
        if self.history:
            self._write_permanent("")

        self._write_permanent(
            f"Epoch {epoch}/{self.total_epochs}"
            f"   {self.steps_per_epoch} steps x {self.batch_size} samples"
        )

    def step(
        self,
        loss: float,
        accuracy: float | None = None,
        lr: float | None = None,
        **extra,
    ) -> None:
        assert self.epoch_rate is not None, "call start_epoch() before step()"

        self.epoch_rate.tick()
        self.epoch_step += 1
        self.global_step += 1

        self.epoch_loss.update(loss)
        if accuracy is not None:
            self.epoch_accuracy.update(accuracy)
        if lr is not None:
            self.last_lr = lr

        line = "  " + self._render_step_line(extra)

        is_heartbeat = (
            self.epoch_step % self.log_every == 0
            or self.epoch_step == self.steps_per_epoch
        )

        if is_heartbeat:
            self._write_permanent(line)
            self._record(
                {
                    "event": "step",
                    "epoch": self.epoch,
                    "epoch_step": self.epoch_step,
                    "global_step": self.global_step,
                    "loss": float(loss),
                    "loss_avg": self.epoch_loss.value,
                    "accuracy": float(accuracy) if accuracy is not None else None,
                    "accuracy_avg": self.epoch_accuracy.value,
                    "lr": self.last_lr,
                    "seconds_per_step": self.epoch_rate.seconds_per_step,
                    "elapsed": self.epoch_rate.elapsed,
                    **{k: v for k, v in extra.items()},
                }
            )
        else:
            self._write_live(line)

    def _render_step_line(self, extra: dict) -> str:
        done = self.epoch_step
        total = self.steps_per_epoch

        bar = self._bar(done / total)

        seconds_per_step = self.epoch_rate.seconds_per_step
        eta_epoch = max(total - done, 0) * seconds_per_step
        remaining_epochs = self.total_epochs - self.epoch
        eta_run = eta_epoch + remaining_epochs * total * seconds_per_step

        parts = [
            f"{bar} {done:>{len(str(total))}}/{total}",
            f"loss {self.epoch_loss.value:.4f}",
        ]

        if self.epoch_accuracy.count:
            parts.append(f"acc {self.epoch_accuracy.value * 100:5.2f}%")

        if seconds_per_step > 0:
            if seconds_per_step >= 1.0:
                parts.append(f"{seconds_per_step:.2f}s/it")
            else:
                parts.append(f"{1.0 / seconds_per_step:.1f}it/s")

        parts.append(f"el {format_duration(self.epoch_rate.elapsed)}")
        parts.append(f"eta {format_duration(eta_epoch)}")

        if self.total_epochs > 1:
            parts.append(f"run {format_duration(eta_run)}")

        if self.last_lr:
            parts.append(f"lr {self.last_lr:.2e}")

        memory = gpu_memory_summary()
        if memory:
            parts.append(memory)

        for key, value in extra.items():
            if isinstance(value, float):
                parts.append(f"{key} {value:.3f}")
            else:
                parts.append(f"{key} {value}")

        return "  ".join(parts)

    @staticmethod
    def _bar(fraction: float, width: int = 20) -> str:
        fraction = min(max(fraction, 0.0), 1.0)
        filled = int(round(fraction * width))
        return "[" + "#" * filled + "." * (width - filled) + f"]{fraction * 100:4.0f}%"

    def end_epoch(self, **extra) -> EpochResult:
        assert self.epoch_rate is not None

        seconds = time.perf_counter() - self.epoch_start

        result = EpochResult(
            epoch=self.epoch,
            loss=self.epoch_loss.value,
            accuracy=self.epoch_accuracy.value,
            seconds=seconds,
            steps=self.epoch_step,
            samples=self.epoch_step * self.batch_size,
            extra=dict(extra),
        )

        self.history.append(result)

        summary = [
            f"epoch {self.epoch} done in {format_duration(seconds)}",
            f"train loss {result.loss:.4f}",
        ]

        if self.epoch_accuracy.count:
            summary.append(f"train acc {result.accuracy * 100:.2f}%")

        summary.append(f"{result.samples / max(seconds, 1e-9):.1f} samples/s")

        memory = gpu_memory_summary()
        if memory:
            summary.append(memory)

        self._write_permanent("  " + "  |  ".join(summary))

        if extra:
            rendered = "   ".join(
                f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                for k, v in extra.items()
            )
            self._write_permanent(f"  validation:  {rendered}")

        remaining = self.total_epochs - self.epoch
        if remaining > 0:
            self._write_permanent(
                f"  {remaining} epoch(s) left  ->  ~{format_duration(remaining * seconds)} remaining"
            )

        self._record(
            {
                "event": "epoch_end",
                "epoch": self.epoch,
                "loss": result.loss,
                "accuracy": result.accuracy,
                "seconds": seconds,
                "steps": result.steps,
                "samples": result.samples,
                **extra,
            }
        )

        return result

    # -- run lifecycle -----------------------------------------------------

    def run_footer(self, **extra) -> None:
        total = time.perf_counter() - self.run_start

        self._write_permanent("=" * 78)
        self._write_permanent(f"  run finished in {format_duration(total)}")

        if self.history:
            average = sum(e.seconds for e in self.history) / len(self.history)
            self._write_permanent(
                f"  {len(self.history)} epoch(s), average {format_duration(average)} per epoch"
            )

        for key, value in extra.items():
            self._write_permanent(f"  {key}: {value}")

        self._write_permanent("=" * 78)

        self._record(
            {
                "event": "run_end",
                "seconds": total,
                **{str(k): str(v) for k, v in extra.items()},
            }
        )


class PhaseTimer:
    """
    Context manager for one-off phases (dataset scan, caching, evaluation) so
    every slow step in the pipeline announces its own cost.

        with PhaseTimer("caching waveforms"):
            ...
    """

    def __init__(self, label: str, stream=None) -> None:
        self.label = label
        self.stream = stream or sys.stdout
        self.start = 0.0
        self.seconds = 0.0

    def __enter__(self) -> "PhaseTimer":
        self.start = time.perf_counter()
        self.stream.write(f"->  {self.label} ...\n")
        self.stream.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.seconds = time.perf_counter() - self.start
        status = "FAILED" if exc_type else "done"
        self.stream.write(
            f"<-  {self.label} {status} in {format_duration(self.seconds)}\n"
        )
        self.stream.flush()
        return False


class IterationProgress:
    """
    Live progress for a single-pass loop (evaluation, caching, manifest scan)
    where there are no epochs. Same ETA logic, lighter output.
    """

    def __init__(
        self,
        total: int,
        label: str,
        log_every: int = 200,
        stream=None,
    ) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.log_every = max(int(log_every), 1)
        self.stream = stream or sys.stdout
        self.live = _is_tty()
        self.rate = RateTracker(window=200)
        self.done = 0
        self._live_line_length = 0

    def update(self, count: int = 1, **extra) -> None:
        for _ in range(max(count, 1)):
            self.rate.tick()
        self.done += count

        eta = (self.total - self.done) * self.rate.seconds_per_step

        line = (
            f"  {self.label}  {self.done}/{self.total} "
            f"({self.done / self.total * 100:5.1f}%)  "
            f"{self.rate.recent_rate:.1f}/s  "
            f"el {format_duration(self.rate.elapsed)}  "
            f"eta {format_duration(eta)}"
        )

        suffix = "  ".join(
            f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
            for k, v in extra.items()
        )
        if suffix:
            line += "  " + suffix

        if self.done % self.log_every == 0 or self.done >= self.total:
            self._clear()
            self.stream.write(line + "\n")
            self.stream.flush()
        elif self.live:
            text = line[: _terminal_width() - 1]
            padding = max(self._live_line_length - len(text), 0)
            self.stream.write("\r" + text + " " * padding)
            self.stream.flush()
            self._live_line_length = len(text)

    def _clear(self) -> None:
        if self.live and self._live_line_length:
            self.stream.write("\r" + " " * self._live_line_length + "\r")
            self.stream.flush()
            self._live_line_length = 0

    def close(self) -> None:
        self._clear()
