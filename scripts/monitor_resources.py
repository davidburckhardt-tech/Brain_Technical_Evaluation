#!/usr/bin/env python3
"""Samples resource usage at a fixed interval and appends rows to a CSV,
until killed (SIGTERM/SIGINT) - meant to run in the background alongside a
benchmark test, started and stopped by the run_*.sh wrappers.

Auto-detects what to monitor:
  - If `nvidia-smi` is on PATH, samples GPU utilization + VRAM per GPU
    (works for the real Brain hardware, NVIDIA GPUs).
  - Otherwise, falls back to CPU utilization + system memory via psutil
    (what this Mac uses today - no GPU monitoring API available without
    sudo-gated tools like powermetrics, which isn't scriptable here).

One CSV schema either way, so downstream tooling doesn't need to care which
mode produced a given file: timestamp_iso, elapsed_s, mode, device, util_percent,
memory_used_mb, memory_total_mb.
"""

import argparse
import csv
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FIELDNAMES = [
    "timestamp_iso", "elapsed_s", "mode", "device", "util_percent",
    "memory_used_mb", "memory_total_mb", "power_draw_w", "temperature_c",
]

_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True


def sample_gpu() -> list[dict]:
    """One row per GPU, via `nvidia-smi --query-gpu=...`."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    rows = []
    for line in result.stdout.strip().splitlines():
        index, name, util, mem_used, mem_total, power_draw, temperature = (
            p.strip() for p in line.split(",")
        )
        rows.append({
            "mode": "gpu",
            "device": f"gpu{index}:{name}",
            "util_percent": float(util),
            "memory_used_mb": float(mem_used),
            "memory_total_mb": float(mem_total),
            "power_draw_w": float(power_draw) if power_draw not in ("", "[N/A]") else None,
            "temperature_c": float(temperature) if temperature not in ("", "[N/A]") else None,
        })
    return rows


def sample_cpu(psutil) -> list[dict]:
    mem = psutil.virtual_memory()
    return [{
        "mode": "cpu",
        "device": "cpu",
        "util_percent": psutil.cpu_percent(interval=None),
        "memory_used_mb": (mem.total - mem.available) / (1024 * 1024),
        "memory_total_mb": mem.total / (1024 * 1024),
        # Power draw and temperature aren't exposed cross-platform without
        # extra sudo-gated tools (e.g. powermetrics on macOS) - left blank in
        # CPU mode rather than approximated.
        "power_draw_w": None,
        "temperature_c": None,
    }]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="CSV file to write samples to")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between samples")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    has_gpu = shutil.which("nvidia-smi") is not None
    psutil_module = None
    if not has_gpu:
        try:
            import psutil as psutil_module
        except ImportError:
            print("Error: no nvidia-smi and psutil is not installed. Run scripts/setup.sh.", file=sys.stderr)
            sys.exit(1)

    mode = "gpu" if has_gpu else "cpu"
    print(f"==> Monitoring {mode} every {args.interval}s -> {args.output}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()

        while not _stop:
            now = datetime.now(timezone.utc).isoformat()
            elapsed = time.monotonic() - start_time
            try:
                rows = sample_gpu() if has_gpu else sample_cpu(psutil_module)
            except Exception as exc:
                print(f"Warning: sample failed: {exc}", file=sys.stderr)
                rows = []

            for row in rows:
                writer.writerow({"timestamp_iso": now, "elapsed_s": round(elapsed, 3), **row})
            f.flush()

            time.sleep(args.interval)

    print(f"==> Monitoring stopped. Samples saved to {output_path}")


if __name__ == "__main__":
    main()
