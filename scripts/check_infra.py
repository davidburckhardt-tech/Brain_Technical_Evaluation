#!/usr/bin/env python3
"""Records the hardware this phase actually runs on - CPU, RAM, disk, and
every GPU (model, VRAM, driver/CUDA version, and published peak compute) -
into a JSON file the dashboard reads per Phase (Mac vs the NVIDIA server
etc). Standard library only, on purpose: this needs to run on any machine
(Mac laptop, bare Linux server) before any project venv necessarily exists.

Usage: ./scripts/check_infra.py --phase <phase-key> [--output <path>]
  --phase    Matches this project's phase keys (phase0 = Mac/Ollama,
             phase0.5 = NVIDIA server/vLLM, ...) - see build_dashboard.py's
             PHASE_LABELS. Stored in the output so the dashboard can group
             infra info by phase without relying on the filename.
  --output   Defaults to results/infra_check_<phase>.json

GPU peak TFLOPS come from a small lookup table of published vendor specs
(NVIDIA_GPU_SPECS below) matched against the GPU name nvidia-smi reports -
no OS-level command exposes theoretical peak FLOP/s directly, it's a fixed
property of the silicon. A GPU not in the table still gets everything else
(name, VRAM, driver/CUDA version); its peak_tflops fields are left null
rather than guessed. Extend the table as new GPUs show up on test hardware.
"""

import argparse
import datetime
import json
import platform
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Published peak compute (TFLOPS) per vendor spec sheets, dense (not sparse)
# math, with tensor cores where applicable. "fp16_tensor"/"bf16_tensor" are
# the figures relevant to LLM inference/training; "fp32" is the plain CUDA
# core figure. Keyed by a substring of nvidia-smi's "name" field.
NVIDIA_GPU_SPECS = {
    "RTX 4070 SUPER": {"fp32_tflops": 35.5, "fp16_tensor_tflops": 141, "memory_bandwidth_gbps": 504},
    "RTX 4070 Ti SUPER": {"fp32_tflops": 44.1, "fp16_tensor_tflops": 176, "memory_bandwidth_gbps": 672},
    "RTX 4070 Ti": {"fp32_tflops": 40.1, "fp16_tensor_tflops": 160, "memory_bandwidth_gbps": 504},
    "RTX 4070": {"fp32_tflops": 29.1, "fp16_tensor_tflops": 116, "memory_bandwidth_gbps": 504},
    "RTX 4080 SUPER": {"fp32_tflops": 52.2, "fp16_tensor_tflops": 209, "memory_bandwidth_gbps": 736},
    "RTX 4080": {"fp32_tflops": 48.7, "fp16_tensor_tflops": 195, "memory_bandwidth_gbps": 717},
    "RTX 4090": {"fp32_tflops": 82.6, "fp16_tensor_tflops": 330, "memory_bandwidth_gbps": 1008},
    "RTX 3090 Ti": {"fp32_tflops": 40.0, "fp16_tensor_tflops": 160, "memory_bandwidth_gbps": 1008},
    "RTX 3090": {"fp32_tflops": 35.6, "fp16_tensor_tflops": 142, "memory_bandwidth_gbps": 936},
    "A100": {"fp32_tflops": 19.5, "fp16_tensor_tflops": 312, "memory_bandwidth_gbps": 2039},
    "H100": {"fp32_tflops": 67.0, "fp16_tensor_tflops": 989, "memory_bandwidth_gbps": 3350},
    "L40S": {"fp32_tflops": 91.6, "fp16_tensor_tflops": 362, "memory_bandwidth_gbps": 864},
    "L4": {"fp32_tflops": 30.3, "fp16_tensor_tflops": 121, "memory_bandwidth_gbps": 300},
    "T4": {"fp32_tflops": 8.1, "fp16_tensor_tflops": 65, "memory_bandwidth_gbps": 320},
}


def run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def lookup_gpu_specs(name: str) -> dict:
    for key, specs in NVIDIA_GPU_SPECS.items():
        if key in name:
            return specs
    return {"fp32_tflops": None, "fp16_tensor_tflops": None, "memory_bandwidth_gbps": None}


def get_gpus_linux() -> list[dict]:
    """NVIDIA only, via nvidia-smi - this project doesn't target AMD/Intel
    GPUs, and nvidia-smi's CSV output is stable enough to parse directly."""
    out = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        index, name, memory_total_mib, driver_version, compute_cap = parts[:5]
        gpus.append({
            "index": int(index),
            "name": name,
            "vram_total_mb": float(memory_total_mib),
            "driver_version": driver_version,
            "compute_capability": compute_cap,
            **lookup_gpu_specs(name),
        })
    return gpus


def get_cuda_version() -> str | None:
    out = run(["nvidia-smi"])
    if not out:
        return None
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return m.group(1) if m else None


def get_cpu_info_linux() -> dict:
    model = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    nproc = run(["nproc"])
    lscpu = run(["lscpu"])
    sockets = threads_per_core = cores_per_socket = max_mhz = None
    if lscpu:
        for line in lscpu.splitlines():
            if line.startswith("Socket(s):"):
                sockets = line.split(":", 1)[1].strip()
            elif line.startswith("Thread(s) per core:"):
                threads_per_core = line.split(":", 1)[1].strip()
            elif line.startswith("Core(s) per socket:"):
                cores_per_socket = line.split(":", 1)[1].strip()
            elif line.startswith("CPU max MHz:"):
                max_mhz = line.split(":", 1)[1].strip()
    return {
        "model": model,
        "logical_cpus": int(nproc) if nproc and nproc.isdigit() else None,
        "sockets": sockets,
        "cores_per_socket": cores_per_socket,
        "threads_per_core": threads_per_core,
        "max_mhz": max_mhz,
    }


def get_memory_info_linux() -> dict:
    total_kb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                    break
    except OSError:
        pass
    return {"total_gb": round(total_kb / 1024 / 1024, 1) if total_kb else None}


def get_disk_info_linux() -> dict:
    out = run(["df", "-B1", "--output=size,avail", "/"])
    if not out:
        return {"total_gb": None, "available_gb": None}
    lines = out.splitlines()
    if len(lines) < 2:
        return {"total_gb": None, "available_gb": None}
    size_b, avail_b = lines[1].split()
    return {"total_gb": round(int(size_b) / 1e9, 1), "available_gb": round(int(avail_b) / 1e9, 1)}


def get_os_info_linux() -> dict:
    uname = run(["uname", "-a"])
    distro = None
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    distro = line.split("=", 1)[1].strip().strip('"')
                    break
    except OSError:
        pass
    return {"kernel": uname, "distro": distro}


def get_cpu_info_mac() -> dict:
    brand = run(["sysctl", "-n", "machdep.cpu.brand_string"])
    physical = run(["sysctl", "-n", "hw.physicalcpu"])
    logical = run(["sysctl", "-n", "hw.logicalcpu"])
    # Apple Silicon has no "brand_string" (that sysctl is Intel-only) - fall
    # back to the chip name from system_profiler.
    if not brand:
        hw = run(["system_profiler", "SPHardwareDataType"])
        if hw:
            m = re.search(r"Chip:\s*(.+)", hw)
            brand = m.group(1).strip() if m else None
    return {
        "model": brand,
        "physical_cpus": int(physical) if physical and physical.isdigit() else None,
        "logical_cpus": int(logical) if logical and logical.isdigit() else None,
    }


def get_memory_info_mac() -> dict:
    total_bytes = run(["sysctl", "-n", "hw.memsize"])
    return {"total_gb": round(int(total_bytes) / 1e9, 1) if total_bytes and total_bytes.isdigit() else None}


def get_disk_info_mac() -> dict:
    out = run(["df", "-g", "/"])
    if not out:
        return {"total_gb": None, "available_gb": None}
    lines = out.splitlines()
    if len(lines) < 2:
        return {"total_gb": None, "available_gb": None}
    fields = lines[1].split()
    if len(fields) < 4:
        return {"total_gb": None, "available_gb": None}
    return {"total_gb": float(fields[1]), "available_gb": float(fields[3])}


def get_os_info_mac() -> dict:
    product_name = run(["sw_vers", "-productName"])
    product_version = run(["sw_vers", "-productVersion"])
    build = run(["sw_vers", "-buildVersion"])
    return {"name": product_name, "version": product_version, "build": build}


def get_gpus_mac() -> list[dict]:
    """Apple Silicon's GPU is integrated and shares unified memory with the
    CPU - there's no separate VRAM figure or nvidia-smi equivalent, so this
    reports what system_profiler actually exposes (chip/core count) rather
    than forcing it into the discrete-GPU shape used for the Linux/NVIDIA
    server. Not meant to be benchmarked against those numbers directly."""
    out = run(["system_profiler", "SPDisplaysDataType"])
    if not out:
        return []
    gpus = []
    current_name = None
    for line in out.splitlines():
        line = line.strip()
        if line.endswith(":") and "Chipset Model" not in line and ("Graphics" in line or "GPU" in line or re.match(r"^Apple M\d", line)):
            current_name = line.rstrip(":")
        m = re.match(r"Chipset Model:\s*(.+)", line)
        if m:
            current_name = m.group(1).strip()
        m = re.match(r"Total Number of Cores:\s*(\d+)", line)
        if m and current_name:
            gpus.append({"name": current_name, "gpu_cores": int(m.group(1))})
            current_name = None
    return gpus


def collect() -> dict:
    system = platform.system()
    if system == "Linux":
        gpus = get_gpus_linux()
        return {
            "platform": "Linux",
            "hostname": platform.node(),
            "os": get_os_info_linux(),
            "cpu": get_cpu_info_linux(),
            "memory": get_memory_info_linux(),
            "disk": get_disk_info_linux(),
            "cuda_driver_version": get_cuda_version(),
            "gpus": gpus,
            "gpu_count": len(gpus),
        }
    if system == "Darwin":
        return {
            "platform": "Darwin",
            "hostname": platform.node(),
            "os": get_os_info_mac(),
            "cpu": get_cpu_info_mac(),
            "memory": get_memory_info_mac(),
            "disk": get_disk_info_mac(),
            "cuda_driver_version": None,
            "gpus": get_gpus_mac(),
            "gpu_count": None,
        }
    return {"platform": system, "hostname": platform.node(), "error": f"unsupported platform: {system}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, help="Phase key, e.g. phase0 or phase0.5 (see build_dashboard.py's PHASE_LABELS)")
    parser.add_argument("--output", default=None, help="Defaults to results/infra_check_<phase>.json")
    args = parser.parse_args()

    info = collect()
    info["phase"] = args.phase
    info["checked_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "results" / f"infra_check_{args.phase}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"==> Infra check written to {output_path}")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
