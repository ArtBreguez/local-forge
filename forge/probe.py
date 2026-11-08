"""Detect the machine: OS, CPU, RAM, GPUs (+backend, VRAM), disk, power source.

Every field may be None: a missing tool or an unreadable file is recorded in
`notes` and never raises. All sizes are MiB.

GPU sources, in order of trust:
  1. `llama-bench --list-devices` / `llama-server --list-devices` of the build that will
     actually run the model. It lists exactly the devices that build can use, with the
     backend (Vulkan0, CUDA0, ROCm0, MTL0...) and free memory right now.
     It does not load a model.
  2. OS tools, only if no llama.cpp binary could be run: nvidia-smi, PowerShell CIM
     (Windows), amdgpu sysfs (Linux), system_profiler (macOS). Backend is then inferred and
     free VRAM is often unknown.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from typing import Any, Callable

from . import util

SCHEMA = 1

_BACKEND_NAMES = {
    "vulkan": "Vulkan", "cuda": "CUDA", "rocm": "ROCm", "hip": "ROCm", "mtl": "Metal",
    "metal": "Metal", "sycl": "SYCL", "cann": "CANN", "opencl": "OpenCL", "musa": "MUSA",
}
_DEVICE_RE = re.compile(r"^\s*([A-Za-z]+)(\d+):\s*(.+?)\s*\((\d+)\s*MiB(?:,\s*(\d+)\s*MiB free)?\)\s*$")


# ---------------------------------------------------------------------------------
# pure parsers (unit-tested on recorded outputs in tests/fixtures)
# ---------------------------------------------------------------------------------

def parse_list_devices(text: str) -> tuple[bool, list[dict[str, Any]]]:
    """Parse `--list-devices` output. Returns (understood, gpus).

    `understood` is True when the "Available devices:" header was seen, i.e. the build ran
    and the (possibly empty) list is authoritative: "(none)" means CPU only.
    CPU and RPC entries are not local accelerators and are skipped.
    """
    seen_header = False
    gpus: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip().lower().startswith("available devices"):
            seen_header = True
            continue
        if not seen_header:
            continue  # ggml_vulkan:/load_backend: chatter before the list
        m = _DEVICE_RE.match(line)
        if not m:
            continue
        prefix, idx, name, total, free = m.groups()
        backend = _BACKEND_NAMES.get(prefix.lower())
        if backend is None:
            continue  # CPU, RPC, BLAS...
        gpus.append({
            "index": len(gpus), "device": f"{prefix}{idx}", "backend": backend, "name": name,
            "total_mib": int(total), "free_mib": int(free) if free is not None else None,
            "source": "llama.cpp --list-devices",
        })
    return seen_header, gpus


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            total, free = int(float(parts[-2])), int(float(parts[-1]))
        except ValueError:
            continue
        gpus.append({"index": len(gpus), "device": f"nvidia{len(gpus)}", "backend": "CUDA",
                     "name": ", ".join(parts[:-2]), "total_mib": total, "free_mib": free,
                     "source": "nvidia-smi (backend inferred)"})
    return gpus


def parse_meminfo(text: str) -> dict[str, int | None]:
    """Parse Linux /proc/meminfo into total/available MiB (falls back to MemFree+Cached)."""
    kb: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            kb[m.group(1)] = int(m.group(2))
    total = kb.get("MemTotal")
    avail = kb.get("MemAvailable")
    if avail is None and "MemFree" in kb:
        avail = kb["MemFree"] + kb.get("Buffers", 0) + kb.get("Cached", 0)
    return {"total_mib": total // 1024 if total else None, "available_mib": avail // 1024 if avail is not None else None}


def parse_cpuinfo(text: str) -> dict[str, Any]:
    """Parse Linux /proc/cpuinfo: model name, logical threads, physical cores."""
    model = None
    logical = 0
    cores: set[tuple[str, str]] = set()
    cur: dict[str, str] = {}

    def flush() -> None:
        nonlocal logical
        if cur and "processor" in cur:       # trailing Hardware/Revision/Serial blocks (ARM boards) are not CPUs
            logical += 1
            if "core id" in cur:
                cores.add((cur.get("physical id", "0"), cur["core id"]))
        cur.clear()

    for line in text.splitlines() + [""]:
        if not line.strip():
            flush()
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "processor":
            flush()
        cur[k] = v
        if k in ("model name", "Hardware", "cpu model") and model is None:
            model = v
    return {"model": model, "logical_threads": logical or None, "physical_cores": len(cores) or None}


def parse_windows_cim_processor(text: str) -> dict[str, Any]:
    """Parse `Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json`."""
    data = json.loads(text)
    items = data if isinstance(data, list) else [data]
    cores = sum(int(i.get("NumberOfCores") or 0) for i in items) or None
    logical = sum(int(i.get("NumberOfLogicalProcessors") or 0) for i in items) or None
    name = (items[0].get("Name") or "").strip() or None if items else None
    return {"model": name, "physical_cores": cores, "logical_threads": logical}


def parse_windows_cim_video(text: str) -> list[dict[str, Any]]:
    """Parse Win32_VideoController JSON. AdapterRAM is a uint32: it caps at 4 GiB, so for
    cards with more memory it understates the total (flagged with total_is_lower_bound)."""
    data = json.loads(text)
    items = data if isinstance(data, list) else [data]
    gpus = []
    for it in items:
        name = (it.get("Name") or "").strip()
        if not name or "microsoft basic" in name.lower() or "remote display" in name.lower():
            continue
        ram = it.get("AdapterRAM")
        total = int(ram) // (1024 * 1024) if ram else None
        gpus.append({"index": len(gpus), "device": f"cim{len(gpus)}", "backend": None, "name": name,
                     "total_mib": total, "free_mib": None,
                     "total_is_lower_bound": bool(ram and int(ram) >= 4278190080),
                     "source": "Win32_VideoController (backend unknown)"})
    return gpus


def parse_power_supply(entries: list[dict[str, str]]) -> dict[str, Any]:
    """Decide battery state from Linux /sys/class/power_supply entries (dicts of file -> content)."""
    batteries = [e for e in entries if e.get("type", "").strip().lower() == "battery"
                 and e.get("scope", "System").strip().lower() != "device"]  # skip mouse/keyboard batteries
    mains = [e for e in entries if e.get("type", "").strip().lower() in ("mains", "usb", "ups")]
    if batteries:
        status = [b.get("status", "").strip().lower() for b in batteries]
        mains_online = any(m.get("online", "0").strip() == "1" for m in mains)
        on_batt = any(s == "discharging" for s in status) and not mains_online
        cap = None
        try:
            cap = int(batteries[0].get("capacity", "").strip())
        except ValueError:
            pass
        return {"on_battery": on_batt, "battery_percent": cap, "source": "sysfs power_supply"}
    if mains:
        return {"on_battery": not any(m.get("online", "0").strip() == "1" for m in mains),
                "battery_percent": None, "source": "sysfs power_supply (mains only)"}
    return {"on_battery": None, "battery_percent": None, "source": None}


def parse_pmset_batt(text: str) -> dict[str, Any]:
    """macOS `pmset -g batt`: "Now drawing from 'Battery Power'" / "'AC Power'"."""
    m = re.search(r"Now drawing from '([^']+)'", text)
    pct = re.search(r"(\d+)%", text)
    return {"on_battery": (m.group(1).lower().startswith("battery")) if m else None,
            "battery_percent": int(pct.group(1)) if pct else None, "source": "pmset"}


# ---------------------------------------------------------------------------------
# platform readers
# ---------------------------------------------------------------------------------

def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _windows_mem() -> dict[str, int | None]:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
        raise OSError("GlobalMemoryStatusEx failed")
    return {"total_mib": st.ullTotalPhys // 2**20, "available_mib": st.ullAvailPhys // 2**20}


def _windows_power() -> dict[str, Any]:
    import ctypes

    class SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    st = SYSTEM_POWER_STATUS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):  # type: ignore[attr-defined]
        raise OSError("GetSystemPowerStatus failed")
    # ACLineStatus: 0 offline (battery / UPS), 1 online, 255 unknown. BatteryFlag 128: no battery.
    on_batt = None if st.ACLineStatus == 255 else st.ACLineStatus == 0
    pct = None if st.BatteryLifePercent == 255 else int(st.BatteryLifePercent)
    return {"on_battery": on_batt, "battery_percent": pct, "source": "GetSystemPowerStatus"}


def _powershell_json(script: str) -> str | None:
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        return None
    r = util.run_cmd([ps, "-NoProfile", "-NonInteractive", "-Command", script], timeout=30)
    if r is None or r[0] != 0 or not r[1].strip():
        return None
    return r[1]


def _sysfs_power_entries() -> list[dict[str, str]]:
    base = "/sys/class/power_supply"
    out = []
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for n in names:
        entry = {}
        for f in ("type", "status", "capacity", "online", "scope"):
            t = _read_text(f"{base}/{n}/{f}")
            if t is not None:
                entry[f] = t.strip()
        out.append(entry)
    return out


def _amdgpu_sysfs() -> list[dict[str, Any]]:
    gpus = []
    base = "/sys/class/drm"
    try:
        cards = sorted(c for c in os.listdir(base) if re.fullmatch(r"card\d+", c))
    except OSError:
        return gpus
    for c in cards:
        dev = f"{base}/{c}/device"
        tot = _read_text(f"{dev}/mem_info_vram_total")
        used = _read_text(f"{dev}/mem_info_vram_used")
        vendor = (_read_text(f"{dev}/vendor") or "").strip()
        if tot is None or not tot.strip().isdigit():
            continue
        total = int(tot) // 2**20
        free = total - int(used) // 2**20 if used and used.strip().isdigit() else None
        gpus.append({"index": len(gpus), "device": c, "backend": None,
                     "name": f"amdgpu {c} (vendor {vendor})", "total_mib": total, "free_mib": free,
                     "source": "amdgpu sysfs (backend unknown)"})
    return gpus


# ---------------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------------

def _section(name: str, notes: list[str], fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - the probe must never crash
        notes.append(f"{name}: {type(e).__name__}: {e}")
        return default


def _cpu_info(notes: list[str]) -> dict[str, Any]:
    info: dict[str, Any] = {"model": None, "physical_cores": None, "logical_threads": os.cpu_count()}
    system = platform.system()
    if system == "Linux":
        t = _read_text("/proc/cpuinfo")
        if t:
            info.update({k: v for k, v in parse_cpuinfo(t).items() if v is not None})
    elif system == "Windows":
        out = _powershell_json("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress")
        if out:
            info.update({k: v for k, v in _section("cim processor", notes, lambda: parse_windows_cim_processor(out), {}).items() if v is not None})
        else:
            notes.append("cpu: PowerShell CIM unavailable; physical core count unknown")
    elif system == "Darwin":
        for key, fld in (("machdep.cpu.brand_string", "model"), ("hw.physicalcpu", "physical_cores"), ("hw.logicalcpu", "logical_threads")):
            r = util.run_cmd(["sysctl", "-n", key])
            if r and r[0] == 0 and r[1].strip():
                info[fld] = int(r[1]) if fld != "model" else r[1].strip()
    if not info["model"]:
        info["model"] = platform.processor() or None
    return info


def _ram_info(notes: list[str]) -> dict[str, Any]:
    system = platform.system()
    if system == "Windows":
        return _windows_mem()
    if system == "Linux":
        t = _read_text("/proc/meminfo")
        if t:
            return parse_meminfo(t)
    if system == "Darwin":
        total = util.run_cmd(["sysctl", "-n", "hw.memsize"])
        vm = util.run_cmd(["vm_stat"])
        res: dict[str, Any] = {"total_mib": int(total[1]) // 2**20 if total and total[0] == 0 else None, "available_mib": None}
        if vm and vm[0] == 0:
            ps = re.search(r"page size of (\d+) bytes", vm[1])
            pages = sum(int(m.group(1)) for m in re.finditer(r"Pages (?:free|inactive|speculative):\s+(\d+)", vm[1]))
            if ps:
                res["available_mib"] = pages * int(ps.group(1)) // 2**20
        return res
    return {"total_mib": None, "available_mib": None}


def _power_info(notes: list[str]) -> dict[str, Any]:
    system = platform.system()
    if system == "Windows":
        return _windows_power()
    if system == "Linux":
        return parse_power_supply(_sysfs_power_entries())
    if system == "Darwin":
        r = util.run_cmd(["pmset", "-g", "batt"])
        if r and r[0] == 0:
            return parse_pmset_batt(r[1])
    return {"on_battery": None, "battery_percent": None, "source": None}


def _os_gpus(notes: list[str]) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    nv = shutil.which("nvidia-smi")
    if nv:
        r = util.run_cmd([nv, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
        if r and r[0] == 0:
            gpus += parse_nvidia_smi(r[1])
    else:
        notes.append("gpu: nvidia-smi not found")
    system = platform.system()
    if not gpus and system == "Windows":
        out = _powershell_json("Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress")
        if out:
            gpus += _section("cim video", notes, lambda: parse_windows_cim_video(out), [])
    if not gpus and system == "Linux":
        gpus += _amdgpu_sysfs()
    if not gpus and system == "Darwin":
        r = util.run_cmd(["system_profiler", "SPDisplaysDataType"], timeout=30)
        if r and r[0] == 0:
            for m in re.finditer(r"Chipset Model:\s*(.+)", r[1]):
                gpus.append({"index": len(gpus), "device": f"mac{len(gpus)}", "backend": "Metal", "name": m.group(1).strip(),
                             "total_mib": None, "free_mib": None, "source": "system_profiler (unified memory, size unknown)"})
    for i, g in enumerate(gpus):
        g["index"] = i
    return gpus


def _llama_gpus(llama_dir: str | None, notes: list[str]) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    """Run --list-devices on the llama.cpp build. Returns (authoritative, gpus, llama_info)."""
    info: dict[str, Any] = {"dir": llama_dir, "bench": util.find_tool("llama-bench", llama_dir),
                            "server": util.find_tool("llama-server", llama_dir)}
    for key in ("bench", "server"):
        exe = info[key]
        if not exe:
            continue
        r = util.run_cmd([exe, "--list-devices"], timeout=60)
        if r is None:
            notes.append(f"llama.cpp: could not run {exe} --list-devices")
            continue
        ok, gpus = parse_list_devices(r[1])
        if ok:
            return True, gpus, info
        notes.append(f"llama.cpp: {os.path.basename(exe)} --list-devices output not understood")
    if not info["bench"] and not info["server"]:
        notes.append("llama.cpp: llama-bench/llama-server not found; GPU list comes from OS tools")
    return False, [], info


def probe(llama_dir: str | None = None, model_dir: str | None = None) -> dict[str, Any]:
    """Return the machine profile as a JSON-serialisable dict. Never raises."""
    notes: list[str] = []
    prof: dict[str, Any] = {"schema": SCHEMA, "taken_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    prof["os"] = _section("os", notes, lambda: {"system": platform.system(), "release": platform.release(),
                                                 "version": platform.version(), "machine": platform.machine()},
                          {"system": None, "release": None, "version": None, "machine": None})
    prof["cpu"] = _section("cpu", notes, lambda: _cpu_info(notes), {"model": None, "physical_cores": None, "logical_threads": None})
    prof["ram"] = _section("ram", notes, lambda: _ram_info(notes), {"total_mib": None, "available_mib": None})

    authoritative, gpus, llama = _section("llama.cpp", notes, lambda: _llama_gpus(llama_dir, notes), (False, [], {}))
    if not authoritative:
        gpus = _section("gpu", notes, lambda: _os_gpus(notes), [])
    prof["gpus"] = gpus
    prof["gpu_source"] = "llama.cpp" if authoritative else ("os" if gpus else None)
    prof["llama"] = llama

    def disk() -> dict[str, Any]:
        path = model_dir or str(util.default_cache_dir())
        while path and not os.path.exists(path):  # walk up to an existing parent
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
        du = shutil.disk_usage(path)
        return {"path": path, "free_mib": du.free // 2**20, "total_mib": du.total // 2**20}

    prof["disk"] = _section("disk", notes, disk, {"path": model_dir, "free_mib": None, "total_mib": None})
    prof["power"] = _section("power", notes, lambda: _power_info(notes), {"on_battery": None, "battery_percent": None, "source": None})
    prof["notes"] = notes
    prof["fingerprint"] = fingerprint(prof)
    return prof


def fingerprint(prof: dict[str, Any]) -> str:
    """Stable identity of the hardware: changes when a part is swapped, not when load changes.

    Uses OS family, CPU model + core/thread counts, total RAM rounded to GiB, and each GPU's
    backend/name/total VRAM. Free memory, battery state and disk space are deliberately left out.
    """
    cpu = prof.get("cpu") or {}
    ram = (prof.get("ram") or {}).get("total_mib")
    parts = {
        "os": (prof.get("os") or {}).get("system"),
        "cpu": [(cpu.get("model") or "").strip(), cpu.get("physical_cores"), cpu.get("logical_threads")],
        "ram_gib": round(ram / 1024) if ram else None,
        "gpus": [[g.get("backend"), g.get("name"), g.get("total_mib")] for g in prof.get("gpus") or []],
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="print the machine profile as JSON")
    ap.add_argument("--llama-dir")
    ap.add_argument("--model-dir")
    a = ap.parse_args(argv)
    json.dump(probe(a.llama_dir, a.model_dir), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
