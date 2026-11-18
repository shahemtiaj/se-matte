"""Device selection: Apple Metal (MPS) on Mac, CUDA on Windows/Linux NVIDIA, CPU fallback.

Kept OS-neutral so the same engine runs on Mac and Windows.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass

# Let unsupported Metal ops fall back to CPU instead of crashing. Must be set before torch import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


@dataclass
class DeviceInfo:
    device: torch.device
    kind: str  # "mps" | "cuda" | "cpu"
    name: str
    memory_gb: float  # unified memory on Mac, VRAM on CUDA, system RAM on CPU
    os: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "memory_gb": round(self.memory_gb, 1), "os": self.os}


def _system_ram_gb() -> float:
    try:
        if platform.system() == "Darwin":
            import subprocess

            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / 1024**3
        if platform.system() == "Windows":
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 1024**3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except Exception:
        return 0.0


def _mac_chip() -> str:
    try:
        import subprocess

        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return "Apple Silicon"


def pick_device(prefer: str | None = None) -> DeviceInfo:
    """Pick the best available device. `prefer` may force 'cpu', 'mps' or 'cuda'."""
    osname = platform.system()
    if prefer in (None, "cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(torch.device("cuda"), "cuda", props.name, props.total_memory / 1024**3, osname)
    if prefer in (None, "mps") and torch.backends.mps.is_available():
        return DeviceInfo(torch.device("mps"), "mps", _mac_chip(), _system_ram_gb(), osname)
    return DeviceInfo(torch.device("cpu"), "cpu", platform.processor() or "CPU", _system_ram_gb(), osname)


def empty_cache(dev: DeviceInfo) -> None:
    if dev.kind == "mps":
        torch.mps.empty_cache()
    elif dev.kind == "cuda":
        torch.cuda.empty_cache()
