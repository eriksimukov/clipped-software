"""Macht die CUDA-Bibliotheken aus den nvidia-*-Paketen fuer CTranslate2 auffindbar."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def enable_cuda_dlls() -> list[Path]:
    """CTranslate2 laedt cublas und cudnn zur Laufzeit ueber den normalen Windows-Suchpfad."""
    if sys.platform != "win32":
        return []
    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    found = [p for p in (site_packages / "cublas" / "bin", site_packages / "cudnn" / "bin") if p.is_dir()]
    for path in found:
        os.add_dll_directory(str(path))
        os.environ["PATH"] = f"{path}{os.pathsep}{os.environ.get('PATH', '')}"
    return found
