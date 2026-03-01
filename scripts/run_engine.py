"""Start the trading engine with runtime environment checks."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _validate_python_version() -> None:
    if sys.version_info[:2] > (3, 12):
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(
            f"Python {current} is not supported for this project. "
            "Use Python 3.11 or 3.12."
        )


def _configure_windows_event_loop() -> None:
    if sys.platform != "win32":
        return
    try:
        policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if policy_cls is not None:
            asyncio.set_event_loop_policy(policy_cls())
    except Exception:
        # Non-fatal: keep default loop policy if override fails.
        pass


def _log_environment_diagnostics() -> None:
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"[ENV] Python: {py_version}")

    try:
        import torch
    except ImportError:
        print("[ENV] Torch: not installed")
        print("[ENV] Device: cpu")
        print("[ENV] CUDA available: False")
        print("[ENV] GPU: Not available")
        print("Torch not installed. Install compatible version.")
        return

    print(f"[ENV] Torch: {getattr(torch, '__version__', 'unknown')}")
    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    print(f"[ENV] Device: {device}")
    print(f"[ENV] CUDA available: {cuda_available}")
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "Unknown GPU"
    else:
        gpu_name = "Not available"
    print(f"[ENV] GPU: {gpu_name}")


def _run_engine() -> None:
    from core.shutdown_controller import shutdown_controller
    from engine.main import main

    shutdown_controller.bind_signals()
    main()


if __name__ == "__main__":
    _validate_python_version()
    _configure_windows_event_loop()
    _log_environment_diagnostics()
    _run_engine()
