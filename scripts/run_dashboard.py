"""Start the dashboard stack (FastAPI + Streamlit)."""
from __future__ import annotations

import signal
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings


def _start_api() -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "dashboard.api:app",
            "--host",
            str(settings.dashboard.api_host),
            "--port",
            str(settings.dashboard.api_port),
        ],
        cwd=str(ROOT),
    )


def _start_streamlit() -> subprocess.Popen:
    api_host_for_ui = settings.dashboard.api_host
    if api_host_for_ui == "0.0.0.0":
        api_host_for_ui = "127.0.0.1"
    env = dict(os.environ)
    env["API_BASE"] = f"http://{api_host_for_ui}:{settings.dashboard.api_port}/api"

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "dashboard/streamlit_app.py",
            "--server.port",
            str(settings.dashboard.dashboard_port),
            "--server.address",
            str(settings.dashboard.dashboard_host),
            "--server.headless",
            "true",
        ],
        cwd=str(ROOT),
        env=env,
    )


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def main() -> None:
    api_proc = _start_api()
    ui_proc = _start_streamlit()

    try:
        # Keep foreground attached to Streamlit lifecycle.
        ui_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        _terminate(ui_proc)
        _terminate(api_proc)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
