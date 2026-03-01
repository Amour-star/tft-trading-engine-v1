"""Single-instance process lock for the trading engine."""
from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path


class InstanceLock:
    def __init__(self, lock_path: str) -> None:
        self.path = Path(lock_path)
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = self._try_create_lock()

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        except Exception:
            pass
        self._fd = None
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            pass

    def _try_create_lock(self) -> int:
        try:
            return self._create_lock_file()
        except FileExistsError:
            if self._is_stale_lock():
                self.path.unlink(missing_ok=True)
                return self._create_lock_file()
            raise RuntimeError(f"Another engine instance appears to be running (lock: {self.path})")

    def _create_lock_file(self) -> int:
        fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.fsync(fd)
        return fd

    def _is_stale_lock(self) -> bool:
        try:
            raw = self.path.read_text(encoding="ascii").strip()
            if not raw:
                return True
            pid = int(raw)
        except Exception:
            return True

        if pid == os.getpid():
            # If the lock file contains our PID, it cannot be guarding another live
            # process in this PID namespace. This commonly happens after container
            # restarts where PID 1 is reused and the previous process was SIGKILLed
            # before it could remove the lock file.
            return True

        if os.name == "nt":
            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                process_query_limited_information,
                False,
                pid,
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
                cmdline = self._get_windows_commandline(pid)
                if cmdline is None:
                    return False
                cmdline_lower = cmdline.lower()
                is_engine = "scripts/run_engine.py" in cmdline_lower or "engine.main" in cmdline_lower
                return not is_engine
            return True

        try:
            os.kill(pid, 0)
            return False
        except Exception:
            return True

    @staticmethod
    def _get_windows_commandline(pid: int) -> str | None:
        try:
            result = subprocess.run(
                [
                    "wmic",
                    "process",
                    "where",
                    f"processid={pid}",
                    "get",
                    "commandline",
                    "/value",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = (result.stdout or "").strip()
            if "CommandLine=" not in output:
                return None
            return output.split("CommandLine=", 1)[1].strip().strip('"')
        except Exception:
            return None
