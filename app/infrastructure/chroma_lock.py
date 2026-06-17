"""Cross-process lock for Chroma PersistentClient (SQLite on Windows)."""

from __future__ import annotations

import os
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x100000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def _read_lock_pid(lock_path: Path) -> int | None:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
        return int(raw) if raw.isdigit() else None
    except OSError:
        return None


class ChromaProcessLock:
    """Exclusive lock so only one process opens ``data/persist_db`` at a time."""

    _LOCK_NAME = ".chroma.lock"

    def __init__(self, persist_path: Path) -> None:
        self._lock_path = persist_path.parent / self._LOCK_NAME
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd: int | None = None
        self._held = False

    def acquire(self) -> None:
        if self._held:
            return
        for attempt in range(2):
            try:
                self._fd = os.open(
                    self._lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                )
                break
            except FileExistsError:
                stale_pid = _read_lock_pid(self._lock_path)
                if attempt == 0 and (stale_pid is None or not _pid_alive(stale_pid)):
                    try:
                        os.remove(self._lock_path)
                    except OSError:
                        pass
                    continue
                holder = stale_pid if stale_pid is not None else "unknown"
                raise RuntimeError(
                    "Chroma database is locked by another process "
                    f"(pid={holder}, lock={self._lock_path}). "
                    "Stop duplicate uvicorn/monitor instances and keep only one "
                    "server running."
                ) from None
        os.write(self._fd, str(os.getpid()).encode())
        self._held = True

    def release(self) -> None:
        if not self._held:
            return
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            os.remove(self._lock_path)
        except OSError:
            pass
        self._held = False
