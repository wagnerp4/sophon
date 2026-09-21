from __future__ import annotations

import os
import string
import sys
from dataclasses import dataclass
from pathlib import Path

_SKIP_MOUNT_NAMES = frozenset({"wsl", "wslg"})
_SKIP_FSTYPES = frozenset(
    {
        "squashfs",
        "overlay",
        "tmpfs",
        "devtmpfs",
        "proc",
        "sysfs",
        "cgroup",
        "cgroup2",
        "devpts",
        "securityfs",
        "debugfs",
        "tracefs",
        "fusectl",
        "configfs",
        "pstore",
        "bpf",
        "hugetlbfs",
        "mqueue",
        "ramfs",
    }
)


@dataclass(frozen=True)
class DiskRoot:
    path: Path
    fstype: str
    label: str


def _readable_dir(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_dir():
            return False
    except OSError:
        return False
    return os.access(path, os.R_OK)


def _mount_key(path: Path) -> str:
    text = str(path)
    if sys.platform == "win32":
        return os.path.normcase(os.path.normpath(text)).rstrip("\\") + "\\"
    return os.path.normpath(text)


def _disk_label(path: Path) -> str:
    if sys.platform == "win32":
        drive = path.drive
        if drive:
            return f"{drive}\\"
    return str(path)


def _add_disk(rows: list[DiskRoot], seen: set[str], path: Path, fstype: str = "") -> None:
    try:
        candidate = path.expanduser()
    except OSError:
        return
    if not candidate.is_absolute():
        try:
            candidate = candidate.resolve()
        except OSError:
            return
    if candidate.name.lower() in _SKIP_MOUNT_NAMES:
        return
    if not _readable_dir(candidate):
        return
    key = _mount_key(candidate)
    if not key or key in seen:
        return
    seen.add(key)
    rows.append(DiskRoot(path=candidate, fstype=fstype, label=_disk_label(candidate)))


def list_readable_disks() -> tuple[DiskRoot, ...]:
    # TODO: include Windows volume labels next to the drive letter
    # TODO: skip or timeout disconnected network drives
    rows: list[DiskRoot] = []
    seen: set[str] = set()
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            partitions = psutil.disk_partitions(all=False)
        except Exception:
            partitions = []
        for part in partitions:
            fstype = str(part.fstype or "")
            if fstype.lower() in _SKIP_FSTYPES:
                continue
            mount = str(part.mountpoint or "")
            if not mount:
                continue
            _add_disk(rows, seen, Path(mount), fstype)
    if sys.platform == "win32":
        for letter in string.ascii_uppercase:
            _add_disk(rows, seen, Path(f"{letter}:\\"))
    else:
        _add_disk(rows, seen, Path("/"))
        mnt = Path("/mnt")
        if _readable_dir(mnt):
            try:
                children = list(mnt.iterdir())
            except OSError:
                children = []
            for child in children:
                _add_disk(rows, seen, child)
    rows.sort(key=lambda row: row.label.lower())
    return tuple(rows)


def same_disk(left: Path, right: Path) -> bool:
    return _mount_key(left) == _mount_key(right)
