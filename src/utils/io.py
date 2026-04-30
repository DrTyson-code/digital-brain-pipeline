"""I/O utilities for durable writes and vault path validation."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class PathTraversalError(ValueError):
    """Raised when a write target resolves outside the allowed root."""

    def __init__(self, path: Path, root: Path) -> None:
        self.path = path
        self.root = root
        super().__init__(f"Path traversal detected: {path} is outside {root}")


def ensure_within_root(path: Path, root: Path) -> Path:
    """Return the resolved path if it stays within root; raise otherwise."""
    resolved_path = Path(path).expanduser().resolve()
    resolved_root = Path(root).expanduser().resolve()
    if not resolved_path.is_relative_to(resolved_root):
        logger.error(
            "Rejected path outside allowed root: %s (root: %s)",
            resolved_path,
            resolved_root,
        )
        raise PathTraversalError(resolved_path, resolved_root)
    return resolved_path


def atomic_write(
    path: Path,
    content: str,
    encoding: str = "utf-8",
    *,
    root: Path | None = None,
) -> None:
    """Write content to path atomically and durably.

    The temp file is created in the destination directory so os.replace() stays
    on the same filesystem. The file is fsynced before rename and the directory
    is fsynced after rename so the replacement is crash-durable.
    """
    path = Path(path)
    if root is not None:
        ensure_within_root(path, root)

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "w", encoding=encoding) as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _fsync_directory(directory: Path) -> None:
    """Fsync a directory entry after a rename."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY

    dir_fd = os.open(directory, flags)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
