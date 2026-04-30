"""Tests for durable atomic write helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

import src.utils.io as io_module
from src.utils.io import PathTraversalError, atomic_write


def test_atomic_write_succeeds(tmp_path):
    target = tmp_path / "note.md"

    atomic_write(target, "hello")

    assert target.read_text(encoding="utf-8") == "hello"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_failure_cleans_temp_file(tmp_path, monkeypatch):
    target = tmp_path / "note.md"
    target.write_text("old", encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated rename failure"):
        atomic_write(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_atomic_write_fsyncs_file_and_directory(tmp_path, monkeypatch):
    target = tmp_path / "note.md"
    fsync_calls: list[tuple[str, int, str]] = []
    temp_fds: list[int] = []
    directory_fds: list[int] = []
    replace_called = [False]

    def record_fsync(fd):
        phase = "after" if replace_called[0] else "before"
        fsync_calls.append(("fsync", fd, phase))

    real_mkstemp = io_module.tempfile.mkstemp

    def record_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        temp_fds.append(fd)
        return fd, path

    real_open = os.open

    def record_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == target.parent:
            directory_fds.append(fd)
        return fd

    real_replace = os.replace

    def record_replace(*args, **kwargs):
        replace_called[0] = True
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(io_module.tempfile, "mkstemp", record_mkstemp)
    monkeypatch.setattr(os, "open", record_open)
    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "replace", record_replace)

    atomic_write(target, "durable")

    assert target.read_text(encoding="utf-8") == "durable"
    assert fsync_calls == [
        ("fsync", temp_fds[0], "before"),
        ("fsync", directory_fds[0], "after"),
    ]


def test_atomic_write_rejects_path_traversal_and_logs(tmp_path, caplog):
    vault = tmp_path / "vault"
    outside = tmp_path / "outside.md"

    with caplog.at_level(logging.ERROR):
        with pytest.raises(PathTraversalError):
            atomic_write(outside, "escape", root=vault)

    assert str(outside.resolve()) in caplog.text
    assert not outside.exists()
