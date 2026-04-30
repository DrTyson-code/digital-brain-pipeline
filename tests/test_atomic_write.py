"""Tests for durable atomic write helpers."""

from __future__ import annotations

import logging
import os

import pytest

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
    fsync_calls: list[int] = []

    def record_fsync(fd):
        fsync_calls.append(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)

    atomic_write(target, "durable")

    assert target.read_text(encoding="utf-8") == "durable"
    assert len(fsync_calls) == 2


def test_atomic_write_rejects_path_traversal_and_logs(tmp_path, caplog):
    vault = tmp_path / "vault"
    outside = tmp_path / "outside.md"

    with caplog.at_level(logging.ERROR):
        with pytest.raises(PathTraversalError):
            atomic_write(outside, "escape", root=vault)

    assert str(outside.resolve()) in caplog.text
    assert not outside.exists()
