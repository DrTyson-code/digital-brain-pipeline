"""Tests for vault backup retention caps."""

from __future__ import annotations

from pathlib import Path

from src.pipeline import BackupConfig, Pipeline, PipelineConfig

BYTES_PER_GB = 1024 ** 3


def _make_backup(backup_root: Path, name: str, size: int) -> Path:
    backup = backup_root / name
    backup.mkdir(parents=True)
    (backup / "payload.bin").write_bytes(b"x" * size)
    return backup


def _pipeline(vault_path: Path, *, max_count: int, max_total_size: int) -> Pipeline:
    return Pipeline(
        PipelineConfig(
            vault_path=vault_path,
            backup=BackupConfig(
                max_count=max_count,
                max_total_size_gb=max_total_size / BYTES_PER_GB,
            ),
        )
    )


def test_backup_retention_enforces_count_cap_oldest_first(tmp_path):
    backup_root = tmp_path / ".backup"
    _make_backup(backup_root, "20240101T000000", 1)
    _make_backup(backup_root, "20240102T000000", 1)
    _make_backup(backup_root, "20240103T000000", 1)

    _pipeline(tmp_path / "vault", max_count=2, max_total_size=100)._prune_backups(
        backup_root
    )

    assert [path.name for path in Pipeline._backup_dirs(backup_root)] == [
        "20240102T000000",
        "20240103T000000",
    ]


def test_backup_retention_enforces_total_size_cap_oldest_first(tmp_path):
    backup_root = tmp_path / ".backup"
    _make_backup(backup_root, "20240101T000000", 4)
    _make_backup(backup_root, "20240102T000000", 4)
    _make_backup(backup_root, "20240103T000000", 4)

    _pipeline(tmp_path / "vault", max_count=10, max_total_size=9)._prune_backups(
        backup_root
    )

    remaining = Pipeline._backup_dirs(backup_root)
    assert [path.name for path in remaining] == [
        "20240102T000000",
        "20240103T000000",
    ]
    assert Pipeline._total_backup_size(remaining) == 8


def test_backup_retention_thresholds_are_configurable():
    config = BackupConfig.from_dict({"max_count": 3, "max_total_size_gb": 0.25})

    assert config.max_count == 3
    assert config.max_total_size_gb == 0.25


def test_backup_vault_prunes_after_successful_copy(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "current.md").write_text("current", encoding="utf-8")
    backup_root = tmp_path / ".backup"
    _make_backup(backup_root, "20240101T000000", 1)

    pipeline = _pipeline(vault, max_count=1, max_total_size=100)
    backup_path = pipeline._backup_vault("Stage 6 Obsidian write")

    remaining = Pipeline._backup_dirs(backup_root)
    assert backup_path in remaining
    assert len(remaining) == 1
    assert not (backup_root / "20240101T000000").exists()
