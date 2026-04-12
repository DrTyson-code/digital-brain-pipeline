#!/usr/bin/env python3
"""Helpers for extracting data from AI platforms.

Provides utilities for locating and preparing export files from
Claude, ChatGPT, and Gemini.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


def find_exports(search_dir: Path) -> dict[str, list[Path]]:
    """Scan a directory for known AI chat export files.

    Returns a dict mapping platform names to lists of found export files.
    """
    results: dict[str, list[Path]] = {
        "claude": [],
        "chatgpt": [],
        "gemini": [],
    }

    for json_file in search_dir.rglob("*.json"):
        platform = detect_platform(json_file)
        if platform:
            results[platform].append(json_file)

    return results


def detect_platform(file_path: Path) -> str | None:
    """Try to detect which AI platform a JSON export file came from."""
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    # ChatGPT: list of objects with "mapping" key
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            if "mapping" in first:
                return "chatgpt"
            if "chat_messages" in first:
                return "claude"
            if "entries" in first:
                return "gemini"

    # Single conversation objects
    if isinstance(data, dict):
        if "chat_messages" in data:
            return "claude"
        if "mapping" in data:
            return "chatgpt"
        if "entries" in data:
            return "gemini"

    return None


def export_stats(file_path: Path) -> dict:
    """Get basic statistics about an export file."""
    data = json.loads(file_path.read_text(encoding="utf-8"))
    conversations = data if isinstance(data, list) else [data]

    total_messages = 0
    for conv in conversations:
        if "chat_messages" in conv:
            total_messages += len(conv["chat_messages"])
        elif "mapping" in conv:
            total_messages += len(conv["mapping"])
        elif "entries" in conv:
            total_messages += len(conv["entries"])

    return {
        "file": str(file_path),
        "conversations": len(conversations),
        "total_messages": total_messages,
        "size_mb": file_path.stat().st_size / (1024 * 1024),
    }


@click.command()
@click.argument(
    "search_dir",
    type=click.Path(exists=True, path_type=Path),
    default="~/Downloads",
)
def main(search_dir: Path) -> None:
    """Scan a directory for AI chat export files and show summary."""
    search_dir = search_dir.expanduser()
    console.print(f"[bold]Scanning {search_dir} for AI chat exports...[/bold]\n")

    results = find_exports(search_dir)

    table = Table(title="Found Export Files")
    table.add_column("Platform", style="cyan")
    table.add_column("File", style="white")
    table.add_column("Conversations", justify="right")
    table.add_column("Messages", justify="right")
    table.add_column("Size (MB)", justify="right")

    total_files = 0
    for platform, files in results.items():
        for f in files:
            stats = export_stats(f)
            table.add_row(
                platform,
                str(f.relative_to(search_dir)),
                str(stats["conversations"]),
                str(stats["total_messages"]),
                f"{stats['size_mb']:.1f}",
            )
            total_files += 1

    if total_files == 0:
        console.print("[yellow]No AI chat export files found.[/yellow]")
        console.print("Export your data from:")
        console.print("  - Claude: claude.ai → Settings → Account → Export Data")
        console.print("  - ChatGPT: chatgpt.com → Settings → Data Controls → Export")
        console.print("  - Gemini: takeout.google.com → Select Gemini Apps")
    else:
        console.print(table)


if __name__ == "__main__":
    main()
