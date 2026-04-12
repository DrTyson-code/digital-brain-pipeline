#!/usr/bin/env python3
"""CLI entry point for the Digital Brain Pipeline.

Usage:
    python scripts/run_pipeline.py --config config/settings.yaml
    python scripts/run_pipeline.py --source claude:/path/to/export.json
    python scripts/run_pipeline.py --source chatgpt:~/Downloads/conversations.json --vault ~/Vault
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import Pipeline, PipelineConfig

console = Console()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to settings.yaml config file.",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Source in format platform:/path/to/export (e.g. claude:~/exports/claude.json). Can be repeated.",
)
@click.option(
    "--vault",
    "vault_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override vault output path.",
)
@click.option(
    "--graph",
    "export_graph",
    is_flag=True,
    default=False,
    help="Also export a JSON graph file.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level.",
)
def main(
    config_path: Path | None,
    sources: tuple[str, ...],
    vault_path: Path | None,
    export_graph: bool,
    log_level: str,
) -> None:
    """Run the Digital Brain Pipeline.

    Processes AI chat exports into a structured knowledge graph and writes
    the results as Obsidian-compatible markdown notes.
    """
    setup_logging(log_level)

    # Load config
    if config_path:
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()

    # Override with CLI args
    if vault_path:
        config.vault_path = vault_path.expanduser()
    if export_graph:
        config.export_graph = True

    # Parse --source flags
    source_files: dict[str, Path] = {}
    for source_str in sources:
        if ":" not in source_str:
            console.print(f"[red]Invalid source format: {source_str}[/red]")
            console.print("Expected format: platform:/path/to/export")
            sys.exit(1)
        platform, path_str = source_str.split(":", 1)
        source_files[platform] = Path(path_str).expanduser()

    # Merge with config sources
    if source_files:
        config.source_dirs.update(source_files)

    if not config.source_dirs:
        console.print("[red]No sources specified.[/red]")
        console.print("Use --source or --config to provide input files.")
        sys.exit(1)

    # Run pipeline
    console.print("[bold green]Digital Brain Pipeline[/bold green]")
    console.print(f"Vault: {config.vault_path}")
    console.print(f"Sources: {', '.join(config.source_dirs.keys())}")
    console.print()

    pipeline = Pipeline(config)
    result = pipeline.run()

    console.print()
    console.print(f"[bold green]{result.summary}[/bold green]")


if __name__ == "__main__":
    main()
