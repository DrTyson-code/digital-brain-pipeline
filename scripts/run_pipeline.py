#!/usr/bin/env python3
"""CLI entry point for the Digital Brain Pipeline.

Usage:
    python scripts/run_pipeline.py --config config/settings.yaml
    python scripts/run_pipeline.py --source claude:/path/to/export.json
    python scripts/run_pipeline.py --source chatgpt:~/Downloads/conversations.json --vault ~/Vault
    python scripts/run_pipeline.py --extraction-mode llm_augmented --budget 2.00
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

from src.pipeline import Pipeline, PipelineConfig, ExtractionMode

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
    "--extraction-mode",
    "extraction_mode",
    type=click.Choice(["rules_only", "llm_augmented", "llm_primary", "llm_only"], case_sensitive=False),
    default=None,
    help="Extraction mode (default: from config, or rules_only).",
)
@click.option(
    "--provider",
    "provider_name",
    type=click.Choice(["claude", "openai", "ollama"], case_sensitive=False),
    default=None,
    help="LLM provider (default: from config, or claude).",
)
@click.option(
    "--budget",
    "budget_usd",
    type=float,
    default=None,
    help="Max USD budget for LLM calls (default: $1.00).",
)
@click.option(
    "--no-mocs",
    "skip_mocs",
    is_flag=True,
    default=False,
    help="Skip MOC generation.",
)
@click.option(
    "--graph",
    "export_graph",
    is_flag=True,
    default=False,
    help="Also export a JSON graph file.",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Show what would be processed without running (cost estimate for LLM modes).",
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
    extraction_mode: str | None,
    provider_name: str | None,
    budget_usd: float | None,
    skip_mocs: bool,
    export_graph: bool,
    dry_run: bool,
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
    if skip_mocs:
        config.generate_mocs = False
    if extraction_mode:
        config.llm.extraction_mode = ExtractionMode(extraction_mode)
    if provider_name:
        config.llm.provider_name = provider_name
    if budget_usd is not None:
        config.llm.max_cost_per_run_usd = budget_usd

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

    # Dry run: just show what would happen
    if dry_run:
        console.print("[bold yellow]DRY RUN[/bold yellow]")
        console.print(f"Vault: {config.vault_path}")
        console.print(f"Sources: {', '.join(config.source_dirs.keys())}")
        console.print(f"Extraction mode: {config.llm.extraction_mode.value}")
        if config.llm.extraction_mode != ExtractionMode.RULES_ONLY:
            console.print(f"Provider: {config.llm.provider_name}")
            console.print(f"Budget: ${config.llm.max_cost_per_run_usd:.2f}")
            # Rough cost estimate: count conversations and estimate
            from src.ingest import ClaudeIngester, ChatGPTIngester, GeminiIngester
            INGEST_MAP = {"claude": ClaudeIngester, "chatgpt": ChatGPTIngester, "gemini": GeminiIngester}
            total_convs = 0
            for platform, path in config.source_dirs.items():
                path = Path(path).expanduser()
                if path.exists() and platform in INGEST_MAP:
                    ingester = INGEST_MAP[platform](min_messages=config.min_messages)
                    convs = ingester.ingest(path)
                    total_convs += len(convs)
                    console.print(f"  {platform}: {len(convs)} conversations")
            est_cost = total_convs * 0.003  # ~$0.003 per conversation with Sonnet
            console.print(f"\nEstimated LLM cost: ~${est_cost:.2f} ({total_convs} conversations × ~$0.003/conv)")
        return

    # Run pipeline
    console.print("[bold green]Digital Brain Pipeline[/bold green]")
    console.print(f"Vault: {config.vault_path}")
    console.print(f"Sources: {', '.join(config.source_dirs.keys())}")
    console.print(f"Extraction mode: {config.llm.extraction_mode.value}")
    if config.llm.extraction_mode != ExtractionMode.RULES_ONLY:
        console.print(f"Provider: {config.llm.provider_name}")
        console.print(f"Budget: ${config.llm.max_cost_per_run_usd:.2f}")
    console.print()

    pipeline = Pipeline(config)
    result = pipeline.run()

    console.print()
    console.print(f"[bold green]{result.summary}[/bold green]")


if __name__ == "__main__":
    main()
