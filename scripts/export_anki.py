#!/usr/bin/env python3
"""CLI tool for exporting Anki flashcards from the Digital Brain Pipeline.

Scans the Obsidian vault for notes that are candidates for Anki review,
generates flashcards, deduplicates against the export history, and writes
an Anki-importable file.

Usage examples:
    # Export all eligible vault notes (TSV, default batch of 50)
    python scripts/export_anki.py

    # Export only stale/decaying notes
    python scripts/export_anki.py --stale-only

    # Export medicine domain notes as .apkg (requires genanki)
    python scripts/export_anki.py --domain medicine --format apkg --output data/medicine.apkg

    # Dry run: preview what would be exported
    python scripts/export_anki.py --dry-run --batch-size 100

    # Export from a specific vault with custom output
    python scripts/export_anki.py --vault-path ~/Vault --output ~/Desktop/review.txt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Add project root to sys.path so ``import src`` works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export.anki import AnkiCardGenerator, AnkiExporter, GENANKI_AVAILABLE
from src.export.anki_scheduler import AnkiScheduler

console = Console()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.command()
@click.option(
    "--vault-path",
    "vault_path",
    type=click.Path(path_type=Path),
    default=Path("~/Vault"),
    show_default=True,
    help="Path to the Obsidian vault to scan.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=Path("data/anki_export.txt"),
    show_default=True,
    help="Output file path.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["tsv", "apkg"], case_sensitive=False),
    default="tsv",
    show_default=True,
    help="Export format.  'apkg' requires the genanki package.",
)
@click.option(
    "--domain",
    "domains",
    multiple=True,
    help=(
        "Filter cards to this domain (can be repeated for multiple domains).  "
        "Valid domains: medicine, reef_keeping, business, fitness, finance, "
        "technology, personal."
    ),
)
@click.option(
    "--stale-only",
    "stale_only",
    is_flag=True,
    default=False,
    help="Only export notes flagged as stale/decaying (status: stale).",
)
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=50,
    show_default=True,
    help="Maximum number of cards to export in one run.",
)
@click.option(
    "--min-confidence",
    "min_confidence",
    type=float,
    default=0.7,
    show_default=True,
    help="Minimum confidence threshold for including a note (0.0–1.0).",
)
@click.option(
    "--db-path",
    "db_path",
    type=click.Path(path_type=Path),
    default=Path("data/anki_exports.db"),
    show_default=True,
    help="Path to the SQLite export history database.",
)
@click.option(
    "--no-dedup",
    "skip_dedup",
    is_flag=True,
    default=False,
    help="Skip deduplication — export all matching cards even if previously exported.",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Preview cards without writing any files.",
)
@click.option(
    "--history",
    "show_history",
    is_flag=True,
    default=False,
    help="Show recent export run history and exit.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging level.",
)
def main(
    vault_path: Path,
    output_path: Path,
    fmt: str,
    domains: tuple[str, ...],
    stale_only: bool,
    batch_size: int,
    min_confidence: float,
    db_path: Path,
    skip_dedup: bool,
    dry_run: bool,
    show_history: bool,
    log_level: str,
) -> None:
    """Export Anki flashcards from the Digital Brain vault.

    Scans the vault for eligible notes, generates cards, deduplicates against
    previous exports, and writes an Anki-importable file.
    """
    setup_logging(log_level)

    scheduler = AnkiScheduler(db_path=db_path, batch_size=batch_size)

    # ------------------------------------------------------------------
    # History mode
    # ------------------------------------------------------------------
    if show_history:
        history = scheduler.export_history(limit=10)
        if not history:
            console.print("[yellow]No export history found.[/yellow]")
            return

        table = Table(title="Recent Anki Export Runs")
        table.add_column("Run ID", style="dim")
        table.add_column("Started", style="cyan")
        table.add_column("Cards", justify="right", style="green")
        table.add_column("Format")
        table.add_column("Output")

        for run in history:
            table.add_row(
                run["run_id"],
                run["started_at"][:19],
                str(run["card_count"]),
                run["format"] or "—",
                run["output_path"] or "—",
            )
        console.print(table)
        console.print(
            f"\nTotal exported sources tracked: "
            f"[bold]{scheduler.exported_count()}[/bold]"
        )
        return

    # ------------------------------------------------------------------
    # Validate options
    # ------------------------------------------------------------------
    vault_path = Path(vault_path).expanduser()
    output_path = Path(output_path)

    if fmt == "apkg" and not GENANKI_AVAILABLE:
        console.print(
            "[yellow]Warning:[/yellow] genanki is not installed — "
            "falling back to TSV format.\n"
            "Install with: [bold]pip install genanki[/bold]"
        )
        fmt = "tsv"
        if not output_path.suffix == ".txt":
            output_path = output_path.with_suffix(".txt")

    domain_filter = list(domains) if domains else None

    # ------------------------------------------------------------------
    # Generate cards
    # ------------------------------------------------------------------
    console.print("[bold green]Anki Flashcard Export[/bold green]")
    console.print(f"Vault:       {vault_path}")
    console.print(f"Output:      {output_path}")
    console.print(f"Format:      {fmt}")
    console.print(f"Batch size:  {batch_size}")
    console.print(f"Min conf:    {min_confidence}")
    if domain_filter:
        console.print(f"Domains:     {', '.join(domain_filter)}")
    if stale_only:
        console.print("[yellow]Mode: stale notes only[/yellow]")
    if dry_run:
        console.print("[yellow]Dry run — no files will be written[/yellow]")
    console.print()

    generator = AnkiCardGenerator(min_confidence=min_confidence)
    cards = generator.generate_from_vault(
        vault_path,
        domain_filter=domain_filter,
        stale_only=stale_only,
    )

    if not cards:
        console.print("[yellow]No eligible cards found in vault.[/yellow]")
        return

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    total_before = len(cards)
    if not skip_dedup:
        cards = scheduler.filter_new(cards)
        skipped = total_before - len(cards)
        if skipped:
            console.print(
                f"[dim]Skipped {skipped} previously-exported card(s)[/dim]"
            )

    if not cards:
        console.print("[green]All eligible cards already exported — nothing new.[/green]")
        return

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    stale_count = sum(1 for c in cards if "review::urgent" in c.tags)
    console.print(
        f"Cards to export: [bold]{len(cards)}[/bold] "
        f"([yellow]{stale_count} urgent[/yellow] / "
        f"{len(cards) - stale_count} regular)"
    )

    if dry_run:
        _print_preview(cards)
        return

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    run_id = scheduler.start_run(fmt, output_path)
    exporter = AnkiExporter()
    written_path = exporter.export(cards, output_path, fmt=fmt)
    scheduler.complete_run(run_id, len(cards))
    scheduler.mark_exported_batch(cards)

    console.print(
        f"\n[bold green]Done![/bold green] "
        f"Wrote [bold]{len(cards)}[/bold] card(s) to [cyan]{written_path}[/cyan]"
    )
    console.print(
        f"\nImport into Anki: [bold]File → Import → {written_path}[/bold]"
    )


def _print_preview(cards) -> None:
    """Print a table preview of cards that would be exported."""
    from rich.table import Table

    table = Table(title=f"Preview — {len(cards)} card(s)", show_lines=True)
    table.add_column("Type", width=6)
    table.add_column("Deck", width=25)
    table.add_column("Front", width=45)
    table.add_column("Tags", width=30)

    for card in cards[:20]:
        tag_str = " ".join(card.sanitized_tags[:4])
        table.add_row(
            card.card_type[:5],
            card.deck,
            card.front[:44],
            tag_str[:29],
        )

    if len(cards) > 20:
        table.add_row("...", "...", f"... and {len(cards) - 20} more", "")

    console.print(table)


if __name__ == "__main__":
    main()
