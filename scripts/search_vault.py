#!/usr/bin/env python3
"""Semantic vault search CLI.

Searches the Digital Brain Vault by *meaning*, not just keywords,
using a TF-IDF index stored in SQLite.

Examples
--------
# First-time setup — build the index:
    python3 scripts/search_vault.py --index-only

# Search:
    python3 scripts/search_vault.py "coding patterns for clinical workflows"
    python3 scripts/search_vault.py "Python error handling" --top 5 --type concept
    python3 scripts/search_vault.py "important decisions" --type decision --top 20

# Force full rebuild (e.g. after a pipeline run):
    python3 scripts/search_vault.py "your query" --rebuild
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# Allow running directly from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.search.embedder import NoteEmbedder  # noqa: E402
from src.search.engine import VaultSearchEngine  # noqa: E402

console = Console()

_DEFAULT_VAULT = str(Path("~/Desktop/claude-vault-output").expanduser())
_DEFAULT_DB = ".dbp_cache/embeddings.db"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("query", default="")
@click.option(
    "--vault", "-v",
    default=_DEFAULT_VAULT,
    show_default=True,
    help="Path to the Obsidian vault to index and search.",
)
@click.option(
    "--db",
    default=_DEFAULT_DB,
    show_default=True,
    help="Path to the SQLite embeddings database.",
)
@click.option(
    "--top", "-n",
    default=10,
    show_default=True,
    help="Number of results to return.",
)
@click.option(
    "--type", "note_type",
    default=None,
    help="Filter by note type: entity | concept | decision | conversation | action | contact | project",
)
@click.option(
    "--tags",
    default=None,
    help="Filter notes whose tags contain this substring.",
)
@click.option(
    "--rebuild",
    is_flag=True,
    help="Force-rebuild the entire embedding index before searching.",
)
@click.option(
    "--index-only",
    is_flag=True,
    help="Build/update the index without performing a search.",
)
@click.option(
    "--verbose", "-V",
    is_flag=True,
    help="Enable debug logging.",
)
def main(
    query: str,
    vault: str,
    db: str,
    top: int,
    note_type: str | None,
    tags: str | None,
    rebuild: bool,
    index_only: bool,
    verbose: bool,
) -> None:
    """Search the Digital Brain vault by meaning.

    QUERY is a natural-language search string.
    Leave QUERY empty when using --index-only.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    vault_path = Path(vault).expanduser()
    db_path = Path(db)

    if not vault_path.exists():
        console.print(f"[bold red]Error:[/bold red] Vault not found at {vault_path}")
        console.print("  Pass --vault <path> or update config/settings.yaml")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Step 1: Build / update embedding index
    # ------------------------------------------------------------------
    needs_index = rebuild or index_only or not db_path.exists()

    if needs_index:
        action = "Rebuilding" if rebuild else "Building"
        with console.status(f"[bold cyan]{action} embedding index…[/bold cyan]"):
            embedder = NoteEmbedder(db_path=db_path)
            changed = embedder.embed_vault(vault_path, force_rebuild=rebuild)
            doc_count = embedder.get_doc_count()
            embedder.close()

        console.print(
            f"[green]✓[/green] Index ready: "
            f"[bold]{doc_count:,}[/bold] notes indexed, "
            f"[bold]{changed:,}[/bold] updated"
        )
    else:
        # Incremental update: embed any notes that have changed since last run
        with console.status("[bold cyan]Checking for changed notes…[/bold cyan]"):
            embedder = NoteEmbedder(db_path=db_path)
            changed = embedder.embed_vault(vault_path, force_rebuild=False)
            doc_count = embedder.get_doc_count()
            embedder.close()

        if changed:
            console.print(
                f"[green]✓[/green] Index updated: "
                f"{changed:,} note(s) re-embedded ({doc_count:,} total)"
            )

    if index_only or not query:
        if not query and not index_only:
            console.print(
                "[yellow]Tip:[/yellow] provide a QUERY argument to search, "
                "or use --index-only to just update the index."
            )
        return

    # ------------------------------------------------------------------
    # Step 2: Search
    # ------------------------------------------------------------------
    engine = VaultSearchEngine(db_path=db_path)

    with console.status("[bold cyan]Loading index into memory…[/bold cyan]"):
        n_loaded = engine.load()

    with console.status(
        f"[bold cyan]Searching {n_loaded:,} notes…[/bold cyan]"
    ):
        results = engine.search(query, top_n=top, note_type=note_type, tags=tags)

    # ------------------------------------------------------------------
    # Step 3: Display results
    # ------------------------------------------------------------------
    filter_desc = ""
    if note_type:
        filter_desc += f"  type={note_type}"
    if tags:
        filter_desc += f"  tags~{tags!r}"

    console.print()
    console.print(
        Panel(
            f'[bold white]"{query}"[/bold white]',
            title="[cyan]Semantic Search[/cyan]",
            subtitle=f"[dim]{len(results)} results from {n_loaded:,} notes{filter_desc}[/dim]",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()

    if not results:
        console.print("[yellow]No matching notes found.[/yellow]")
        console.print(
            "[dim]Try broadening your query, or run with --rebuild if the "
            "vault has changed.[/dim]"
        )
        return

    for i, r in enumerate(results, 1):
        # Colour-code the similarity score
        if r.similarity >= 0.25:
            score_style = "bold green"
        elif r.similarity >= 0.10:
            score_style = "yellow"
        else:
            score_style = "dim"

        # Header line: rank  title  score  type
        header = Text()
        header.append(f"{i:2}. ", style="bold white")
        header.append(r.title, style="bold cyan")
        header.append(f"  [{r.similarity:.3f}]", style=score_style)
        if r.note_type:
            header.append(f"  {r.note_type}", style="dim magenta")
        console.print(header)

        # Tags
        if r.tags:
            console.print(f"    [dim]tags:[/dim] [dim]{r.tags}[/dim]")

        # Snippet
        if r.snippet:
            console.print(f"    [dim italic]{r.snippet}[/dim italic]")

        # File path
        console.print(f"    [dim blue]{r.note_path}[/dim blue]")
        console.print()


if __name__ == "__main__":
    main()
