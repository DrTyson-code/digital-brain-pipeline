#!/usr/bin/env python3
"""Query the Digital Brain RAG server from the command line.

Sends a natural-language question to the running RAG server and
pretty-prints the retrieved notes and assembled LLM context.

Usage::

    # Basic query
    python3 scripts/query_vault.py "What do I know about ketamine?"

    # More results, filtered by note type
    python3 scripts/query_vault.py "workflow automation" --top-k 10 --note-type concept

    # Filter by domain
    python3 scripts/query_vault.py "TIVA protocols" --domain medicine

    # Print only the context block (pipe into another tool)
    python3 scripts/query_vault.py "anesthesia induction" --context-only

    # Raw JSON output
    python3 scripts/query_vault.py "morning rounds" --json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Ensure repo root is importable when running the script directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _post_query(
    url: str,
    query: str,
    top_k: int,
    note_type: str | None,
    domain: str | None,
    tags: str | None,
) -> dict:
    payload = {
        "query": query,
        "top_k": top_k,
        "filters": {
            "note_type": note_type,
            "domain": domain,
            "tags": tags,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _print_results(response: dict) -> None:
    results = response.get("results", [])
    print(f"\n{'='*60}")
    print(f"  Query : {response.get('query', '')!r}")
    print(f"  Found : {response.get('total', len(results))} results")
    print(f"{'='*60}\n")

    for r in results:
        tag_line = f"  Tags  : {r['tags']}" if r.get("tags") else ""
        dom_line = f"  Domain: {r['domain']}" if r.get("domain") else ""
        snippet = r["snippet"]
        if len(snippet) > 220:
            snippet = snippet[:220].rsplit(" ", 1)[0] + "…"
        print(
            f"[{r.get('note_type') or 'note'}] {r['title']}  "
            f"(score: {r['score']:.3f})"
        )
        print(f"  Path  : {r['note_path']}")
        if dom_line:
            print(dom_line)
        if tag_line:
            print(tag_line)
        print(f"  Snip  : {snippet}")
        print()

    context = response.get("context", "")
    if context:
        print("─" * 60)
        print("LLM context (paste directly into a prompt):\n")
        print(context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the Digital Brain RAG server"
    )
    parser.add_argument("query", nargs="+", help="Natural language query")
    parser.add_argument(
        "--port", type=int, default=8742, help="RAG server port (default: 8742)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="RAG server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to retrieve"
    )
    parser.add_argument("--note-type", help="Filter by note type (e.g. concept)")
    parser.add_argument("--domain", help="Filter by domain field")
    parser.add_argument("--tags", help="Filter by tags field")
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Print only the assembled LLM context",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output raw JSON response",
    )
    args = parser.parse_args()

    query = " ".join(args.query)
    url = f"http://{args.host}:{args.port}/query"

    try:
        response = _post_query(
            url,
            query=query,
            top_k=args.top_k,
            note_type=args.note_type,
            domain=args.domain,
            tags=args.tags,
        )
    except urllib.error.URLError as exc:
        print(
            f"Error: Cannot reach RAG server at {url}\n"
            f"       Start it: python3 scripts/start_rag_server.py\n"
            f"       Details : {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output_json:
        print(json.dumps(response, indent=2))
        return

    if args.context_only:
        print(response.get("context", ""))
        return

    _print_results(response)


if __name__ == "__main__":
    main()
