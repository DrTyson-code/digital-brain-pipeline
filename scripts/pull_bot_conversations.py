#!/usr/bin/env python3
"""Pull fitness bot conversations from Railway and write to claude-export directory.

Reads BOT_URL and WEBHOOK_SECRET_TOKEN from environment.
Writes ~/code/claude-export/fitness_bot_conversations.json in Claude export format.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


def main() -> int:
    bot_url = os.environ.get("BOT_URL", "").rstrip("/")
    token = os.environ.get("WEBHOOK_SECRET_TOKEN", "")

    if not bot_url:
        logger.error("BOT_URL not set in environment — skipping bot conversation pull")
        return 0  # non-fatal: pipeline continues without bot conversations

    if not token:
        logger.error("WEBHOOK_SECRET_TOKEN not set — skipping bot conversation pull")
        return 0

    url = f"{bot_url}/export/{token}/conversations"
    logger.info(f"Pulling bot conversations from {bot_url}/export/<token>/conversations")

    try:
        with urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        logger.error(f"HTTP {e.code} fetching bot conversations: {e.reason}")
        return 0  # non-fatal
    except URLError as e:
        logger.error(f"Could not reach bot at {bot_url}: {e.reason}")
        return 0  # non-fatal

    out_path = Path("~/code/claude-export/fitness_bot_conversations.json").expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote {len(data)} bot conversations to {out_path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
