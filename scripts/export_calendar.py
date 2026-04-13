#!/usr/bin/env python3
"""Export Google Calendar events to a JSON file for pipeline ingestion.

This script fetches events from your Google Calendar using the Calendar API
and writes them to a JSON file that CalendarIngester can process.

Prerequisites
-------------
1. Enable the Google Calendar API in Google Cloud Console.
2. Create OAuth 2.0 credentials (Desktop app) and download as
   ``credentials.json`` (or set ``--credentials`` to its path).
3. Install dependencies:
       pip install google-api-python-client google-auth-oauthlib

Usage
-----
First run (will open browser for OAuth consent):
    python scripts/export_calendar.py

Subsequent runs use a cached token:
    python scripts/export_calendar.py

Options:
    --output PATH       Where to write the JSON (default: calendar_export.json)
    --credentials PATH  Path to credentials.json (default: credentials.json)
    --token PATH        Path to cached token file (default: token.json)
    --calendar-id ID    Calendar to export (default: primary)
    --days-back N       How many past days to include (default: 90)
    --days-forward N    How many future days to include (default: 30)
    --no-single-events  Do not expand recurring events
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Scopes required (read-only is sufficient)
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def _get_credentials(credentials_path: Path, token_path: Path):
    """Return valid user credentials, running OAuth flow if needed."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        logger.error(
            "Missing dependencies. Install with:\n"
            "  pip install google-api-python-client google-auth-oauthlib"
        )
        sys.exit(1)

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                logger.error(
                    "credentials.json not found at %s.\n"
                    "Download it from Google Cloud Console → APIs & Services → "
                    "Credentials → OAuth 2.0 Client IDs → Download JSON.",
                    credentials_path,
                )
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), SCOPES
            )
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
        logger.info("Saved token to %s", token_path)

    return creds


def _build_service(creds):
    try:
        from googleapiclient.discovery import build
    except ImportError:
        logger.error("Install google-api-python-client: pip install google-api-python-client")
        sys.exit(1)
    return build("calendar", "v3", credentials=creds)


def fetch_events(
    service,
    calendar_id: str,
    time_min: datetime,
    time_max: datetime,
    single_events: bool = True,
) -> list[dict]:
    """Fetch all events in the given time range, handling pagination."""
    events: list[dict] = []
    page_token = None

    while True:
        result = (
            service.events()
            .list(
                calendarId=calendar_id,
                timeMin=time_min.isoformat(),
                timeMax=time_max.isoformat(),
                singleEvents=single_events,
                orderBy="startTime" if single_events else "updated",
                pageToken=page_token,
                maxResults=2500,
            )
            .execute()
        )
        events.extend(result.get("items", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            break

    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Google Calendar events to JSON for the digital-brain pipeline."
    )
    parser.add_argument(
        "--output",
        default="calendar_export.json",
        help="Output JSON path (default: calendar_export.json)",
    )
    parser.add_argument(
        "--credentials",
        default="credentials.json",
        help="Path to Google OAuth credentials.json",
    )
    parser.add_argument(
        "--token",
        default="token.json",
        help="Path to cached OAuth token (default: token.json)",
    )
    parser.add_argument(
        "--calendar-id",
        default="primary",
        help="Google Calendar ID to export (default: primary)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="Number of past days to include (default: 90)",
    )
    parser.add_argument(
        "--days-forward",
        type=int,
        default=30,
        help="Number of future days to include (default: 30)",
    )
    parser.add_argument(
        "--no-single-events",
        action="store_true",
        help="Do not expand recurring events into individual instances",
    )
    args = parser.parse_args()

    credentials_path = Path(args.credentials)
    token_path = Path(args.token)
    output_path = Path(args.output)

    now = datetime.now(tz=timezone.utc)
    time_min = now - timedelta(days=args.days_back)
    time_max = now + timedelta(days=args.days_forward)

    logger.info(
        "Fetching events from %s to %s (calendar: %s)",
        time_min.strftime("%Y-%m-%d"),
        time_max.strftime("%Y-%m-%d"),
        args.calendar_id,
    )

    creds = _get_credentials(credentials_path, token_path)
    service = _build_service(creds)

    events = fetch_events(
        service,
        calendar_id=args.calendar_id,
        time_min=time_min,
        time_max=time_max,
        single_events=not args.no_single_events,
    )

    logger.info("Fetched %d events", len(events))

    export_data = {
        "exported_at": now.isoformat(),
        "calendar_id": args.calendar_id,
        "time_min": time_min.isoformat(),
        "time_max": time_max.isoformat(),
        "items": events,
    }

    output_path.write_text(json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d events to %s", len(events), output_path)


if __name__ == "__main__":
    main()
