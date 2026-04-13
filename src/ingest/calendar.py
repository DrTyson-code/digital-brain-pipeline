"""Parser for Google Calendar export format.

Accepts Google Calendar JSON exports in two shapes:

1. Google Takeout / Calendar API list response:
   {"items": [ <event>, ... ]}

2. Plain list:
   [ <event>, ... ]

Each event follows the Google Calendar API Event resource:
   {
       "id": "...",
       "summary": "Meeting Title",
       "description": "Optional body text",
       "start": {"dateTime": "2026-04-12T10:00:00-05:00"},
       "end":   {"dateTime": "2026-04-12T11:00:00-05:00"},
       "location": "Conference Room B",
       "attendees": [
           {"email": "alice@example.com", "displayName": "Alice"},
           {"email": "bob@example.com",   "displayName": "Bob"}
       ],
       "status": "confirmed"
   }

All-day events use `"date": "YYYY-MM-DD"` instead of `"dateTime"`.

Events are grouped by calendar date and returned as one Conversation per day.
Each event becomes one ChatMessage whose content is a human-readable
summary of the event details.  The ingester also exposes
`write_vault_notes()` to generate Obsidian-compatible markdown:
  - Calendar/YYYY-MM-DD Schedule.md     — daily agenda
  - Calendar/Meetings/YYYY-MM-DD Title.md — meeting notes template
    (created for any event with 2 or more attendees)
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Union

import yaml

from src.ingest.base import BaseIngester
from src.models.base import Platform
from src.models.message import ChatMessage, Conversation, Role

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event-type classification
# ---------------------------------------------------------------------------

_EVENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "meeting": [
        "meeting", "sync", "standup", "stand-up", "standup",
        "review", "discussion", "call", "conference", "1:1",
        "one-on-one", "retrospective", "sprint", "demo", "debrief",
    ],
    "appointment": [
        "appointment", "dr.", "doctor", "dentist", "medical",
        "checkup", "check-up", "clinic", "hospital", "therapy",
        "physical", "exam",
    ],
    "travel": [
        "flight", "travel", "trip", "departure", "arrival",
        "hotel", "train", "drive to", "layover", "boarding",
    ],
    "personal": [
        "birthday", "anniversary", "dinner", "lunch", "coffee",
        "gym", "workout", "date", "family", "vacation", "holiday",
    ],
    "deadline": [
        "deadline", "due", "submit", "submission", "delivery", "release",
    ],
}

_ACTION_ITEM_RE = re.compile(
    r"(?:"
    r"[-*•]\s*(?:TODO|Action item|Task|Follow-?up)[:\s]+(.+)"
    r"|Action items?:\s*(.+?)(?:\n|$)"
    r"|[-*•]\s*\[\s*\]\s*(.+)"   # markdown checkbox
    r")",
    re.IGNORECASE,
)


def _classify_event(title: str, description: str = "") -> str:
    """Return a coarse event-type tag based on title/description keywords."""
    text = (title + " " + description).lower()
    for event_type, keywords in _EVENT_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return event_type
    return "event"


def _extract_action_items(text: str) -> list[str]:
    """Pull action items from free-text event description."""
    items: list[str] = []
    for m in _ACTION_ITEM_RE.finditer(text):
        item = next((g for g in m.groups() if g), "").strip()
        if item:
            items.append(item)
    return items


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

def _parse_gcal_dt(dt_dict: dict) -> Optional[datetime]:
    """Parse a Google Calendar start/end dict into a datetime.

    Handles:
    - {"dateTime": "2026-04-12T10:00:00-05:00"}
    - {"date": "2026-04-12"}  (all-day)
    """
    if not dt_dict:
        return None
    raw = dt_dict.get("dateTime") or dt_dict.get("date")
    if not raw:
        return None
    try:
        if "T" not in raw:
            # All-day: treat as midnight UTC
            return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


def _event_date(event: dict) -> Optional[date]:
    dt = _parse_gcal_dt(event.get("start", {}))
    return dt.date() if dt else None


# ---------------------------------------------------------------------------
# Content formatting helpers
# ---------------------------------------------------------------------------

def _format_event_content(event: dict) -> str:
    """Render a calendar event as a human-readable string."""
    lines: list[str] = []

    title = event.get("summary", "Untitled Event")
    start_dt = _parse_gcal_dt(event.get("start", {}))
    end_dt = _parse_gcal_dt(event.get("end", {}))

    is_all_day = "dateTime" not in event.get("start", {})
    if not is_all_day and start_dt and end_dt:
        time_range = f"{start_dt.strftime('%H:%M')}–{end_dt.strftime('%H:%M')}"
        lines.append(f"**{title}** ({time_range})")
    else:
        lines.append(f"**{title}** (all day)")

    location = (event.get("location") or "").strip()
    if location:
        lines.append(f"Location: {location}")

    attendees = event.get("attendees", [])
    if attendees:
        names = [
            a.get("displayName") or a.get("email", "")
            for a in attendees
            if isinstance(a, dict)
        ]
        names = [n for n in names if n]
        if names:
            lines.append(f"Attendees: {', '.join(names)}")

    description = (event.get("description") or "").strip()
    if description:
        lines.append("")
        lines.append(description)

    action_items = _extract_action_items(description)
    if action_items:
        lines.append("")
        lines.append("Action items:")
        for item in action_items:
            lines.append(f"- [ ] {item}")

    return "\n".join(lines)


def _extract_event_title(content: str) -> str:
    """Pull the event title back out of a formatted content string."""
    for line in content.splitlines():
        m = re.match(r"\*\*(.+?)\*\*", line)
        if m:
            return m.group(1)
    return "Meeting"


def _is_meeting_content(content: str) -> bool:
    """True if the event has 2 or more attendees (qualifies for a meeting note)."""
    for line in content.splitlines():
        if line.startswith("Attendees:"):
            attendees = [a.strip() for a in line[len("Attendees:"):].split(",") if a.strip()]
            return len(attendees) >= 2
    return False


def _sanitize_filename(name: str) -> str:
    for ch in r'<>:"/\|?*':
        name = name.replace(ch, "")
    return name.strip()[:100]


# ---------------------------------------------------------------------------
# Main ingester
# ---------------------------------------------------------------------------

class CalendarIngester(BaseIngester):
    """Parse Google Calendar JSON exports into daily Conversation objects.

    Each calendar day with at least one event becomes one Conversation.
    Each event within a day becomes one ChatMessage.

    Args:
        min_messages: Minimum events per day to include (default 1).
        vault_path: Optional vault root for ``write_vault_notes()``.
    """

    platform_name = "calendar"

    def __init__(
        self,
        min_messages: int = 1,
        vault_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(min_messages=min_messages)
        self.vault_path = Path(vault_path).expanduser() if vault_path else None

    # ------------------------------------------------------------------
    # BaseIngester interface
    # ------------------------------------------------------------------

    def parse_export(
        self,
        data: Union[dict, list],
        source_file: Optional[Path] = None,
    ) -> list[Conversation]:
        """Parse calendar JSON data into per-day Conversation objects."""
        events = self._extract_events(data)
        if not events:
            logger.debug("No calendar events found in export")
            return []

        # Group by date, skipping cancelled events
        by_date: dict[date, list[dict]] = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("status") == "cancelled":
                continue
            ev_date = _event_date(event)
            if ev_date is None:
                continue
            by_date.setdefault(ev_date, []).append(event)

        conversations: list[Conversation] = []
        for day in sorted(by_date):
            conv = self._build_day_conversation(day, by_date[day])
            if conv:
                conversations.append(conv)

        logger.info(
            "calendar: parsed %d day(s) from %s",
            len(conversations),
            source_file or "data",
        )
        return conversations

    # ------------------------------------------------------------------
    # Vault note writing
    # ------------------------------------------------------------------

    def write_vault_notes(
        self,
        conversations: list[Conversation],
        vault_path: Optional[Union[str, Path]] = None,
    ) -> list[Path]:
        """Write calendar-specific Obsidian notes.

        Creates:
        - ``<vault>/Calendar/YYYY-MM-DD Schedule.md`` per day
        - ``<vault>/Calendar/Meetings/YYYY-MM-DD Title.md`` for events
          with 2+ attendees

        Returns the list of paths that were written.
        """
        vp = Path(vault_path).expanduser() if vault_path else self.vault_path
        if not vp:
            logger.warning("No vault_path set; skipping calendar note writing")
            return []

        calendar_dir = vp / "Calendar"
        meetings_dir = calendar_dir / "Meetings"
        calendar_dir.mkdir(parents=True, exist_ok=True)
        meetings_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for conv in conversations:
            path = self._write_daily_schedule(conv, calendar_dir)
            if path:
                written.append(path)
            for msg in conv.messages:
                if _is_meeting_content(msg.content):
                    path = self._write_meeting_note(conv.id, msg, meetings_dir)
                    if path:
                        written.append(path)

        logger.info("Wrote %d calendar vault notes to %s", len(written), vp)
        return written

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_events(data: Union[dict, list]) -> list[dict]:
        """Normalise various JSON shapes into a flat list of event dicts."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Google API / Takeout: {"items": [...]}
            for key in ("items", "events"):
                if key in data:
                    value = data[key]
                    if isinstance(value, list):
                        return value
        return []

    def _build_day_conversation(
        self, day: date, events: list[dict]
    ) -> Optional[Conversation]:
        day_str = day.strftime("%Y-%m-%d")
        day_dt = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)

        def _sort_key(e: dict) -> datetime:
            dt = _parse_gcal_dt(e.get("start", {}))
            return dt if dt else datetime.min.replace(tzinfo=timezone.utc)

        messages: list[ChatMessage] = []
        topics: list[str] = []

        for event in sorted(events, key=_sort_key):
            content = _format_event_content(event)
            event_type = _classify_event(
                event.get("summary", ""),
                event.get("description", ""),
            )
            if event_type not in topics:
                topics.append(event_type)

            # Stable ID: hash of (gcal event id, date) to avoid collisions
            raw_id = str(event.get("id") or event.get("iCalUID") or content)
            msg_id = hashlib.md5(f"{raw_id}:{day_str}".encode()).hexdigest()[:12]

            messages.append(
                ChatMessage(
                    id=msg_id,
                    conversation_id=day_str,
                    role=Role.USER,
                    content=content,
                    timestamp=_parse_gcal_dt(event.get("start", {})),
                    platform=Platform.CALENDAR,
                )
            )

        if not messages:
            return None

        return Conversation(
            id=day_str,
            title=f"{day_str} Schedule",
            messages=messages,
            platform=Platform.CALENDAR,
            created_at=day_dt,
            updated_at=day_dt,
            topics=topics,
        )

    def _write_daily_schedule(
        self, conv: Conversation, calendar_dir: Path
    ) -> Optional[Path]:
        try:
            day_str = conv.id
            all_attendees: list[str] = []
            for msg in conv.messages:
                for line in msg.content.splitlines():
                    if line.startswith("Attendees:"):
                        all_attendees.extend(
                            a.strip()
                            for a in line[len("Attendees:"):].split(",")
                            if a.strip()
                        )

            frontmatter = {
                "type": "calendar",
                "date": day_str,
                "event_count": conv.message_count,
                "attendees": sorted(set(all_attendees)),
                "tags": [f"calendar/{t}" for t in (conv.topics or [])],
            }

            lines = [
                "---",
                yaml.dump(frontmatter, default_flow_style=False).rstrip(),
                "---",
                "",
                f"# {day_str} Schedule",
                "",
            ]
            for msg in conv.messages:
                lines.append(msg.content)
                lines.append("")

            note_path = calendar_dir / f"{day_str} Schedule.md"
            note_path.write_text("\n".join(lines), encoding="utf-8")
            return note_path
        except Exception as exc:
            logger.warning("Failed to write daily schedule for %s: %s", conv.id, exc)
            return None

    def _write_meeting_note(
        self, day_str: str, msg: ChatMessage, meetings_dir: Path
    ) -> Optional[Path]:
        try:
            title = _extract_event_title(msg.content)
            safe_title = _sanitize_filename(title)

            attendees: list[str] = []
            agenda_lines: list[str] = []
            for line in msg.content.splitlines():
                if line.startswith("Attendees:"):
                    attendees = [a.strip() for a in line[len("Attendees:"):].split(",") if a.strip()]
                elif not line.startswith("**") and not line.startswith("Location:"):
                    if line.strip():
                        agenda_lines.append(line)

            frontmatter = {
                "type": "meeting",
                "date": day_str,
                "attendees": attendees,
                "tags": ["calendar/meeting"],
            }

            lines = [
                "---",
                yaml.dump(frontmatter, default_flow_style=False).rstrip(),
                "---",
                "",
                f"# {title}",
                "",
                "## Agenda",
                "",
            ]
            if agenda_lines:
                lines.extend(agenda_lines)
            else:
                lines.append("_(No agenda provided)_")
            lines.extend([
                "",
                "## Notes",
                "",
                "",
                "## Action Items",
                "",
                "- [ ] ",
                "",
            ])

            note_path = meetings_dir / f"{day_str} {safe_title}.md"
            note_path.write_text("\n".join(lines), encoding="utf-8")
            return note_path
        except Exception as exc:
            logger.warning("Failed to write meeting note for %s: %s", day_str, exc)
            return None
