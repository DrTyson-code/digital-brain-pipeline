"""Tests for CalendarIngester."""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.ingest.calendar import (
    CalendarIngester,
    _classify_event,
    _extract_action_items,
    _extract_event_title,
    _format_event_content,
    _is_meeting_content,
    _parse_gcal_dt,
    _sanitize_filename,
)
from src.models.base import Platform


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

# Minimal single-event export (plain list)
SINGLE_EVENT_LIST = [
    {
        "id": "evt-001",
        "summary": "Team Standup",
        "status": "confirmed",
        "start": {"dateTime": "2026-04-12T09:00:00+00:00"},
        "end":   {"dateTime": "2026-04-12T09:30:00+00:00"},
        "attendees": [
            {"email": "alice@example.com", "displayName": "Alice"},
            {"email": "bob@example.com",   "displayName": "Bob"},
        ],
        "description": "Daily standup.\n- [ ] Review blockers",
    }
]

# Google API / Takeout envelope format
TAKEOUT_FORMAT = {
    "kind": "calendar#events",
    "summary": "My Calendar",
    "items": SINGLE_EVENT_LIST,
}

# Two events on different days
MULTI_DAY_EVENTS = [
    {
        "id": "evt-day1",
        "summary": "Morning Meeting",
        "status": "confirmed",
        "start": {"dateTime": "2026-04-10T10:00:00+00:00"},
        "end":   {"dateTime": "2026-04-10T11:00:00+00:00"},
    },
    {
        "id": "evt-day2",
        "summary": "Afternoon Call",
        "status": "confirmed",
        "start": {"dateTime": "2026-04-11T14:00:00+00:00"},
        "end":   {"dateTime": "2026-04-11T15:00:00+00:00"},
    },
]

# All-day event
ALL_DAY_EVENT = [
    {
        "id": "evt-allday",
        "summary": "Company Holiday",
        "status": "confirmed",
        "start": {"date": "2026-04-13"},
        "end":   {"date": "2026-04-14"},
    }
]

# Cancelled events should be skipped
CANCELLED_EVENT = [
    {
        "id": "evt-cancelled",
        "summary": "Cancelled Meeting",
        "status": "cancelled",
        "start": {"dateTime": "2026-04-12T09:00:00+00:00"},
        "end":   {"dateTime": "2026-04-12T10:00:00+00:00"},
    }
]

# Event with no attendees (personal entry)
PERSONAL_EVENT = [
    {
        "id": "evt-personal",
        "summary": "Gym workout",
        "status": "confirmed",
        "start": {"dateTime": "2026-04-12T06:00:00+00:00"},
        "end":   {"dateTime": "2026-04-12T07:00:00+00:00"},
    }
]


def _write_json(data) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def test_parse_gcal_dt_datetime():
    dt = _parse_gcal_dt({"dateTime": "2026-04-12T10:00:00+00:00"})
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 4
    assert dt.day == 12
    assert dt.hour == 10


def test_parse_gcal_dt_zulu():
    dt = _parse_gcal_dt({"dateTime": "2026-04-12T10:00:00Z"})
    assert dt is not None
    assert dt.hour == 10


def test_parse_gcal_dt_date_only():
    dt = _parse_gcal_dt({"date": "2026-04-13"})
    assert dt is not None
    assert dt.date() == date(2026, 4, 13)


def test_parse_gcal_dt_empty():
    assert _parse_gcal_dt({}) is None
    assert _parse_gcal_dt(None) is None


def test_classify_event_meeting():
    assert _classify_event("Weekly Standup") == "meeting"
    assert _classify_event("1:1 with Alice") == "meeting"


def test_classify_event_appointment():
    assert _classify_event("Dr. Smith checkup") == "appointment"


def test_classify_event_travel():
    assert _classify_event("Flight to NYC") == "travel"


def test_classify_event_personal():
    assert _classify_event("Birthday dinner") == "personal"


def test_classify_event_deadline():
    assert _classify_event("Project deadline") == "deadline"


def test_classify_event_fallback():
    assert _classify_event("Miscellaneous thing") == "event"


def test_extract_action_items():
    text = "Discuss roadmap.\n- [ ] Update docs\n- [ ] Send email"
    items = _extract_action_items(text)
    assert "Update docs" in items
    assert "Send email" in items


def test_format_event_content_includes_title():
    content = _format_event_content(SINGLE_EVENT_LIST[0])
    assert "Team Standup" in content


def test_format_event_content_includes_time():
    content = _format_event_content(SINGLE_EVENT_LIST[0])
    assert "09:00" in content


def test_format_event_content_includes_attendees():
    content = _format_event_content(SINGLE_EVENT_LIST[0])
    assert "Alice" in content
    assert "Bob" in content


def test_format_event_content_includes_action_items():
    content = _format_event_content(SINGLE_EVENT_LIST[0])
    assert "- [ ] Review blockers" in content


def test_format_event_content_all_day():
    content = _format_event_content(ALL_DAY_EVENT[0])
    assert "all day" in content


def test_is_meeting_content_true():
    content = _format_event_content(SINGLE_EVENT_LIST[0])
    assert _is_meeting_content(content) is True


def test_is_meeting_content_false_no_attendees():
    content = _format_event_content(PERSONAL_EVENT[0])
    assert _is_meeting_content(content) is False


def test_extract_event_title():
    content = _format_event_content(SINGLE_EVENT_LIST[0])
    assert _extract_event_title(content) == "Team Standup"


def test_sanitize_filename():
    assert _sanitize_filename('File: "name"') == "File name"
    assert "/" not in _sanitize_filename("a/b")


# ---------------------------------------------------------------------------
# Ingester — parse_export
# ---------------------------------------------------------------------------

def test_parse_list_format():
    ingester = CalendarIngester(min_messages=1)
    conversations = ingester.parse_export(SINGLE_EVENT_LIST)
    assert len(conversations) == 1
    conv = conversations[0]
    assert conv.id == "2026-04-12"
    assert conv.title == "2026-04-12 Schedule"
    assert conv.platform == Platform.CALENDAR
    assert conv.message_count == 1


def test_parse_takeout_format():
    ingester = CalendarIngester(min_messages=1)
    conversations = ingester.parse_export(TAKEOUT_FORMAT)
    assert len(conversations) == 1
    assert conversations[0].id == "2026-04-12"


def test_parse_multi_day():
    ingester = CalendarIngester(min_messages=1)
    conversations = ingester.parse_export(MULTI_DAY_EVENTS)
    assert len(conversations) == 2
    ids = {c.id for c in conversations}
    assert "2026-04-10" in ids
    assert "2026-04-11" in ids


def test_cancelled_events_skipped():
    ingester = CalendarIngester(min_messages=1)
    conversations = ingester.parse_export(CANCELLED_EVENT)
    assert len(conversations) == 0


def test_all_day_event_parsed():
    ingester = CalendarIngester(min_messages=1)
    conversations = ingester.parse_export(ALL_DAY_EVENT)
    assert len(conversations) == 1
    assert conversations[0].id == "2026-04-13"


def test_empty_export_returns_empty():
    ingester = CalendarIngester(min_messages=1)
    assert ingester.parse_export([]) == []
    assert ingester.parse_export({}) == []


def test_topics_classified():
    ingester = CalendarIngester(min_messages=1)
    conversations = ingester.parse_export(SINGLE_EVENT_LIST)
    assert "meeting" in conversations[0].topics


def test_message_content_has_title():
    ingester = CalendarIngester(min_messages=1)
    conversations = ingester.parse_export(SINGLE_EVENT_LIST)
    msg = conversations[0].messages[0]
    assert "Team Standup" in msg.content


def test_message_platform():
    ingester = CalendarIngester(min_messages=1)
    conversations = ingester.parse_export(SINGLE_EVENT_LIST)
    assert conversations[0].messages[0].platform == Platform.CALENDAR


def test_min_messages_filter():
    """Days with fewer events than min_messages are dropped."""
    ingester = CalendarIngester(min_messages=2)
    # SINGLE_EVENT_LIST has only 1 event → should be filtered out
    path = _write_json(SINGLE_EVENT_LIST)
    result = ingester.ingest(path)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Ingester — file ingestion
# ---------------------------------------------------------------------------

def test_ingest_from_file():
    path = _write_json(SINGLE_EVENT_LIST)
    ingester = CalendarIngester(min_messages=1)
    conversations = ingester.ingest(path)
    assert len(conversations) == 1


def test_ingest_from_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        (d / "cal1.json").write_text(json.dumps(SINGLE_EVENT_LIST))
        (d / "cal2.json").write_text(json.dumps(MULTI_DAY_EVENTS))

        ingester = CalendarIngester(min_messages=1)
        conversations = ingester.ingest(d)
        # 1 day from cal1, 2 days from cal2 = 3 total
        assert len(conversations) == 3


def test_ingest_invalid_json():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    f.write("not valid {{{")
    f.close()
    ingester = CalendarIngester(min_messages=1)
    assert ingester.ingest(Path(f.name)) == []


# ---------------------------------------------------------------------------
# Vault note writing
# ---------------------------------------------------------------------------

def test_write_vault_notes_daily_schedule(tmp_path):
    ingester = CalendarIngester(min_messages=1, vault_path=tmp_path)
    conversations = ingester.parse_export(SINGLE_EVENT_LIST)
    written = ingester.write_vault_notes(conversations)

    schedule_path = tmp_path / "Calendar" / "2026-04-12 Schedule.md"
    assert schedule_path in written
    assert schedule_path.exists()

    text = schedule_path.read_text()
    assert "type: calendar" in text
    assert "2026-04-12" in text
    assert "Team Standup" in text


def test_write_vault_notes_meeting_template(tmp_path):
    ingester = CalendarIngester(min_messages=1, vault_path=tmp_path)
    conversations = ingester.parse_export(SINGLE_EVENT_LIST)
    written = ingester.write_vault_notes(conversations)

    meeting_files = list((tmp_path / "Calendar" / "Meetings").glob("*.md"))
    assert len(meeting_files) == 1

    text = meeting_files[0].read_text()
    assert "type: meeting" in text
    assert "## Notes" in text
    assert "## Action Items" in text


def test_write_vault_notes_no_meeting_for_solo_event(tmp_path):
    ingester = CalendarIngester(min_messages=1, vault_path=tmp_path)
    conversations = ingester.parse_export(PERSONAL_EVENT)
    ingester.write_vault_notes(conversations)

    meeting_files = list((tmp_path / "Calendar" / "Meetings").glob("*.md"))
    assert len(meeting_files) == 0


def test_write_vault_notes_frontmatter_has_event_count(tmp_path):
    ingester = CalendarIngester(min_messages=1, vault_path=tmp_path)
    conversations = ingester.parse_export(SINGLE_EVENT_LIST)
    ingester.write_vault_notes(conversations)

    text = (tmp_path / "Calendar" / "2026-04-12 Schedule.md").read_text()
    assert "event_count: 1" in text


def test_write_vault_notes_no_vault_path_warns(caplog):
    import logging
    ingester = CalendarIngester(min_messages=1)
    conversations = ingester.parse_export(SINGLE_EVENT_LIST)
    with caplog.at_level(logging.WARNING):
        result = ingester.write_vault_notes(conversations)
    assert result == []
    assert "vault_path" in caplog.text.lower() or "No vault_path" in caplog.text


def test_write_vault_notes_vault_path_override(tmp_path):
    ingester = CalendarIngester(min_messages=1)  # no vault_path at init
    conversations = ingester.parse_export(SINGLE_EVENT_LIST)
    written = ingester.write_vault_notes(conversations, vault_path=tmp_path)
    assert len(written) >= 1


def test_daily_schedule_frontmatter_includes_attendees(tmp_path):
    ingester = CalendarIngester(min_messages=1, vault_path=tmp_path)
    conversations = ingester.parse_export(SINGLE_EVENT_LIST)
    ingester.write_vault_notes(conversations)

    text = (tmp_path / "Calendar" / "2026-04-12 Schedule.md").read_text()
    assert "Alice" in text or "Bob" in text
