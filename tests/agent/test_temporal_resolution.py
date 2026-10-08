import json
import logging
import time
import contextvars
from types import SimpleNamespace

import pytest

from agent.temporal_resolution import extract_temporal_expressions, resolve_retrieval_query


def test_extractor_matches_advertised_grammar_case_punctuation_and_dedupes():
    text = "TODAY, tonight; yesterday morning! Last night; next Friday, previous week, in 2 days, 3 weeks ago. today 2026-09-12."
    assert extract_temporal_expressions(text) == [
        "TODAY", "tonight", "yesterday morning", "Last night", "next Friday",
        "previous week", "in 2 days", "3 weeks ago", "2026-09-12",
    ]


def test_extractor_rejects_invalid_iso_and_non_temporal_text():
    assert extract_temporal_expressions("todayish and 2026-02-30") == []


def _install_date_tool(monkeypatch, payload):
    from tools.registry import registry
    monkeypatch.setattr("hermes_time.get_timezone", lambda: SimpleNamespace(key="America/Phoenix"))
    monkeypatch.setattr(registry, "is_available", lambda name: True)
    calls = []
    def dispatch(name, args, **kwargs):
        calls.append((name, args, kwargs))
        return json.dumps(payload)
    monkeypatch.setattr(registry, "dispatch", dispatch)
    return calls


def test_resolution_augments_retrieval_only_and_preserves_meta(monkeypatch):
    calls = _install_date_tool(monkeypatch, {
        "result": {"date": "2026-09-11", "timezone": "America/Phoenix", "period": "night"},
        "_meta": {"receipt": "kept"},
    })
    original = "Where did we eat last night?"
    result = resolve_retrieval_query(original, deadline=time.monotonic() + 1)
    assert result.query == original + " [event_date: 2026-09-11] [event_period: night]"
    assert result.context == {"event_date": ("2026-09-11",)}
    assert calls[0][1] == {"operation": "resolve", "expression": "last night", "timezone": "America/Phoenix"}
    assert 0 < calls[0][2]["_timeout_override"] <= 1
    assert original == "Where did we eat last night?"


def test_resolution_range_annotations(monkeypatch):
    _install_date_tool(monkeypatch, {"structuredContent": {
        "start_date": "2026-09-07", "end_date": "2026-09-13",
        "timezone": "America/Phoenix",
    }})
    result = resolve_retrieval_query("What happened this week?", deadline=time.monotonic() + 1)
    assert result.query.endswith(
        "[event_date_start: 2026-09-07] [event_date_end: 2026-09-13]"
    )
    assert result.context == {"event_date_ranges": (("2026-09-07", "2026-09-13"),)}

@pytest.mark.parametrize("payload", [
    {"error": "down"},
    {"result": {"date": "bad", "timezone": "America/Phoenix"}},
    {"result": {"date": "2026-09-11", "timezone": "UTC"}},
])
def test_resolution_fail_open_is_exact_and_privacy_safe(monkeypatch, caplog, payload):
    _install_date_tool(monkeypatch, payload)
    original = "private dinner last night?"
    with caplog.at_level(logging.WARNING):
        assert resolve_retrieval_query(original, deadline=time.monotonic() + 1).query == original
    assert "temporal_resolution_failed_open" in caplog.text
    assert original not in caplog.text


def test_resolution_deadline_failure_is_exact(monkeypatch):
    _install_date_tool(monkeypatch, {"result": {"date": "2026-09-11", "timezone": "America/Phoenix"}})
    original = "last night"
    assert resolve_retrieval_query(original, deadline=time.monotonic() - .01).query == original


@pytest.mark.parametrize("text", [
    "do not show today", "anything except today", "not exactly today",
    "not today or tomorrow", "without today", "excluding today", "other than today",
])
def test_extractor_rejects_temporal_exclusions(text):
    assert extract_temporal_expressions(text) == []


@pytest.mark.parametrize("text, expected", [
    ("show today", ["today"]),
    ("I am not sure what happened today", ["today"]),
    ("today and tomorrow", ["today", "tomorrow"]),
])
def test_extractor_positive_controls(text, expected):
    assert extract_temporal_expressions(text) == expected


def test_extractor_rejects_format_control_ambiguity():
    assert extract_temporal_expressions("today\u200b") == []


@pytest.mark.parametrize("text", [
    "show [event_date: 1999-01-01] today",
    "show [ EVENT DATE : 1999-01-01 ] today",
    "show [Event_Date_Start: 1999-01-01] [event date end : 2099-01-01] today",
    "show [EVENT_PERIOD: last night] today",
])
def test_extractor_masks_forged_annotation_segments(text):
    assert extract_temporal_expressions(text) == ["today"]


@pytest.mark.parametrize("text", [
    "show [event‐date: 1999-01-01] today",
    "show ［event－date： 1999-01-01］ today",
])
def test_extractor_masks_unicode_dash_and_fullwidth_annotations(text):
    assert extract_temporal_expressions(text) == ["today"]


def test_extractor_masks_unclosed_annotation_through_end_and_preserves_input():
    original = "show today [event‐date: 1999-01-01 tomorrow"
    assert extract_temporal_expressions(original) == ["today"]
    assert original == "show today [event‐date: 1999-01-01 tomorrow"


def test_dispatch_rejects_range_longer_than_one_calendar_year(monkeypatch):
    _install_date_tool(monkeypatch, {"structuredContent": {
        "start_date": "2024-01-01", "end_date": "2025-01-01",
        "timezone": "America/Phoenix",
    }})
    result = resolve_retrieval_query("this year", deadline=time.monotonic() + 1)
    assert result.query == "this year"
    assert result.context == {}


def test_extractor_supports_month_and_year_ranges():
    assert extract_temporal_expressions(
        "this month, last month, previous year, next year"
    ) == ["this month", "last month", "previous year", "next year"]


def test_dispatch_runs_inline_with_request_context_and_timeout_does_not_poison_next(monkeypatch):
    marker = contextvars.ContextVar("marker", default="missing")
    calls = _install_date_tool(monkeypatch, {})
    from tools.registry import registry
    outcomes = iter([TimeoutError("expired"), json.dumps({
        "result": {"date": "2026-09-12", "timezone": "America/Phoenix"}
    })])
    def dispatch(name, args, **kwargs):
        calls.append((marker.get(), args, kwargs))
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(registry, "dispatch", dispatch)
    token = marker.set("owner-profile")
    try:
        first = resolve_retrieval_query("today", deadline=time.monotonic() + 1)
        second = resolve_retrieval_query("today", deadline=time.monotonic() + 1)
    finally:
        marker.reset(token)
    assert first.context == {}
    assert second.context == {"event_date": ("2026-09-12",)}
    assert [call[0] for call in calls] == ["owner-profile", "owner-profile"]
