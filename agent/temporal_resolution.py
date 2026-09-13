"""Retrieval-only temporal pre-resolution through the registered Date MCP."""
from __future__ import annotations

import json
import logging
import math
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

logger = logging.getLogger(__name__)
DATE_TOOL = "mcp__date__calculate_date"
_DEFAULT_TIMEOUT = 1.0
_WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_TEMPORAL_RE = re.compile(
    rf"(?<![\w-])(?:"
    rf"yesterday\s+(?:morning|afternoon|evening)|last\s+night|"
    rf"(?:this|next|previous|last)\s+(?:week|month|year|{_WEEKDAY})|"
    rf"in\s+\d+\s+(?:days?|weeks?)|\d+\s+(?:days?|weeks?)\s+ago|"
    rf"today|tonight|tomorrow|yesterday|\d{{4}}-\d{{2}}-\d{{2}}"
    rf")(?![\w-])", re.IGNORECASE,
)
_DASHES = r"\-\u2010\u2011\u2012\u2013\u2014\u2212\ufe58\ufe63\uff0d"
_ANNOTATION_RE = re.compile(
    rf"[\[（(【［]\s*event[ _{_DASHES}]?(?:date(?:[ _{_DASHES}]?(?:start|end))?|period)"
    rf"\s*[:：][^\]\)）】］\n]*(?:[\]\)）】］]|(?=\n)|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TemporalResolution:
    """Remote-search text plus trusted host-generated retrieval metadata."""

    query: str
    context: dict[str, Any]


def _has_temporal_exclusion(text: str) -> bool:
    """Detect explicit exclusion constructions; ambiguity fails closed."""
    temporal = _TEMPORAL_RE.pattern
    return bool(re.search(
        rf"(?:\bdo\s+not\b[^,.!?;:]{{0,48}}|"
        rf"\b(?:without|excluding|except)\b[^,.!?;:]{{0,32}}|"
        rf"\bother\s+than\b[^,.!?;:]{{0,32}}|"
        rf"\bnot\s+(?:exactly\s+)?){temporal}", text, re.IGNORECASE,
    ))


def extract_temporal_expressions(text: str) -> list[str]:
    """Extract unambiguous expressions from the Date MCP resolve grammar."""
    normalized_text = unicodedata.normalize("NFKC", text or "")
    if any(unicodedata.category(char) == "Cf" for char in normalized_text):
        return []
    normalized_text = _ANNOTATION_RE.sub(" ", normalized_text)
    if _has_temporal_exclusion(normalized_text):
        return []
    found: list[str] = []
    seen: set[str] = set()
    for match in _TEMPORAL_RE.finditer(normalized_text):

        expression = " ".join(match.group(0).split())
        key = expression.casefold()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", expression):
            try:
                date.fromisoformat(expression)
            except ValueError:
                continue
        if key not in seen:
            seen.add(key)
            found.append(expression)
    return found


def _configured_timeout() -> float:
    try:
        from hermes_cli.config import load_config
        config = load_config()
        memory = config.get("memory", {}) if isinstance(config, dict) else {}
        value = float(memory.get("temporal_resolution_timeout", _DEFAULT_TIMEOUT))
        if math.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    return _DEFAULT_TIMEOUT


def _parse_dispatch_result(raw: Any, timezone_name: str) -> Optional[dict]:
    try:
        envelope = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(envelope, dict) or "error" in envelope:
            return None
        payload = envelope.get("structuredContent")
        if payload is None:
            payload = envelope.get("result")
            payload = json.loads(payload) if isinstance(payload, str) else payload
        if not isinstance(payload, dict) or payload.get("timezone") != timezone_name:
            return None
        if "date" in payload:
            date.fromisoformat(payload["date"])
        elif "start_date" in payload and "end_date" in payload:
            start = date.fromisoformat(payload["start_date"])
            end = date.fromisoformat(payload["end_date"])
            if end < start or (end - start).days > 365:
                return None
        else:
            return None
        return payload
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def resolve_retrieval_query(original_query: str, *, deadline: float) -> TemporalResolution:
    """Resolve temporal terms without exposing trusted metadata to user text."""
    expressions = extract_temporal_expressions(original_query)
    if not expressions:
        return TemporalResolution(original_query, {})
    started = time.monotonic()
    try:
        from hermes_time import get_timezone
        timezone_name = getattr(get_timezone(), "key", None)
        if not timezone_name:
            raise ValueError("missing timezone")
        from tools.registry import registry
        if not registry.is_available(DATE_TOOL):
            raise LookupError("date tool unavailable")
        annotations: list[str] = []
        resolved_dates: list[str] = []
        resolved_ranges: list[tuple[str, str]] = []
        for expression in expressions:
            remaining = min(_configured_timeout(), deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError("prefetch deadline")
            raw = registry.dispatch(
                DATE_TOOL,
                {"operation": "resolve", "expression": expression, "timezone": timezone_name},
                _timeout_override=remaining,
            )
            payload = _parse_dispatch_result(raw, timezone_name)
            if payload is None:
                raise ValueError("malformed date result")
            if "date" in payload:
                annotations.append(f"[event_date: {payload['date']}]")
                resolved_dates.append(payload["date"])
                if payload.get("period"):
                    annotations.append(f"[event_period: {payload['period']}]")
            else:
                annotations.extend((
                    f"[event_date_start: {payload['start_date']}]",
                    f"[event_date_end: {payload['end_date']}]",
                ))
                resolved_ranges.append((payload["start_date"], payload["end_date"]))
        context: dict[str, Any] = {}
        if resolved_dates:
            context["event_date"] = tuple(dict.fromkeys(resolved_dates))
        if resolved_ranges:
            context["event_date_ranges"] = tuple(dict.fromkeys(resolved_ranges))
        return TemporalResolution(
            original_query + " " + " ".join(dict.fromkeys(annotations)),
            context,
        )
    except Exception as exc:
        logger.warning(
            "temporal_resolution_failed_open outcome=%s elapsed_ms=%d",
            type(exc).__name__, round((time.monotonic() - started) * 1000),
        )
        return TemporalResolution(original_query, {})
