"""App-owned, read-only Apple Calendar tool for the Owner CLI."""
from __future__ import annotations

import json
import re
import stat
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_TOKEN_PATH = Path.home() / "Library/Application Support/Jarvis/private-executor/calendar-read-token"
_ENDPOINT = "http://192.168.150.155:8790/v1/internal/calendar/read"
_MAX_RESPONSE_BYTES = 1024 * 1024

_SCHEMA = {
    "name": "read_owner_calendar",
    "description": (
        "Read Dennis's Apple Calendar for one America/Phoenix day. Read-only. "
        "Accepts today, tomorrow, a weekday, or an explicit calendar date directly."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "date": {
                "type": "string",
                "minLength": 3,
                "maxLength": 32,
                "description": "One bounded day expression, such as today, tomorrow, Saturday, or 2026-09-19.",
            }
        },
        "required": ["date"],
    },
}


def _read_token() -> str | None:
    try:
        info = _TOKEN_PATH.stat()
        if _TOKEN_PATH.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            return None
        token = _TOKEN_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token if len(token.encode()) >= 32 else None


def _token_is_available() -> bool:
    return _read_token() is not None


def _valid_date_expression(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.strip().split())
    allowed = (
        r"today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
        r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?"
    )
    if not re.fullmatch(allowed, value, re.IGNORECASE):
        return None
    return value


def _handle_read(params: dict[str, Any], **kwargs: Any) -> str:
    expression = _valid_date_expression(params.get("date"))
    if expression is None:
        return json.dumps({
            "success": False,
            "error": "date must name exactly one bounded America/Phoenix calendar day",
        })
    token = _read_token()
    if token is None:
        return json.dumps({"success": False, "error": "Owner Calendar service is unavailable"})
    opener: Callable[..., Any] = kwargs.pop("opener", urlopen)
    payload = json.dumps({"date": expression}, separators=(",", ":")).encode()
    request = Request(
        _ENDPOINT,
        data=payload,
        headers={
            "Authorization": "Jarvis-Calendar " + token,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=20) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("oversized response")
        result = json.loads(raw)
        resolved_day = result.get("date") if isinstance(result, dict) else None
        if (not isinstance(result, dict) or result.get("timezone") != "America/Phoenix"
                or not isinstance(resolved_day, str)
                or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", resolved_day)):
            raise ValueError("invalid response")
        events = result.get("events")
        reply = result.get("reply")
        if not isinstance(events, list) or len(events) > 100 or not isinstance(reply, str):
            raise ValueError("invalid response")
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError):
        return json.dumps({"success": False, "error": "Owner Calendar service is unavailable"})
    return json.dumps({"success": True, **result})


def register(ctx: Any) -> None:
    ctx.register_tool(
        name="read_owner_calendar",
        toolset="owner_calendar",
        schema=_SCHEMA,
        handler=_handle_read,
        check_fn=_token_is_available,
        emoji="📅",
    )
