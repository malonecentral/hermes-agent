"""Supermemory memory plugin using the MemoryProvider interface.

Provides semantic long-term memory with profile recall, semantic search,
explicit memory tools, and cleaned completed-turn conversation capture.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret, is_multiplex_active
from tools.registry import tool_error

logger = logging.getLogger(__name__)


class OwnerAppCaptureUnavailable(RuntimeError):
    """Owner app capture cannot run and should be retried by its caller."""

_DEFAULT_CONTAINER_TAG = "hermes"
_DEFAULT_MAX_RECALL_RESULTS = 10
_DEFAULT_PROFILE_FREQUENCY = 50
_DEFAULT_CAPTURE_MODE = "all"
_DEFAULT_SEARCH_MODE = "hybrid"
_VALID_SEARCH_MODES = ("hybrid", "memories", "documents")
_DEFAULT_API_TIMEOUT = 5.0
_DEFAULT_PREFETCH_TIMEOUT = 7.0
_PREFETCH_FORMAT_MARGIN = 0.01
_MIN_CAPTURE_LENGTH = 10
_MAX_ENTITY_CONTEXT_LENGTH = 1500
_DEFAULT_BASE_URL = "https://api.supermemory.ai"
_API_KEY_URL = "http://app.supermemory.ai/integrations?connect=hermes"
_OWNER_CANONICAL_CONTAINER = "owner_primary"
_OWNER_CONVERSATION_CONTAINER = "owner_conversations"
_OWNER_RERANK_URL = "http://mcomen.malonecentral.com:8082/rerank"
_OWNER_RERANK_MODEL = "qwen3-reranker-0.6b-q8_0.gguf"
_OWNER_RERANK_TIMEOUT_SECONDS = 6.0
_OWNER_RERANK_MIN_SCORE = 0.5
_OWNER_RERANK_CANDIDATE_LIMIT = 8
_OWNER_RERANK_EVIDENCE_LIMIT = 4
_OWNER_TIMEZONE = ZoneInfo("America/Phoenix")
_TRIVIAL_RE = re.compile(
    r"^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope|k|ty|thx|np)\.?$",
    re.IGNORECASE,
)
_CONTEXT_STRIP_RE = re.compile(
    r"<supermemory-context>[\s\S]*?</supermemory-context>\s*", re.DOTALL
)
_CONTAINERS_STRIP_RE = re.compile(
    r"<supermemory-containers>[\s\S]*?</supermemory-containers>\s*", re.DOTALL
)
_DEFAULT_ENTITY_CONTEXT = (
    "User-assistant conversation. Format: [role: user]...[user:end] and "
    "[role: assistant]...[assistant:end].\n\n"
    "Only extract things useful in future conversations. Most messages are not worth remembering.\n\n"
    "Remember lasting personal facts, preferences, routines, tools, ongoing projects, working context, "
    "and explicit requests to remember something.\n\n"
    "Do not remember temporary intents, one-time tasks, assistant actions, implementation details, or in-progress status.\n\n"
    "When in doubt, store less."
)

def _owner_now() -> datetime:
    """Return the Owner's wall-clock time for date resolution."""
    return datetime.now(_OWNER_TIMEZONE)


def _owner_expand_relative_dates(query: str, *, now: Optional[datetime] = None) -> str:
    """Annotate simple relative dates before semantic memory retrieval."""
    current = now or _owner_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=_OWNER_TIMEZONE)
    else:
        current = current.astimezone(_OWNER_TIMEZONE)
    offsets = {"yesterday": -1, "today": 0, "tomorrow": 1}
    pattern = re.compile(
        r"\b(yesterday|today|tomorrow)\b(?!\s*\(\d{4}-\d{2}-\d{2}\))",
        re.IGNORECASE,
    )

    def annotate(match: re.Match[str]) -> str:
        resolved = (current.date() + timedelta(days=offsets[match.group(1).lower()])).isoformat()
        return f"{match.group(1)} ({resolved})"

    return pattern.sub(annotate, query)


def _owner_prefetch_query(query: str) -> str:
    """Resolve relative dates and add retrieval vocabulary for dated events."""
    expanded = _owner_expand_relative_dates(query)
    if expanded != query and re.search(r"\b(?:dinner|lunch|breakfast|eat|ate|restaurant)\b", query, re.IGNORECASE):
        return f"{expanded} Dennis dining event restaurant location party ordered"
    return expanded


def _owner_prefetch_fallback_query(original_query: str) -> Optional[str]:
    """Return a broad semantic query when ISO dates damage embedding recall."""
    has_relative_date = re.search(r"\b(?:yesterday|today|tomorrow)\b", original_query, re.IGNORECASE)
    has_dining_cue = re.search(
        r"\b(?:dinner|lunch|breakfast|eat|ate|restaurant)\b", original_query, re.IGNORECASE
    )
    if has_relative_date and has_dining_cue:
        meal = re.search(r"\b(dinner|lunch|breakfast)\b", original_query, re.IGNORECASE)
        meal_term = meal.group(1).lower() if meal else "dining"
        return f"Dennis {meal_term} restaurant location party ordered"
    return None


def _scope_owner_dated_event_results(query: str, results: list[dict]) -> list[dict]:
    """Select records whose structured header matches the requested date and meal."""
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
    if not dates:
        return results
    dated = [item for item in results if any(date in str(item.get("memory") or "") for date in dates)]
    if not dated:
        return results
    meal = re.search(r"\b(dinner|lunch|breakfast)\b", query, re.IGNORECASE)
    if meal:
        date_pattern = "|".join(map(re.escape, dates))
        header = re.compile(
            rf"(?mi)^#+\s*(?:{date_pattern})[^\n]*\b{re.escape(meal.group(1))}\b"
        )
        meal_matches = [item for item in dated if header.search(str(item.get("memory") or ""))]
        if meal_matches:
            dated = meal_matches
    unique = []
    seen = set()
    for item in dated:
        identity = item.get("id") or str(item.get("memory") or "")
        if identity not in seen:
            seen.add(identity)
            unique.append(item)
    return unique


def _call_owner_reranker(
    query: str, candidates: list[dict], *, timeout: Optional[float] = None,
) -> dict:
    """Rank bounded evidence with the dedicated non-generative Qwen reranker.

    Candidate text is sent only as a document. Authority remains trusted local
    metadata and is applied after scoring, so document instructions cannot
    promote a conversation record into the canonical partition.
    """
    payload = {
        "model": _OWNER_RERANK_MODEL,
        "query": query,
        "documents": [str(candidate.get("text") or "") for candidate in candidates],
        "top_n": len(candidates),
    }
    request = urllib.request.Request(
        _OWNER_RERANK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    request_timeout = _OWNER_RERANK_TIMEOUT_SECONDS if timeout is None else min(
        _OWNER_RERANK_TIMEOUT_SECONDS, max(0.001, timeout)
    )
    with urllib.request.urlopen(request, timeout=request_timeout) as response:
        body = json.load(response)
    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list):
        return {}
    scored: list[tuple[str, str, float, int]] = []
    seen_indexes: set[int] = set()
    for item in results:
        if not isinstance(item, dict):
            return {}
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or isinstance(index, bool) or index in seen_indexes:
            return {}
        if index < 0 or index >= len(candidates) or not isinstance(score, (int, float)):
            return {}
        seen_indexes.add(index)
        candidate = candidates[index]
        authority = "canonical" if candidate.get("authority") == "canonical" else "conversation"
        scored.append((authority, str(candidate["id"]), float(score), index))
    if len(seen_indexes) != len(candidates):
        return {}

    relevant = [item for item in scored if item[2] >= _OWNER_RERANK_MIN_SCORE]
    canonical = [item for item in relevant if item[0] == "canonical"]
    # Once canonical evidence clears the semantic relevance threshold, do not
    # expose lower-authority text in the same answer context. This preserves
    # the established canonical-wins boundary without asking candidate text or
    # a generative model to classify its own authority.
    eligible = canonical or [item for item in relevant if item[0] != "canonical"]
    eligible.sort(key=lambda item: (-item[2], item[3]))
    selected_ids = [item[1] for item in eligible[:_OWNER_RERANK_EVIDENCE_LIMIT]]
    selected_set = set(selected_ids)
    return {
        "selected_ids": selected_ids,
        "rejected_ids": [str(candidate["id"]) for candidate in candidates if str(candidate["id"]) not in selected_set],
        "sufficient": bool(selected_ids),
    }


def _is_capture_worthy_owner_statement(text: str) -> bool:
    """Conservatively retain declarative user evidence, not requests or probes."""
    normalized = " ".join((text or "").strip().split())
    if len(normalized) < _MIN_CAPTURE_LENGTH or "?" in normalized:
        return False
    lowered = normalized.lower()
    if re.match(r"^(who|what|when|where|why|how|is|are|can|could|would|will|do|does|did)\b", lowered):
        return False
    if re.match(r"^(run|execute|check|find|search|show|tell|give|make|create|write|open|close|turn|set|send|answer|please)\b", lowered):
        return False
    if re.search(r"\b(test probe|testing|ignore (?:this|these)|answer [a-z0-9_-]+)\b", lowered):
        return False
    return True


def _default_config() -> dict:
    return {
        "container_tag": _DEFAULT_CONTAINER_TAG,
        "auto_recall": True,
        "auto_capture": True,
        "max_recall_results": _DEFAULT_MAX_RECALL_RESULTS,
        "profile_frequency": _DEFAULT_PROFILE_FREQUENCY,
        "capture_mode": _DEFAULT_CAPTURE_MODE,
        "search_mode": _DEFAULT_SEARCH_MODE,
        "entity_context": _DEFAULT_ENTITY_CONTEXT,
        "api_timeout": _DEFAULT_API_TIMEOUT,
        "prefetch_timeout": _DEFAULT_PREFETCH_TIMEOUT,
        "base_url": "",
        "enable_custom_container_tags": False,
        "custom_containers": [],
        "custom_container_instructions": "",
    }


def _sanitize_tag(raw: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9_]", "_", raw or "")
    tag = re.sub(r"_+", "_", tag)
    return tag.strip("_") or _DEFAULT_CONTAINER_TAG


def _resolve_base_url(config_value: Any = "") -> str:
    """Resolve the API base URL: config > SUPERMEMORY_BASE_URL env var > default.

    Supports self-hosted Supermemory servers (e.g. http://localhost:6767).
    """
    raw = (
        str(config_value or "").strip()
        or os.environ.get("SUPERMEMORY_BASE_URL", "").strip()
    )
    return (raw or _DEFAULT_BASE_URL).rstrip("/") or _DEFAULT_BASE_URL


def _clamp_entity_context(text: str) -> str:
    if not text:
        return _DEFAULT_ENTITY_CONTEXT
    text = text.strip()
    return text[:_MAX_ENTITY_CONTEXT_LENGTH]


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _load_supermemory_config(hermes_home: str) -> dict:
    config = _default_config()
    config_path = Path(hermes_home) / "supermemory.json"
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                config.update({k: v for k, v in raw.items() if v is not None})
        except Exception:
            logger.debug("Failed to parse %s", config_path, exc_info=True)

    # Keep raw container_tag — template variables like {identity} are resolved
    # in initialize(), and _sanitize_tag runs AFTER resolution.
    raw_tag = str(config.get("container_tag", _DEFAULT_CONTAINER_TAG)).strip()
    config["container_tag"] = raw_tag if raw_tag else _DEFAULT_CONTAINER_TAG
    config["auto_recall"] = _as_bool(config.get("auto_recall"), True)
    config["auto_capture"] = _as_bool(config.get("auto_capture"), True)
    try:
        config["max_recall_results"] = max(1, min(20, int(config.get("max_recall_results", _DEFAULT_MAX_RECALL_RESULTS))))
    except Exception:
        config["max_recall_results"] = _DEFAULT_MAX_RECALL_RESULTS
    try:
        config["profile_frequency"] = max(1, min(500, int(config.get("profile_frequency", _DEFAULT_PROFILE_FREQUENCY))))
    except Exception:
        config["profile_frequency"] = _DEFAULT_PROFILE_FREQUENCY
    config["capture_mode"] = "everything" if config.get("capture_mode") == "everything" else "all"
    raw_search_mode = str(config.get("search_mode", _DEFAULT_SEARCH_MODE)).strip().lower()
    config["search_mode"] = raw_search_mode if raw_search_mode in _VALID_SEARCH_MODES else _DEFAULT_SEARCH_MODE
    config["entity_context"] = _clamp_entity_context(str(config.get("entity_context", _DEFAULT_ENTITY_CONTEXT)))
    try:
        config["api_timeout"] = max(0.5, min(15.0, float(config.get("api_timeout", _DEFAULT_API_TIMEOUT))))
    except Exception:
        config["api_timeout"] = _DEFAULT_API_TIMEOUT
    try:
        configured_timeout = float(config.get("prefetch_timeout", _DEFAULT_PREFETCH_TIMEOUT))
        if not math.isfinite(configured_timeout) or configured_timeout <= 0:
            raise ValueError("prefetch_timeout must be finite and positive")
        config["prefetch_timeout"] = min(30.0, configured_timeout)
    except Exception:
        config["prefetch_timeout"] = _DEFAULT_PREFETCH_TIMEOUT
    config["base_url"] = str(config.get("base_url", "") or "").strip()

    # Multi-container support
    config["enable_custom_container_tags"] = _as_bool(config.get("enable_custom_container_tags"), False)
    raw_containers = config.get("custom_containers", [])
    if isinstance(raw_containers, list):
        config["custom_containers"] = [_sanitize_tag(str(t)) for t in raw_containers if t]
    else:
        config["custom_containers"] = []
    config["custom_container_instructions"] = str(config.get("custom_container_instructions", "")).strip()

    return config


def _save_supermemory_config(values: dict, hermes_home: str) -> None:
    config_path = Path(hermes_home) / "supermemory.json"
    existing = {}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
        except Exception:
            existing = {}
    existing.update(values)
    from utils import atomic_json_write
    atomic_json_write(config_path, existing, mode=0o600, sort_keys=True)


def _detect_category(text: str) -> str:
    lowered = text.lower()
    if re.search(r"prefer|like|love|hate|want", lowered):
        return "preference"
    if re.search(r"decided|will use|going with", lowered):
        return "decision"
    if re.search(r"\bis\b|\bare\b|\bhas\b|\bhave\b", lowered):
        return "fact"
    return "other"


def _format_relative_time(iso_timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        seconds = (now - dt).total_seconds()
        if seconds < 1800:
            return "just now"
        if seconds < 3600:
            return f"{int(seconds / 60)}m ago"
        if seconds < 86400:
            return f"{int(seconds / 3600)}h ago"
        if seconds < 604800:
            return f"{int(seconds / 86400)}d ago"
        if dt.year == now.year:
            return dt.strftime("%d %b")
        return dt.strftime("%d %b %Y")
    except Exception:
        return ""


def _deduplicate_recall(static_facts: list, dynamic_facts: list, search_results: list) -> tuple[list, list, list]:
    seen = set()
    out_static, out_dynamic, out_search = [], [], []
    for fact in static_facts or []:
        if fact and fact not in seen:
            seen.add(fact)
            out_static.append(fact)
    for fact in dynamic_facts or []:
        if fact and fact not in seen:
            seen.add(fact)
            out_dynamic.append(fact)
    for item in search_results or []:
        memory = item.get("memory", "")
        if memory and memory not in seen:
            seen.add(memory)
            out_search.append(item)
    return out_static, out_dynamic, out_search


def _is_canonical_result(item: dict) -> bool:
    metadata = item.get("metadata") or {}
    return metadata.get("authority") == "canonical" and metadata.get("source") == "obsidian"


def _authoritative_search_results(search_results: list) -> list:
    """Fail closed: Owner facts must come from canonical Obsidian documents."""
    return [item for item in search_results or [] if _is_canonical_result(item)]


def _restaurant_key(text: str) -> str:
    """Normalize spoken punctuation and harmless doubled consonants."""
    compact = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    return re.sub(r"([a-z])\1+", r"\1", compact)


def _scope_owner_named_person_results(query: str, results: list) -> list:
    """For an explicit biography query, retain only the exact person note."""
    match = re.search(
        r"\b(?:tell me (?:a little (?:bit )?)?(?:more )?about|who is)\s+"
        r"([A-Za-z][A-Za-z'’.-]*(?:\s+[A-Za-z][A-Za-z'’.-]*){1,3})\s*[?.!]*$",
        query or "", re.IGNORECASE,
    )
    if not match:
        return results
    requested = match.group(1).strip()
    if re.match(r"^(?:my|his|her|their|our|that)\b", requested, re.IGNORECASE):
        return results

    def tokens(value: str) -> list[str]:
        return re.findall(r"[a-z]+", value.casefold())

    wanted = tokens(requested)
    exact = []
    for item in results or []:
        relative_path = str((item.get("metadata") or {}).get("relative_path") or "")
        person = re.search(r"/People/([^/]+)\.md$", relative_path, re.IGNORECASE)
        if not person:
            continue
        candidate = tokens(person.group(1))
        if candidate == wanted or (len(wanted) == 2 and len(candidate) >= 2 and candidate[0] == wanted[0] and candidate[-1] == wanted[-1]):
            exact.append(item)
    return exact


def _scope_owner_restaurant_results(query: str, results: list) -> tuple[list, bool]:
    """Keep a named venue's canonical note and its reciprocal dish notes.

    Person names recur across restaurant notes, so semantic similarity alone
    can otherwise transfer an order from a different venue.
    """
    query_key = _restaurant_key(query)
    restaurants: list[tuple[str, dict]] = []
    for item in results or []:
        metadata = item.get("metadata") or {}
        relative_path = str(metadata.get("relative_path") or "")
        match = re.search(r"/Food/Restaurants/([^/]+)\.md$", relative_path, re.IGNORECASE)
        if match:
            restaurants.append((match.group(1), item))
    def venue_keys(name: str) -> list[str]:
        # Restaurant filenames may carry a branch suffix ("Venue - City") even
        # when ordinary questions name only the venue. Both are canonical IDs.
        variants = [name]
        if " - " in name:
            variants.append(name.split(" - ", 1)[0])
        return [key for key in (_restaurant_key(value) for value in variants) if key]

    named = [
        (name, item)
        for name, item in restaurants
        if any(key in query_key for key in venue_keys(name))
    ]
    if not named:
        # A query that syntactically names a venue must never fall through to
        # semantically similar records from another restaurant. Speech
        # normalization may repair known aliases upstream; unknown names fail
        # closed here rather than becoming someone else's order.
        if re.search(r"\b(?:at|from)\s+(?:the\s+)?[^?.!]+", query or "", re.IGNORECASE):
            return [], True
        return results, False

    # Prefer the longest match if one venue name contains another.
    venue, exact = max(named, key=lambda pair: len(_restaurant_key(pair[0])))
    venue_key = _restaurant_key(venue)
    reciprocal = []
    for item in results or []:
        metadata = item.get("metadata") or {}
        relative_path = str(metadata.get("relative_path") or "")
        if not re.search(r"/Food/Dishes/[^/]+\.md$", relative_path, re.IGNORECASE):
            continue
        memory_key = _restaurant_key(str(item.get("memory") or ""))
        if f"restaurants{venue_key}" in memory_key:
            reciprocal.append(item)
    query_words = {
        word for word in re.findall(r"\b[A-Z][a-z]+\b", query or "")
        if word not in {"What", "Where", "When", "Who", "Does", "Did", "Tell"}
    }
    reciprocal.sort(key=lambda item: not any(
        re.search(rf"\b{re.escape(word)}\b", str(item.get("memory") or ""))
        for word in query_words
    ))
    return [exact, *reciprocal], True


def _scope_usual_order_evidence(query: str, item: dict) -> dict:
    """For explicit usual-order questions, expose only the recorded answer."""
    if not re.search(r"\b(?:normally|usual(?:ly)?|typically)\b", query or "", re.IGNORECASE):
        return item
    memory = str(item.get("memory") or "")
    direct = re.search(r"(?mi)^-\s*Usual order:\s*.+$", memory)
    if direct:
        copy = dict(item)
        copy["memory"] = "### Dennis\n" + direct.group(0)
        return copy
    section = re.search(
        r"(?mis)^##\s+Usual orders by person\s*$.*?^-\s*Dennis:\s*.+$",
        memory,
    )
    if section:
        dennis = re.search(r"(?mi)^-\s*Dennis:\s*.+$", section.group(0))
        if dennis:
            copy = dict(item)
            copy["memory"] = "## Usual orders by person\n" + dennis.group(0)
            return copy
    return item


def _owner_canonical_query(query: str) -> str:
    """Resolve first-person parent terms against the authenticated Owner.

    Supermemory embeds a short question such as ``Who is my dad?`` too broadly:
    corpus policy text can outrank the actual person note.  Add the known
    speaker identity and relation vocabulary to the search query only; this
    changes neither the cached system prompt nor canonical authority.
    """
    text = (query or "").strip()
    lowered = text.lower()
    father = bool(re.search(r"\bmy\s+(?:dad|father)\b(?!-in-law)(?!['’]s(?!\s+name\b))", lowered))
    mother = bool(re.search(r"\bmy\s+(?:mom|mother)\b(?!-in-law)(?!['’]s(?!\s+name\b))", lowered))
    parents = bool(re.search(r"\bmy\s+parents?\b", lowered))
    if father:
        return (
            f"{text} Authenticated requester: Dennis Malone. "
            "Resolve Dennis Malone parent relationship: father or dad."
        )
    if mother:
        return (
            f"{text} Authenticated requester: Dennis Malone. "
            "Resolve Dennis Malone parent relationship: mother or mom."
        )
    if parents:
        return (
            f"{text} Authenticated requester: Dennis Malone. "
            "Resolve Dennis Malone parent relationship: father, dad, mother, or mom."
        )
    # A bare "about/from <name>" is commonly a person biography query. Only
    # add restaurant-specific expansion when the question itself contains a
    # food/order cue; exact restaurant notes can still scope ambiguous queries.
    if not re.search(
        r"\b(?:restaurant|menu|dish|food|order(?:ed|s)?|get|gets|got|eat|eats|ate|drink|drinks|favorite|usual|normally)\b",
        lowered,
    ):
        return text
    venue_match = re.search(
        r"\b(?:at|about|from)\s+([A-Za-z0-9][A-Za-z0-9 &'’.-]{0,80}?)(?:\s*[?.!]|$)",
        text,
        re.IGNORECASE,
    )
    if venue_match:
        venue = venue_match.group(1).strip()
        if re.match(r"^(?:my|his|her|their|our|the)\b", venue, re.IGNORECASE):
            return text
        return (
            f"{text} Canonical restaurant venue: {venue}. "
            f"Prefer the exact Food/Restaurants note and reciprocal Food/Dishes records for {venue}; exclude other venues."
        )
    return text


def _rank_owner_canonical_results(query: str, results: list) -> list:
    """Put direct Owner relationship evidence ahead of name collisions."""
    lowered = (query or "").lower()
    if re.search(r"\bmy\s+mother-in-law\b", lowered):
        def mother_in_law_rank(item: dict) -> int:
            text = str(item.get("memory") or "")
            metadata = item.get("metadata") or {}
            relative_path = str(metadata.get("relative_path") or "")
            if re.search(r"(?:^|\n)-\s*Son-in-law:\s*Dennis Malone(?:\s|$)", text, re.IGNORECASE):
                return 0
            if relative_path.endswith("/Courtnee Malone.md") and re.search(
                r"(?:^|\n)-\s*Mother:\s*(?:\[\[)?Mary Pat Thompson", text, re.IGNORECASE
            ):
                return 1
            return 2

        return sorted(results or [], key=mother_in_law_rank)
    relations = []
    if re.search(r"\bmy\s+(?:dad|father)\b(?!['’]s|-in-law)", lowered):
        relations.append("father")
    if re.search(r"\bmy\s+(?:mom|mother)\b(?!['’]s|-in-law)", lowered):
        relations.append("mother")
    if not relations and re.search(r"\bmy\s+parents?\b", lowered):
        relations = ["father", "mother"]
    if not relations:
        return results

    direct_patterns = [
        re.compile(
            rf"(?:^|\n)\s*[-*]\s+(?![^\n]*(?:\bnot\b|\bfalse\b|\buntrue\b|\bincorrect\b|\bunknown\b|\bwhether\b))"
            rf"[^\n]{{1,100}}?\bis\s+Dennis Malone['’]s\s+{relation}(?:[.,]|$)",
            re.IGNORECASE,
        )
        for relation in relations
    ]

    def evidence_rank(item: dict) -> int:
        text = str(item.get("memory") or "")
        metadata = item.get("metadata") or {}
        relative_path = str(metadata.get("relative_path") or "")
        if any(pattern.search(text) for pattern in direct_patterns):
            return 0
        if relative_path.endswith("/Dennis Malone.md") and any(
            re.search(
                rf"(?:^|\n)-\s*{relation.title()}:\s*(?![^\n]*(?:\bnot\b|\bfalse\b|\buntrue\b|\bincorrect\b|\bunknown\b|\bwhether\b))",
                text,
                re.IGNORECASE,
            )
            for relation in relations
        ):
            return 0
        return 1

    return sorted(results or [], key=evidence_rank)


def _owner_query_subject(query: str) -> str:
    """Resolve only the person explicitly requested by an Owner fact query."""
    text = (query or "").strip()
    if re.search(r"\b(?:my|I|me)\b", text, re.IGNORECASE):
        return "Dennis"
    match = re.search(
        r"\b(?:does|did|is|was|about)\s+([A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]+){0,2})(?:['’]s)?\b",
        text,
    )
    return match.group(1).strip() if match else ""


def _scope_owner_person_sections(query: str, items: list[dict]) -> list[dict]:
    """Remove sibling person sections from selected canonical evidence."""
    subject = _owner_query_subject(query)
    if not subject:
        return items
    names = {subject.casefold(), subject.split()[0].casefold()}
    scoped = []
    heading_re = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
    for item in items:
        memory = str(item.get("memory") or "")
        headings = list(heading_re.finditer(memory))
        selected = next((heading for heading in headings if heading.group(2).strip().casefold() in names), None)
        if selected is None:
            scoped.append(item)
            continue
        level = len(selected.group(1))
        end = len(memory)
        for heading in headings:
            if heading.start() > selected.start() and len(heading.group(1)) <= level:
                end = heading.start()
                break
        copy = dict(item)
        copy["memory"] = (memory[:headings[0].start()] + memory[selected.start():end]).strip()
        scoped.append(copy)
    return scoped


def _format_prefetch_context(
    static_facts: list,
    dynamic_facts: list,
    search_results: list,
    max_results: int,
    *,
    owner_context: bool = False,
) -> str:
    statics, dynamics, search = _deduplicate_recall(static_facts, dynamic_facts, search_results)
    statics = statics[:max_results]
    dynamics = dynamics[:max_results]
    search = search[:max_results]
    if not statics and not dynamics and not search:
        return ""

    sections = []
    if statics:
        sections.append("## User Profile (Persistent)\n" + "\n".join(f"- {item}" for item in statics))
    if dynamics:
        sections.append("## Recent Context\n" + "\n".join(f"- {item}" for item in dynamics))
    if search:
        lines = []
        for item in search:
            memory = item.get("memory", "")
            if not memory:
                continue
            if owner_context:
                memory = re.sub(
                    r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]",
                    lambda match: (match.group(2) or match.group(1).rsplit("/", 1)[-1]).strip(),
                    str(memory),
                )
            similarity = item.get("similarity")
            updated = item.get("updated_at") or item.get("updatedAt") or ""
            prefix_bits = []
            rel = _format_relative_time(updated)
            if rel:
                prefix_bits.append(f"[{rel}]")
            if similarity is not None:
                try:
                    prefix_bits.append(f"[{round(float(similarity) * 100)}%]")
                except Exception:
                    pass
            prefix = " ".join(prefix_bits)
            lines.append(f"- {prefix} {memory}".strip())
        if lines:
            sections.append("## Relevant Memories\n" + "\n".join(lines))
    if not sections:
        return ""

    intro = "The following is background context from long-term memory. Use it silently when relevant. "
    if owner_context:
        intro += (
            "The authenticated Owner/requester is Dennis. Never infer that Dennis is another person merely because "
            "a retrieved result describes that person; third-party records are context, not requester identity. "
            "For identity questions, answer Dennis only when directly supported by Owner context, otherwise state uncertainty. "
            "For factual answers, use only explicit facts below. Preserve proper names, dates, places, employers, and relationship "
            "direction exactly as written. Do not add connective biography, motives, inferred roles, relatives, or corrected spellings. "
            "If canonical Obsidian evidence directly conflicts with conversation evidence, canonical evidence wins; otherwise an explicit "
            "user-authored conversation fact may answer when canonical evidence is silent. "
            "Prefer a terse list or direct sentence over narrative prose. If a canonical venue record says a person's usual "
            "order is not stated, answer that no usual order is recorded; never promote liked foods, occasional choices, or "
            "drinks into a usual order. If the requested identity differs from the selected record, "
            "state the mismatch rather than treating them as the same person. Answer in plain text facts; never emit Obsidian wikilinks. "
        )
    intro += "Do not force memories into the conversation."
    body = "\n\n".join(sections)
    return f"<supermemory-context>\n{intro}\n\n{body}\n</supermemory-context>"


def _clean_text_for_capture(text: str) -> str:
    text = _CONTEXT_STRIP_RE.sub("", text or "")
    text = _CONTAINERS_STRIP_RE.sub("", text)
    return text.strip()


def _is_trivial_message(text: str) -> bool:
    return bool(_TRIVIAL_RE.match((text or "").strip()))


class _SupermemoryClient:
    def __init__(self, api_key: str, timeout: float, container_tag: str,
                 search_mode: str = "hybrid", base_url: str = ""):
        # Lazy-install the supermemory SDK on demand. ensure() honors
        # security.allow_lazy_installs (default true) and, on a sealed Docker
        # venv, redirects the install to the durable target. On failure we
        # fall through so the raw import below produces the canonical
        # ImportError message.
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("memory.supermemory", prompt=False)
        except ImportError:
            pass
        except Exception:
            pass
        from supermemory import Supermemory

        self._api_key = api_key
        self._container_tag = container_tag
        self._search_mode = search_mode if search_mode in _VALID_SEARCH_MODES else _DEFAULT_SEARCH_MODE
        self._timeout = timeout
        self._base_url = _resolve_base_url(base_url)
        self._client = Supermemory(
            api_key=api_key,
            base_url=self._base_url,
            timeout=timeout,
            max_retries=0,
            default_headers={"x-sm-source": "hermes"},
        )

    def _merge_metadata(self, metadata: Optional[dict]) -> dict:
        # sm_source routes Hermes writes into the "Hermes" Space in the Supermemory
        # app so the user can filter / bulk-manage them per source agent. This is a
        # functional routing key for the user, not vendor telemetry.
        merged = {"sm_source": "hermes", **(metadata or {})}
        legacy_source = merged.pop("source", None)
        if legacy_source and "type" not in merged:
            merged["type"] = str(legacy_source)
        return merged

    def add_memory(self, content: str, metadata: Optional[dict] = None, *,
                   entity_context: str = "", container_tag: Optional[str] = None,
                   custom_id: Optional[str] = None,
                   task_type: Optional[str] = None) -> dict:
        tag = container_tag or self._container_tag
        kwargs: dict[str, Any] = {
            "content": content.strip(),
            "container_tags": [tag],
        }
        if metadata:
            kwargs["metadata"] = self._merge_metadata(metadata)
        if entity_context:
            kwargs["entity_context"] = _clamp_entity_context(entity_context)
        if custom_id:
            kwargs["custom_id"] = custom_id
        if task_type:
            kwargs["task_type"] = task_type
        result = self._client.documents.add(**kwargs)
        return {"id": getattr(result, "id", "")}

    def search_memories(self, query: str, *, limit: int = 5,
                        container_tag: Optional[str] = None,
                        search_mode: Optional[str] = None,
                        timeout: Optional[float] = None) -> list[dict]:
        tag = container_tag or self._container_tag
        mode = search_mode or self._search_mode
        kwargs: dict[str, Any] = {"q": query, "container_tag": tag, "limit": limit}
        if mode in _VALID_SEARCH_MODES:
            kwargs["search_mode"] = mode
        if timeout is not None:
            kwargs["timeout"] = max(0.001, timeout)
        response = self._client.search.memories(**kwargs)
        results = []
        for item in (getattr(response, "results", None) or []):
            results.append({
                "id": getattr(item, "id", ""),
                "memory": (
                    getattr(item, "memory", "")
                    or getattr(item, "chunk", "")
                    or getattr(item, "content", "")
                    or ""
                ),
                "similarity": getattr(item, "similarity", None),
                "updated_at": getattr(item, "updated_at", None) or getattr(item, "updatedAt", None),
                "metadata": getattr(item, "metadata", None),
            })
        return results

    def search_documents(self, query: str, *, limit: int = 5,
                         container_tag: Optional[str] = None,
                         timeout: Optional[float] = None) -> list[dict]:
        """Search canonical superrag chunks rather than extracted memories."""
        tag = container_tag or self._container_tag
        kwargs: dict[str, Any] = dict(
            q=query,
            container_tags=[tag],
            limit=limit,
            rerank=False,
            rewrite_query=False,
            only_matching_chunks=True,
        )
        if timeout is not None:
            kwargs["timeout"] = max(0.001, timeout)
        response = self._client.search.documents(**kwargs)
        results = []
        for document in (getattr(response, "results", None) or []):
            document_id = (
                getattr(document, "document_id", None)
                or getattr(document, "documentId", None)
                or ""
            )
            metadata = getattr(document, "metadata", None)
            updated_at = (
                getattr(document, "updated_at", None)
                or getattr(document, "updatedAt", None)
            )
            chunks = getattr(document, "chunks", None) or []
            for index, chunk in enumerate(chunks):
                text = getattr(chunk, "content", "") or ""
                if not text:
                    continue
                identity = []
                if isinstance(metadata, dict):
                    for key in (
                        "canonical_path", "entity_type", "entity_name",
                        "venue_name", "branch", "schema_version",
                    ):
                        value = metadata.get(key)
                        if value not in (None, ""):
                            identity.append(f"{key}: {value}")
                if identity:
                    text = "[canonical-identity]\n" + "\n".join(identity) + "\n[/canonical-identity]\n\n" + text
                results.append({
                    "id": f"{document_id}:{index}" if document_id else "",
                    "memory": text,
                    "similarity": (
                        getattr(chunk, "score", None)
                        if getattr(chunk, "score", None) is not None
                        else getattr(document, "score", None)
                    ),
                    "updated_at": updated_at,
                    "metadata": metadata,
                })
        return results[:limit]

    def get_profile(self, query: Optional[str] = None, *,
                    container_tag: Optional[str] = None,
                    timeout: Optional[float] = None,
                    augment_search: bool = True) -> dict:
        tag = container_tag or self._container_tag
        kwargs: dict[str, Any] = {"container_tag": tag}
        if query:
            kwargs["q"] = query
        if timeout is not None:
            kwargs["timeout"] = max(0.001, timeout)
        response = self._client.profile(**kwargs)
        profile_data = getattr(response, "profile", None)
        search_data = getattr(response, "search_results", None) or getattr(response, "searchResults", None)
        static = getattr(profile_data, "static", []) or [] if profile_data else []
        dynamic = getattr(profile_data, "dynamic", []) or [] if profile_data else []
        raw_results = getattr(search_data, "results", None) or search_data or []
        search_results = []
        if isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, dict):
                    search_results.append(item)
                else:
                    search_results.append({
                        "memory": getattr(item, "memory", ""),
                        "updated_at": getattr(item, "updated_at", None) or getattr(item, "updatedAt", None),
                        "similarity": getattr(item, "similarity", None),
                    })
        if augment_search and query and self._search_mode in {"hybrid", "documents"}:
            if tag == _OWNER_CANONICAL_CONTAINER:
                hybrid_results = self.search_documents(
                    query,
                    limit=20,
                    container_tag=tag,
                )
            else:
                hybrid_results = self.search_memories(
                    query,
                    limit=20,
                    container_tag=tag,
                    search_mode=self._search_mode,
                )
            seen = {
                str(item.get("id") or "") + "\0" + str(item.get("memory") or "")
                for item in search_results
            }
            for item in hybrid_results:
                key = str(item.get("id") or "") + "\0" + str(item.get("memory") or "")
                if item.get("memory") and key not in seen:
                    search_results.append(item)
                    seen.add(key)
        return {"static": static, "dynamic": dynamic, "search_results": search_results}

    def forget_memory(self, memory_id: str, *, container_tag: Optional[str] = None) -> None:
        tag = container_tag or self._container_tag
        self._client.memories.forget(container_tag=tag, id=memory_id)

    def forget_by_query(self, query: str, *, container_tag: Optional[str] = None) -> dict:
        results = self.search_memories(query, limit=5, container_tag=container_tag)
        if not results:
            return {"success": False, "message": "No matching memory found to forget."}
        target = results[0]
        memory_id = target.get("id", "")
        if not memory_id:
            return {"success": False, "message": "Best matching memory has no id."}
        self.forget_memory(memory_id, container_tag=container_tag)
        preview = (target.get("memory") or "")[:100]
        return {"success": True, "message": f'Forgot: "{preview}"', "id": memory_id}


def _resolve_container_tag_for_setup(hermes_home: str, *, identity: str = "default") -> str:
    config = _load_supermemory_config(hermes_home)
    env_tag = os.environ.get("SUPERMEMORY_CONTAINER_TAG", "").strip()
    raw_tag = env_tag or config["container_tag"]
    return _sanitize_tag(raw_tag.replace("{identity}", identity))


def _probe_supermemory_connection(api_key: str, hermes_home: str, *, identity: str = "default") -> dict:
    config = _load_supermemory_config(hermes_home)
    base_url = _resolve_base_url(config["base_url"])
    status = {
        "ok": False,
        "error": "",
        "container_tag": _resolve_container_tag_for_setup(hermes_home, identity=identity),
        "profile_facts": 0,
        "auto_recall": bool(config["auto_recall"]),
        "auto_capture": bool(config["auto_capture"]),
    }
    if not (api_key or "").strip():
        status["error"] = "SUPERMEMORY_API_KEY not set"
        return status
    try:
        __import__("supermemory")
    except ImportError:
        status["error"] = "supermemory package not installed"
        return status
    try:
        client = _SupermemoryClient(
            api_key=api_key.strip(),
            timeout=config["api_timeout"],
            container_tag=status["container_tag"],
            search_mode=config["search_mode"],
            base_url=base_url,
        )
        profile = client.get_profile()
        facts = [
            fact for fact in (profile.get("static") or []) + (profile.get("dynamic") or [])
            if fact and str(fact).strip()
        ]
        status["profile_facts"] = len(facts)
        status["ok"] = True
    except Exception as exc:
        status["error"] = str(exc).strip()[:160] or "connection failed"
    return status


def _format_connection_summary(status: dict) -> str:
    recall = "on" if status.get("auto_recall") else "off"
    capture = "on" if status.get("auto_capture") else "off"
    container = status.get("container_tag") or _DEFAULT_CONTAINER_TAG
    if status.get("ok"):
        facts = int(status.get("profile_facts") or 0)
        fact_label = "fact" if facts == 1 else "facts"
        return (
            f"✓ Connected · container: {container} · {facts} profile {fact_label} · "
            f"auto_recall {recall} · auto_capture {capture}"
        )
    err = status.get("error") or "connection failed"
    return f"✗ {err} · container: {container} · auto_recall {recall} · auto_capture {capture}"


STORE_SCHEMA = {
    "name": "supermemory_store",
    "description": "Store an explicit memory for future recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to store."},
            "metadata": {"type": "object", "description": "Optional metadata attached to the memory."},
        },
        "required": ["content"],
    },
}

SEARCH_SCHEMA = {
    "name": "supermemory_search",
    "description": "Search long-term memory by semantic similarity.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Maximum results to return, 1 to 20."},
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "supermemory_forget",
    "description": "Forget a memory by exact id or by best-match query.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Exact memory id to delete."},
            "query": {"type": "string", "description": "Query used to find the memory to forget."},
        },
    },
}

PROFILE_SCHEMA = {
    "name": "supermemory_profile",
    "description": "Retrieve persistent profile facts and recent memory context.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Optional query to focus the profile response."},
        },
    },
}


class SupermemoryMemoryProvider(MemoryProvider):
    def __init__(self):
        self._config = _default_config()
        self._api_key = ""
        self._client: Optional[_SupermemoryClient] = None
        self._container_tag = _DEFAULT_CONTAINER_TAG
        self._session_id = ""
        self._turn_count = 0
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._write_thread: Optional[threading.Thread] = None
        self._auto_recall = True
        self._auto_capture = True
        self._max_recall_results = _DEFAULT_MAX_RECALL_RESULTS
        self._profile_frequency = _DEFAULT_PROFILE_FREQUENCY
        self._capture_mode = _DEFAULT_CAPTURE_MODE
        self._search_mode = _DEFAULT_SEARCH_MODE
        self._entity_context = _DEFAULT_ENTITY_CONTEXT
        self._api_timeout = _DEFAULT_API_TIMEOUT
        self._prefetch_timeout = _DEFAULT_PREFETCH_TIMEOUT
        self._base_url = _DEFAULT_BASE_URL
        self._hermes_home = ""
        self._write_enabled = True
        self._active = False
        # Multi-container support
        self._enable_custom_containers = False
        self._custom_containers: List[str] = []
        self._custom_container_instructions = ""
        self._allowed_containers: List[str] = []
        self._session_turns: List[Dict[str, str]] = []

    @property
    def name(self) -> str:
        return "supermemory"

    def is_available(self) -> bool:
        # Key presence only — no SDK import check. The supermemory SDK is
        # lazy-installed when the client is first constructed in initialize()
        # (see _SupermemoryClient.__init__). Gating availability on the SDK
        # being importable here would be a chicken-and-egg trap: on a sealed
        # Docker venv the package isn't present until ensure() runs, but
        # ensure() only runs once the provider is loaded — which this gates.
        # Mirrors honcho/mem0, which check config only. No network calls.
        return bool(get_secret("SUPERMEMORY_API_KEY", ""))

    def get_config_schema(self):
        # Only prompt for the API key during `hermes memory setup`.
        # All other options are documented for $HERMES_HOME/supermemory.json
        # or the SUPERMEMORY_CONTAINER_TAG env var.
        return [
            {"key": "api_key", "description": "Supermemory API key", "secret": True, "required": True, "env_var": "SUPERMEMORY_API_KEY", "url": _API_KEY_URL},
        ]

    def save_config(self, values, hermes_home):
        sanitized = dict(values or {})
        if "container_tag" in sanitized:
            sanitized["container_tag"] = _sanitize_tag(str(sanitized["container_tag"]))
        if "entity_context" in sanitized:
            sanitized["entity_context"] = _clamp_entity_context(str(sanitized["entity_context"]))
        _save_supermemory_config(sanitized, hermes_home)

    def get_status_config(self, provider_config: dict) -> dict:
        from hermes_constants import get_hermes_home

        del provider_config
        hermes_home = str(get_hermes_home())
        api_key = get_secret("SUPERMEMORY_API_KEY", "") or ""
        status = _probe_supermemory_connection(api_key, hermes_home)
        return {"summary": _format_connection_summary(status)}

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from hermes_cli.config import save_config
        from hermes_cli.memory_setup import _prompt, _write_env_vars

        print("\n  Configuring supermemory:\n")
        print(f"  Get your API key at {_API_KEY_URL}\n")

        env_writes: dict[str, str] = {}
        existing = os.environ.get("SUPERMEMORY_API_KEY", "")
        if existing:
            masked = f"...{existing[-4:]}" if len(existing) > 4 else "set"
            val = _prompt(f"Supermemory API key (current: {masked}, blank to keep)", secret=True)
        else:
            val = _prompt("Supermemory API key", secret=True)
        if val:
            env_writes["SUPERMEMORY_API_KEY"] = val

        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = self.name
        save_config(config)

        if env_writes:
            _write_env_vars(env_writes, hermes_home=hermes_home)

        api_key = env_writes.get("SUPERMEMORY_API_KEY") or existing
        # Make the freshly-entered key visible to the connection probe below.
        # (Checks the VALUE of SUPERMEMORY_API_KEY, not whether the key string
        # happens to name some unrelated env var.)
        # Single-profile convenience only: never write a profile's key into
        # the process-global environ under a multiplexed gateway — sibling
        # profiles' turns (and any subprocess spawned with env=os.environ)
        # would inherit it.
        if (
            api_key
            and not is_multiplex_active()
            and os.environ.get("SUPERMEMORY_API_KEY") != api_key
        ):
            os.environ["SUPERMEMORY_API_KEY"] = api_key

        status = _probe_supermemory_connection(api_key, hermes_home)
        print(f"\n  {_format_connection_summary(status)}")
        print("\n  Memory provider: supermemory")
        print("  Activation saved to config.yaml")
        if env_writes:
            print("  API keys saved to .env")
        print("\n  Start a new session to activate.\n")

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        self._hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        self._session_id = session_id
        self._turn_count = 0
        self._config = _load_supermemory_config(self._hermes_home)
        self._api_key = get_secret("SUPERMEMORY_API_KEY", "") or ""

        # Resolve container tag: env var > config > default.
        # Supports {identity} template for profile-scoped containers.
        env_tag = os.environ.get("SUPERMEMORY_CONTAINER_TAG", "").strip()
        raw_tag = env_tag or self._config["container_tag"]
        identity = kwargs.get("agent_identity", "default")
        self._container_tag = _sanitize_tag(raw_tag.replace("{identity}", identity))

        self._auto_recall = self._config["auto_recall"]
        self._auto_capture = self._config["auto_capture"]
        self._max_recall_results = self._config["max_recall_results"]
        self._profile_frequency = self._config["profile_frequency"]
        self._capture_mode = self._config["capture_mode"]
        self._search_mode = self._config["search_mode"]
        self._entity_context = self._config["entity_context"]
        self._api_timeout = self._config["api_timeout"]
        self._prefetch_timeout = self._config["prefetch_timeout"]
        # Base URL: config > SUPERMEMORY_BASE_URL env var > api.supermemory.ai.
        # Supports self-hosted Supermemory servers.
        self._base_url = _resolve_base_url(self._config["base_url"])
        self._enable_custom_containers = self._config["enable_custom_container_tags"]
        self._custom_containers = self._config["custom_containers"]
        self._custom_container_instructions = self._config["custom_container_instructions"]
        self._allowed_containers = [self._container_tag] + list(self._custom_containers)

        self._session_turns = []

        agent_context = kwargs.get("agent_context", "")
        self._write_enabled = agent_context not in {"cron", "flush", "subagent"}
        self._active = bool(self._api_key)
        self._client = None
        if self._active:
            try:
                self._client = _SupermemoryClient(
                    api_key=self._api_key,
                    timeout=self._api_timeout,
                    container_tag=self._container_tag,
                    search_mode=self._search_mode,
                    base_url=self._base_url,
                )
            except Exception:
                logger.warning("Supermemory initialization failed", exc_info=True)
                self._active = False
                self._client = None

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = max(turn_number, 0)

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        lines = [
            "# Supermemory",
            f"Active. Container: {self._container_tag}.",
            "Use supermemory-search, supermemory-save, supermemory-forget, and supermemory-profile (aliases: supermemory_search, supermemory_store, supermemory_forget, supermemory_profile).",
        ]
        if self._enable_custom_containers and self._custom_containers:
            tags_str = ", ".join(self._allowed_containers)
            lines.append(f"\nMulti-container mode enabled. Available containers: {tags_str}.")
            lines.append("Pass an optional container_tag to supermemory_search, supermemory_store, supermemory_forget, and supermemory_profile to target a specific container.")
            if self._custom_container_instructions:
                lines.append(f"\n{self._custom_container_instructions}")
        return "\n".join(lines)

    def _rerank_owner_candidates(
        self, query: str, items: list[dict], *, deadline: Optional[float] = None,
    ) -> list[dict]:
        candidates = []
        by_id = {}
        for index, item in enumerate(items):
            text = str(item.get("memory") or "").strip()
            metadata = item.get("metadata") or {}
            canonical = _is_canonical_result(item)
            conversation = metadata.get("source") == "conversation" or metadata.get("type") == "owner_conversation"
            if not text or not (canonical or conversation):
                continue
            if conversation and "[role: user]" not in text and metadata.get("speaker") != "user":
                continue
            candidate_id = str(item.get("id") or f"candidate-{index}")
            if candidate_id in by_id:
                candidate_id = f"{candidate_id}-{index}"
            candidate = {
                "id": candidate_id,
                "authority": "canonical" if canonical else "non-authoritative",
                "source": "obsidian" if canonical else "conversation",
                "speaker": metadata.get("speaker") or ("user-with-assistant-context" if conversation else "document"),
                "timestamp": item.get("updated_at") or item.get("updatedAt") or "",
                "text": text,
            }
            candidates.append(candidate)
            by_id[candidate_id] = item
        # Deterministic authority/entity/venue gates have already run. The
        # scorer may order eligible evidence, but its score cannot make an
        # otherwise ineligible record authoritative.
        # Give both eligible sources a path into the bounded reranker input.
        # The upstream lists are independently relevance-ordered, so alternate
        # them rather than allowing a long canonical list to starve explicit
        # user conversation facts before scoring begins.
        canonical_candidates = [candidate for candidate in candidates if candidate["authority"] == "canonical"]
        conversation_candidates = [candidate for candidate in candidates if candidate["authority"] != "canonical"]
        candidates = []
        for index in range(max(len(canonical_candidates), len(conversation_candidates))):
            if index < len(canonical_candidates):
                candidates.append(canonical_candidates[index])
            if index < len(conversation_candidates):
                candidates.append(conversation_candidates[index])
            if len(candidates) >= _OWNER_RERANK_CANDIDATE_LIMIT:
                break
        candidates = candidates[:_OWNER_RERANK_CANDIDATE_LIMIT]
        by_id = {candidate["id"]: by_id[candidate["id"]] for candidate in candidates}
        if len(candidates) <= 1:
            logger.warning("owner reranker bypassed candidates=%d reason=%s", len(candidates), "single" if candidates else "empty")
            return [by_id[candidates[0]["id"]]] if candidates else []
        started = time.monotonic()
        logger.warning("owner reranker request model=%s candidates=%d", _OWNER_RERANK_MODEL, len(candidates))
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            logger.warning("supermemory_prefetch stage=reranker outcome=deadline elapsed_ms=0 required=true")
            return []
        result = _call_owner_reranker(query, candidates, timeout=remaining)
        if not isinstance(result, dict) or set(result) != {"selected_ids", "rejected_ids", "sufficient"}:
            return []
        selected = result.get("selected_ids")
        rejected = result.get("rejected_ids")
        sufficient = result.get("sufficient")
        if not isinstance(selected, list) or not isinstance(rejected, list) or not isinstance(sufficient, bool):
            return []
        combined = selected + rejected
        expected = set(by_id)
        if (not all(isinstance(value, str) for value in combined)
                or len(combined) != len(set(combined)) or set(combined) != expected):
            return []
        if not sufficient or not selected:
            return []
        selected = selected[:_OWNER_RERANK_EVIDENCE_LIMIT]
        logger.warning(
            "owner reranker response candidates=%d selected=%d sufficient=%s elapsed_ms=%d",
            len(candidates), len(selected), sufficient,
            round((time.monotonic() - started) * 1000),
        )
        return [by_id[candidate_id] for candidate_id in selected]

    def _parallel_owner_retrieval(
        self, query: str, retrieval_query: str, recall_query: str, deadline: float,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Run independent read stages under one monotonic deadline.

        Every production HTTP call receives the remaining budget. Pending
        futures are cancelled and executor shutdown never waits past the outer
        deadline. Late results are ignored, so they cannot leak into a later turn.
        """
        client = self._client
        assert client is not None
        fallback_query = _owner_prefetch_fallback_query(query)
        stages: dict[str, tuple[bool, Any]] = {
            "profile": (False, lambda timeout: client.get_profile(
                query=recall_query[:200], timeout=timeout, augment_search=False,
            )),
            "canonical": (True, lambda timeout: client.search_documents(
                recall_query[:511], limit=20, container_tag=_OWNER_CANONICAL_CONTAINER,
                timeout=timeout,
            )),
            "conversation": (False, lambda timeout: client.search_memories(
                retrieval_query, limit=20, container_tag=_OWNER_CONVERSATION_CONTAINER,
                search_mode=self._search_mode, timeout=timeout,
            )),
        }
        if fallback_query:
            stages["fallback"] = (False, lambda timeout: client.search_documents(
                fallback_query, limit=20, container_tag=_OWNER_CANONICAL_CONTAINER,
                timeout=timeout,
            ))

        started_at = time.monotonic()

        def run_stage(name: str, call: Any) -> tuple[str, str, Any, int]:
            stage_started = time.monotonic()
            remaining = deadline - stage_started
            if remaining <= 0:
                return name, "deadline", None, 0
            try:
                value = call(remaining)
                outcome = "ok"
            except Exception:
                value = None
                outcome = "error"
            completed_at = time.monotonic()
            if completed_at > deadline:
                value = None
                outcome = "deadline"
            return name, outcome, value, round((completed_at - stage_started) * 1000)

        values: dict[str, Any] = {}
        outcomes: dict[str, str] = {}
        executor = ThreadPoolExecutor(
            max_workers=len(stages), thread_name_prefix="supermemory-prefetch"
        )
        futures = {
            executor.submit(run_stage, name, call): name
            for name, (_, call) in stages.items()
        }
        pending = set(futures)
        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                if not done:
                    break
                for future in done:
                    name, outcome, value, elapsed_ms = future.result()
                    outcomes[name] = outcome
                    if outcome == "ok":
                        values[name] = value
                    logger.info(
                        "supermemory_prefetch stage=%s outcome=%s elapsed_ms=%d required=%s",
                        name, outcome, elapsed_ms, str(stages[name][0]).lower(),
                    )
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        elapsed_ms = round((time.monotonic() - started_at) * 1000)
        for future in pending:
            name = futures[future]
            outcomes[name] = "deadline"
            logger.warning(
                "supermemory_prefetch stage=%s outcome=deadline elapsed_ms=%d required=%s",
                name, elapsed_ms, str(stages[name][0]).lower(),
            )
        return values, outcomes

    @staticmethod
    def _owner_tool_query(query: str) -> str:
        """Resolve first-person Owner tool queries before canonical scoping."""
        if re.search(r"\b(?:I|my|mine|myself)\b|(?<!tell )\bme\b", query, re.IGNORECASE) and not re.search(
            r"\bDennis(?:\s+Malone)?\b", query, re.IGNORECASE
        ):
            return f"Authenticated Owner is Dennis. Dennis asks about himself: {query}"
        return query

    def _search_owner_evidence(self, query: str, *, limit: int) -> list[dict]:
        """Apply prefetch's authority, venue, person, and rerank gates to tools."""
        owner_query = self._owner_tool_query(query)
        if self._client is None:
            return []
        recall_query = _owner_canonical_query(owner_query)
        results = self._client.search_documents(
            recall_query[:511], limit=20, container_tag=_OWNER_CANONICAL_CONTAINER,
        )
        results = _authoritative_search_results(results)
        results = _scope_owner_named_person_results(owner_query, results)
        results, named_restaurant = _scope_owner_restaurant_results(owner_query, results)
        venue_record = results[0] if named_restaurant and results else None
        results = _rank_owner_canonical_results(owner_query, results)
        conversations = [] if named_restaurant else self._client.search_memories(
            owner_query, limit=20, container_tag=_OWNER_CONVERSATION_CONTAINER,
            search_mode=self._search_mode,
        )
        results = self._rerank_owner_candidates(owner_query, results + conversations)
        if venue_record is not None:
            scoped_venue = _scope_usual_order_evidence(owner_query, venue_record)
            if scoped_venue is not venue_record:
                results = [scoped_venue]
            else:
                venue_id = venue_record.get("id")
                results = [
                    item for item in results
                    if item is not venue_record and (not venue_id or item.get("id") != venue_id)
                ]
                results = [venue_record] + results[:max(0, limit - 1)]
        return _scope_owner_person_sections(owner_query, results)[:limit]

    def prefetch(
        self, query: str, *, session_id: str = "", deadline: Optional[float] = None,
    ) -> str:
        if not self._active or not self._auto_recall or not self._client or not query.strip():
            return ""
        try:
            canonical_owner = self._container_tag == _OWNER_CANONICAL_CONTAINER
            retrieval_query = _owner_prefetch_query(query) if canonical_owner else query
            recall_query = _owner_canonical_query(retrieval_query) if canonical_owner else retrieval_query
            values: dict[str, Any] = {}
            include_profile = self._turn_count <= 1 or (self._turn_count % self._profile_frequency == 0)
            configured_deadline = time.monotonic() + self._prefetch_timeout
            if deadline is not None:
                configured_deadline = min(configured_deadline, deadline)
            deadline = configured_deadline - _PREFETCH_FORMAT_MARGIN
            if deadline <= time.monotonic():
                return ""
            if canonical_owner:
                values, outcomes = self._parallel_owner_retrieval(
                    query, retrieval_query, recall_query, deadline,
                )
                if outcomes.get("canonical") != "ok":
                    logger.warning(
                        "supermemory_prefetch stage=canonical outcome=%s required=true action=discard",
                        outcomes.get("canonical", "missing"),
                    )
                    return ""
                profile = values.get("profile") or {"static": [], "dynamic": [], "search_results": []}
                profile_results = list(profile.get("search_results") or [])
                profile_results += list(values.get("canonical") or [])
                profile_results += list(values.get("fallback") or [])
            else:
                # Preserve the common provider's historical one-call fast path.
                profile = self._client.get_profile(
                    query=recall_query[:200], timeout=deadline - time.monotonic(),
                )
                profile_results = profile["search_results"]
            search_results = _authoritative_search_results(profile_results)
            named_restaurant = False
            if canonical_owner:
                search_results = _scope_owner_named_person_results(retrieval_query, search_results)
                search_results, named_restaurant = _scope_owner_restaurant_results(retrieval_query, search_results)
                venue_record = search_results[0] if named_restaurant and search_results else None
                search_results = _rank_owner_canonical_results(retrieval_query, search_results)
                conversation_results = [] if named_restaurant else list(values.get("conversation") or [])
                candidates = _scope_owner_dated_event_results(
                    retrieval_query, search_results + conversation_results,
                )
                search_results = self._rerank_owner_candidates(
                    retrieval_query, candidates, deadline=deadline,
                )
                if venue_record is not None:
                    scoped_venue = _scope_usual_order_evidence(retrieval_query, venue_record)
                    if scoped_venue is not venue_record:
                        search_results = [scoped_venue]
                    else:
                        venue_id = venue_record.get("id")
                        search_results = [
                            item for item in search_results
                            if item is not venue_record and (not venue_id or item.get("id") != venue_id)
                        ]
                        search_results = [venue_record] + search_results[:3]
                search_results = _scope_owner_person_sections(retrieval_query, search_results)
            else:
                search_results, named_restaurant = _scope_owner_restaurant_results(query, search_results)
            context = _format_prefetch_context(
                static_facts=profile["static"] if include_profile and not canonical_owner else [],
                dynamic_facts=profile["dynamic"] if include_profile and not canonical_owner else [],
                search_results=(
                    search_results if canonical_owner else profile["search_results"]
                ),
                # Recalled blocks persist in prior user messages. Keep the 8K
                # retrieval runner below its context threshold across turns.
                max_results=(
                    2 if canonical_owner and re.search(r"\bmy\s+parents?\b", query, re.IGNORECASE)
                    else _OWNER_RERANK_EVIDENCE_LIMIT if canonical_owner
                    else self._max_recall_results
                ),
                owner_context=canonical_owner,
            )
            return context
        except Exception:
            logger.debug("Supermemory prefetch failed", exc_info=True)
            return ""

    @staticmethod
    def _clean_conversation(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        cleaned: List[Dict[str, str]] = []
        for message in messages or []:
            role = str(message.get("role") or "")
            # Assistant replies are generated claims, not personalization evidence.
            if role != "user":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            content = _clean_text_for_capture(content)
            if content:
                cleaned.append({"role": role, "content": content})
        return cleaned

    @staticmethod
    def _format_conversation(messages: List[Dict[str, str]]) -> str:
        return "\n\n".join(
            f"[role: {message['role']}]\n{message['content']}\n[{message['role']}:end]"
            for message in messages
        )

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages: Optional[List[Dict[str, Any]]] = None) -> None:
        # A provider can outlive a config change (CLI, desktop, and gateway
        # sessions are routinely long-running).  A capture disable must take
        # effect immediately instead of relying on those processes to restart.
        capture_enabled = bool(_load_supermemory_config(self._hermes_home)["auto_capture"])
        if (not self._active or not self._auto_capture or not capture_enabled
                or not self._write_enabled or not self._client):
            return
        capture_session_id = str(session_id or self._session_id).strip()
        if not capture_session_id:
            return
        if capture_session_id != self._session_id:
            self._session_id = capture_session_id
            self._session_turns = []

        owner_capture = self._container_tag == _OWNER_CANONICAL_CONTAINER
        if owner_capture:
            clean_user = _clean_text_for_capture(user_content)
            if messages is not None:
                completed = []
                for message in messages:
                    role = str(message.get("role") or "")
                    content = message.get("content")
                    if not isinstance(content, str):
                        continue
                    if role == "user":
                        completed.append(_clean_text_for_capture(content))
                worthy = [user for user in completed if _is_capture_worthy_owner_statement(user)]
            else:
                worthy = [clean_user] if _is_capture_worthy_owner_statement(clean_user) else []
            cleaned = [{"role": "user", "content": user} for user in worthy]
        elif messages is not None:
            cleaned = self._clean_conversation(messages)
            self._session_turns = [
                {"user": m["content"], "assistant": ""} if m["role"] == "user"
                else {"user": "", "assistant": m["content"]}
                for m in cleaned
            ]
        else:
            clean_user = _clean_text_for_capture(user_content)
            if clean_user:
                self._session_turns.append({"user": clean_user, "assistant": ""})
            cleaned = []
            for turn in self._session_turns:
                if turn.get("user"):
                    cleaned.append({"role": "user", "content": turn["user"]})
                if turn.get("assistant"):
                    cleaned.append({"role": "assistant", "content": turn["assistant"]})
        if not cleaned:
            return
        metadata = {
            "type": "owner_conversation" if owner_capture else "user_conversation",
            "session_id": capture_session_id,
            "message_count": len(cleaned),
        }
        if owner_capture:
            metadata.update({
                "authority": "non-authoritative",
                "provenance": "user-authored role-delimited statement",
            })
        self._client.add_memory(
            self._format_conversation(cleaned),
            metadata=metadata,
            entity_context=self._entity_context,
            container_tag=_OWNER_CONVERSATION_CONTAINER if owner_capture else None,
            custom_id=(f"hermes-owner-conversation:{capture_session_id}" if owner_capture else f"hermes-session:{capture_session_id}"),
            task_type="memory",
        )

    def capture_owner_app_turn(self, session_id: str, request_id: str,
                               user_content: str, assistant_content: str) -> bool:
        """Capture a turn, returning False only for terminal content filtering."""
        session_id = str(session_id or "").strip()
        request_id = str(request_id or "").strip()
        # Disabled capture is a terminal policy decision, not transient
        # provider unavailability.  Durable callers must acknowledge and stop
        # replaying these turns while the switch is off.
        if not self._auto_capture or not _load_supermemory_config(self._hermes_home)["auto_capture"]:
            return False
        unavailable = []
        if not self._active:
            unavailable.append("provider inactive")

        if not self._write_enabled:
            unavailable.append("writes disabled")
        if not self._client:
            unavailable.append("client unavailable")
        if self._container_tag != _OWNER_CANONICAL_CONTAINER:
            unavailable.append("non-canonical container")
        if not session_id or not request_id:
            unavailable.append("missing stable identifiers")
        if unavailable:
            raise OwnerAppCaptureUnavailable(", ".join(unavailable))
        user = _clean_text_for_capture(user_content)
        if not _is_capture_worthy_owner_statement(user):
            return False
        messages = [{"role": "user", "content": user}]
        self._client.add_memory(
            self._format_conversation(messages),
            metadata={"type": "owner_conversation", "capture_source": "jarvis_owner_app",
                      "session_id": session_id, "request_id": request_id,
                      "message_count": len(messages), "authority": "non-authoritative",
                      "provenance": "user-authored role-delimited statement"},
            entity_context=self._entity_context,
            container_tag=_OWNER_CONVERSATION_CONTAINER,
            custom_id=f"jarvis-owner-app:{session_id}:{request_id}",
            task_type="memory",
        )
        return True

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # Completed turns are already captured through MemoryManager's writer.
        self._session_turns = []

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """Rotate local capture state; completed turns need no boundary ingest."""
        self._session_id = str(new_session_id or "").strip() or self._session_id
        self._session_turns = []
        self._turn_count = 0

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._active or not self._write_enabled or not self._client:
            return
        if action != "add" or not (content or "").strip():
            return

        def _run():
            try:
                self._client.add_memory(
                    content.strip(),
                    metadata={"target": target, "type": "explicit_memory"},
                    entity_context=self._entity_context,
                )
            except Exception:
                logger.debug("Supermemory on_memory_write failed", exc_info=True)

        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=2.0)
        self._write_thread = None
        self._write_thread = threading.Thread(target=_run, daemon=False, name="supermemory-memory-write")
        self._write_thread.start()

    def shutdown(self) -> None:
        self._session_turns = []
        for attr_name in ("_prefetch_thread", "_sync_thread", "_write_thread"):
            thread = getattr(self, attr_name, None)
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
            setattr(self, attr_name, None)

    def _resolve_tool_container_tag(self, args: dict) -> Optional[str]:
        """Validate and resolve container_tag from tool call args.

        Returns None (use primary) if multi-container is disabled or no tag provided.
        Returns the validated tag if it's in the allowed list.
        Raises ValueError if the tag is not whitelisted.
        """
        if not self._enable_custom_containers:
            return None
        tag = str(args.get("container_tag") or "").strip()
        if not tag:
            return None
        sanitized = _sanitize_tag(tag)
        if sanitized not in self._allowed_containers:
            raise ValueError(
                f"Container tag '{sanitized}' is not allowed. "
                f"Allowed: {', '.join(self._allowed_containers)}"
            )
        return sanitized

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        def with_kebab_aliases(schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            aliases = {
                "supermemory_store": "supermemory-save",
                "supermemory_search": "supermemory-search",
                "supermemory_forget": "supermemory-forget",
                "supermemory_profile": "supermemory-profile",
            }
            expanded = list(schemas)
            for schema in schemas:
                kebab = aliases.get(schema.get("name", ""))
                if not kebab:
                    continue
                copy = json.loads(json.dumps(schema))
                copy["name"] = kebab
                expanded.append(copy)
            return expanded

        if not self._enable_custom_containers:
            return with_kebab_aliases([STORE_SCHEMA, SEARCH_SCHEMA, FORGET_SCHEMA, PROFILE_SCHEMA])

        # When multi-container is enabled, add optional container_tag to relevant tools
        container_param = {
            "type": "string",
            "description": f"Optional container tag. Allowed: {', '.join(self._allowed_containers)}. Defaults to primary ({self._container_tag}).",
        }
        schemas = []
        for base in [STORE_SCHEMA, SEARCH_SCHEMA, FORGET_SCHEMA, PROFILE_SCHEMA]:
            schema = json.loads(json.dumps(base))  # deep copy
            schema["parameters"]["properties"]["container_tag"] = container_param
            schemas.append(schema)
        return with_kebab_aliases(schemas)

    def _tool_store(self, args: dict) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        try:
            tag = self._resolve_tool_container_tag(args)
        except ValueError as exc:
            return tool_error(str(exc))
        metadata = args.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("type", _detect_category(content))
        metadata.pop("source", None)
        try:
            result = self._client.add_memory(content, metadata=metadata, entity_context=self._entity_context, container_tag=tag)
            preview = content[:80] + ("..." if len(content) > 80 else "")
            resp: dict[str, Any] = {"saved": True, "id": result.get("id", ""), "preview": preview}
            if tag:
                resp["container_tag"] = tag
            return json.dumps(resp)
        except Exception as exc:
            return tool_error(f"Failed to store memory: {exc}")

    def _tool_search(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        try:
            tag = self._resolve_tool_container_tag(args)
        except ValueError as exc:
            return tool_error(str(exc))
        try:
            limit = max(1, min(20, int(args.get("limit", 5) or 5)))
        except Exception:
            limit = 5
        try:
            if self._container_tag == _OWNER_CANONICAL_CONTAINER and tag in {None, _OWNER_CANONICAL_CONTAINER}:
                results = self._search_owner_evidence(query, limit=limit)
            else:
                results = self._client.search_memories(query, limit=limit, container_tag=tag)
            formatted = []
            for item in results:
                entry: dict[str, Any] = {"id": item.get("id", ""), "content": item.get("memory", "")}
                if item.get("similarity") is not None:
                    try:
                        entry["similarity"] = round(float(item["similarity"]) * 100)
                    except Exception:
                        pass
                formatted.append(entry)
            resp: dict[str, Any] = {"results": formatted, "count": len(formatted)}
            if tag:
                resp["container_tag"] = tag
            return json.dumps(resp)
        except Exception as exc:
            return tool_error(f"Search failed: {exc}")

    def _tool_forget(self, args: dict) -> str:
        memory_id = str(args.get("id") or "").strip()
        query = str(args.get("query") or "").strip()
        if not memory_id and not query:
            return tool_error("Provide either id or query")
        try:
            tag = self._resolve_tool_container_tag(args)
        except ValueError as exc:
            return tool_error(str(exc))
        try:
            if memory_id:
                self._client.forget_memory(memory_id, container_tag=tag)
                return json.dumps({"forgotten": True, "id": memory_id})
            return json.dumps(self._client.forget_by_query(query, container_tag=tag))
        except Exception as exc:
            return tool_error(f"Forget failed: {exc}")

    def _tool_profile(self, args: dict) -> str:
        query = str(args.get("query") or "").strip() or None
        try:
            tag = self._resolve_tool_container_tag(args)
        except ValueError as exc:
            return tool_error(str(exc))
        try:
            profile = self._client.get_profile(query=query, container_tag=tag)
            sections = []
            if profile["static"]:
                sections.append("## User Profile (Persistent)\n" + "\n".join(f"- {item}" for item in profile["static"]))
            if profile["dynamic"]:
                sections.append("## Recent Context\n" + "\n".join(f"- {item}" for item in profile["dynamic"]))
            resp: dict[str, Any] = {
                "profile": "\n\n".join(sections),
                "static_count": len(profile["static"]),
                "dynamic_count": len(profile["dynamic"]),
            }
            if tag:
                resp["container_tag"] = tag
            return json.dumps(resp)
        except Exception as exc:
            return tool_error(f"Profile failed: {exc}")

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._active or not self._client:
            return tool_error("Supermemory is not configured")
        aliases = {
            "supermemory-save": "supermemory_store",
            "supermemory-search": "supermemory_search",
            "supermemory-forget": "supermemory_forget",
            "supermemory-profile": "supermemory_profile",
        }
        tool_name = aliases.get(tool_name, tool_name)
        if tool_name == "supermemory_store":
            return self._tool_store(args)
        if tool_name == "supermemory_search":
            return self._tool_search(args)
        if tool_name == "supermemory_forget":
            return self._tool_forget(args)
        if tool_name == "supermemory_profile":
            return self._tool_profile(args)
        return tool_error(f"Unknown tool: {tool_name}")


def register(ctx):
    ctx.register_memory_provider(SupermemoryMemoryProvider())
