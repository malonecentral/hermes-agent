"""Supermemory memory plugin using the MemoryProvider interface.

Provides semantic long-term memory with profile recall, semantic search,
explicit memory tools, and cleaned completed-turn conversation capture.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret, is_multiplex_active
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_CONTAINER_TAG = "hermes"
_DEFAULT_MAX_RECALL_RESULTS = 10
_DEFAULT_PROFILE_FREQUENCY = 50
_DEFAULT_CAPTURE_MODE = "all"
_DEFAULT_SEARCH_MODE = "hybrid"
_VALID_SEARCH_MODES = ("hybrid", "memories", "documents")
_DEFAULT_API_TIMEOUT = 5.0
_MIN_CAPTURE_LENGTH = 10
_MAX_ENTITY_CONTEXT_LENGTH = 1500
_DEFAULT_BASE_URL = "https://api.supermemory.ai"
_API_KEY_URL = "http://app.supermemory.ai/integrations?connect=hermes"
_OWNER_CANONICAL_CONTAINER = "owner_primary"
_OWNER_CONVERSATION_CONTAINER = "owner_conversations"
_OWNER_RERANK_URL = "http://mcomen.malonecentral.com:11434/api/chat"
_OWNER_RERANK_MODEL = "qwen3.5:4b"
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

_OWNER_RERANK_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["selected_ids", "rejected_ids", "sufficient"],
    "properties": {
        "selected_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "rejected_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "sufficient": {"type": "boolean"},
    },
}
_OWNER_RERANK_SYSTEM = """You are a strict evidence gate. Return only schema-valid JSON.
Every supplied ID must appear exactly once across selected_ids and rejected_ids.
Select the smallest set of supplied evidence that directly answers the exact question; otherwise select none and set sufficient false.
Match literal proper names, including middle names or initials. A similar spelling, semantic neighbour, or conflicting middle initial is not the requested person.
Preserve relationship direction exactly: evidence that A is B's parent does not support the reverse relationship.
Match the requested entity, relationship, venue, and domain. Never invent, rewrite, merge, extrapolate, or repair facts.
Reject evidence that merely mentions the same surname, venue category, or related person without supporting the requested claim.
Canonical evidence outranks conversation evidence for hard factual conflicts.
Explicit user statements may support personal preferences or recent decisions when canonical evidence is silent.
Assistant assertions and assistant-context text are not factual evidence and MUST be rejected.
A later explicit user correction supersedes an older conversational claim.
Set sufficient true only when the selected evidence, by itself, supports a concise answer to the exact question.
If any requested identity, relationship, date, occupation, location, or other material detail remains unsupported, set sufficient false.
"""


def _call_owner_reranker(query: str, candidates: list[dict]) -> dict:
    payload = {
        "model": _OWNER_RERANK_MODEL,
        "messages": [
            {"role": "system", "content": _OWNER_RERANK_SYSTEM},
            {"role": "user", "content": json.dumps({"question": query, "candidates": candidates}, ensure_ascii=False, separators=(",", ":"))},
        ],
        "stream": False, "think": False, "format": _OWNER_RERANK_SCHEMA, "keep_alive": "24h",
        "options": {"num_ctx": 16384, "temperature": 0, "top_p": 0.8, "top_k": 20, "min_p": 0, "presence_penalty": 0, "num_predict": 300},
    }
    request = urllib.request.Request(_OWNER_RERANK_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3.0) as response:
        body = json.load(response)
    return json.loads(body["message"]["content"])


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
    named = [(name, item) for name, item in restaurants if _restaurant_key(name) in query_key]
    if not named:
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


def _owner_canonical_query(query: str) -> str:
    """Resolve first-person parent terms against the authenticated Owner.

    Supermemory embeds a short question such as ``Who is my dad?`` too broadly:
    corpus policy text can outrank the actual person note.  Add the known
    speaker identity and relation vocabulary to the search query only; this
    changes neither the cached system prompt nor canonical authority.
    """
    text = (query or "").strip()
    lowered = text.lower()
    spouse_parent = re.search(
        r"\bmy\s+(wife|husband|spouse)['’]s\s+(dad|father|mom|mother)\b",
        lowered,
    )
    if spouse_parent:
        spouse_term, parent_term = spouse_parent.groups()
        parent_relation = "father" if parent_term in {"dad", "father"} else "mother"
        return (
            f"{text} Authenticated requester: Dennis Malone. "
            f"Resolve Dennis Malone's {spouse_term}, then that spouse's {parent_relation}. "
            "Require canonical evidence for both relationship edges."
        )
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
            "Prefer a terse list or direct sentence over narrative prose. If the requested identity differs from the selected record, "
            "state the mismatch rather than treating them as the same person. "
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
                        search_mode: Optional[str] = None) -> list[dict]:
        tag = container_tag or self._container_tag
        mode = search_mode or self._search_mode
        kwargs: dict[str, Any] = {"q": query, "container_tag": tag, "limit": limit}
        if mode in _VALID_SEARCH_MODES:
            kwargs["search_mode"] = mode
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

    def get_profile(self, query: Optional[str] = None, *,
                    container_tag: Optional[str] = None) -> dict:
        tag = container_tag or self._container_tag
        kwargs: dict[str, Any] = {"container_tag": tag}
        if query:
            kwargs["q"] = query
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
        if query and self._search_mode in {"hybrid", "documents"}:
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

    def _rerank_owner_candidates(self, query: str, items: list[dict]) -> list[dict]:
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
        # The fixed reranker contract was benchmarked with eight bounded
        # candidates. Deterministic authority/entity ranking has already run;
        # do not hand a local pre-call model an unbounded contaminated pile.
        candidates = candidates[:8]
        by_id = {candidate["id"]: by_id[candidate["id"]] for candidate in candidates}
        if len(candidates) <= 1:
            logger.info("owner reranker bypassed candidates=%d reason=%s", len(candidates), "single" if candidates else "empty")
            return [by_id[candidates[0]["id"]]] if candidates else []
        started = time.monotonic()
        logger.info("owner reranker request model=%s candidates=%d", _OWNER_RERANK_MODEL, len(candidates))
        result = _call_owner_reranker(query, candidates)
        logger.info(
            "owner reranker response candidates=%d selected=%d sufficient=%s elapsed_ms=%d",
            len(candidates),
            len(result.get("selected_ids") or []) if isinstance(result, dict) else 0,
            result.get("sufficient") if isinstance(result, dict) else None,
            round((time.monotonic() - started) * 1000),
        )
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
        return [by_id[candidate_id] for candidate_id in selected]

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active or not self._auto_recall or not self._client or not query.strip():
            return ""
        try:
            canonical_owner = self._container_tag == _OWNER_CANONICAL_CONTAINER
            recall_query = _owner_canonical_query(query) if canonical_owner else query
            profile = self._client.get_profile(query=recall_query[:200])
            include_profile = self._turn_count <= 1 or (self._turn_count % self._profile_frequency == 0)
            search_results = _authoritative_search_results(profile["search_results"])
            named_restaurant = False
            if canonical_owner:
                search_results = _scope_owner_named_person_results(query, search_results)
                search_results, named_restaurant = _scope_owner_restaurant_results(query, search_results)
                search_results = _rank_owner_canonical_results(query, search_results)
                conversation_results = self._client.search_memories(
                    query, limit=20, container_tag=_OWNER_CONVERSATION_CONTAINER, search_mode=self._search_mode
                )
                search_results = self._rerank_owner_candidates(query, search_results + conversation_results)
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
                    else 4 if canonical_owner and named_restaurant
                    else 1 if canonical_owner
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
        if not self._active or not self._auto_capture or not self._write_enabled or not self._client:
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
            clean_assistant = _clean_text_for_capture(assistant_content)
            if messages is not None:
                completed = []
                pending_user = ""
                for message in messages:
                    role = str(message.get("role") or "")
                    content = message.get("content")
                    if not isinstance(content, str):
                        continue
                    if role == "user":
                        pending_user = _clean_text_for_capture(content)
                    elif role == "assistant" and pending_user:
                        completed.append((pending_user, _clean_text_for_capture(content)))
                        pending_user = ""
                worthy = [(user, assistant) for user, assistant in completed if _is_capture_worthy_owner_statement(user)]
            else:
                worthy = [(clean_user, clean_assistant)] if _is_capture_worthy_owner_statement(clean_user) else []
            cleaned = []
            for user, assistant in worthy:
                cleaned.append({"role": "user", "content": user})
                if assistant:
                    cleaned.append({"role": "assistant-context", "content": assistant})
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
                "provenance": "role-delimited; assistant-context is non-evidence",
            })
        self._client.add_memory(
            self._format_conversation(cleaned),
            metadata=metadata,
            entity_context=self._entity_context,
            container_tag=_OWNER_CONVERSATION_CONTAINER if owner_capture else None,
            custom_id=(f"hermes-owner-conversation:{capture_session_id}" if owner_capture else f"hermes-session:{capture_session_id}"),
            task_type="memory",
        )

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
