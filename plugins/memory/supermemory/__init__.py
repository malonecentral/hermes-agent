"""Supermemory memory plugin using the MemoryProvider interface.

Provides semantic long-term memory with profile recall, semantic search,
explicit memory tools, and cleaned completed-turn conversation capture.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


from .search_v4 import (
    canonical_chunk_preauthorized,
    canonical_scope_from_path,
    field as _v4_field,
    normalize_document_chunk,
    search_documents_v4,
)

from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret, is_multiplex_active
from tools.registry import tool_error

logger = logging.getLogger(__name__)


def _stage_receipt(stage: str, **values) -> None:
    allowed = {"received", "eligible", "rejected", "selected", "limit", "elapsed_ms", "chars", "bytes"}
    receipt = {"stage": stage}
    receipt.update({key: value for key, value in values.items()
                    if key in allowed and isinstance(value, int) and not isinstance(value, bool)})
    if values.get("outcome") in {"ok", "error", "deadline", "overflow"}:
        receipt["outcome"] = values["outcome"]
    logger.info("supermemory_stage %s", json.dumps(receipt, sort_keys=True))


class OwnerAppCaptureUnavailable(RuntimeError):
    """Owner app capture cannot run and should be retried by its caller."""

_DEFAULT_CONTAINER_TAG = "hermes"
_DEFAULT_MAX_RECALL_RESULTS = 5
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
_FAMILY_CANONICAL_CONTAINER = "family_shared"
_OWNER_CONVERSATION_CONTAINER = "owner_conversations"
_OWNER_RERANK_URL = "http://mcomen.malonecentral.com:8082/rerank"
_OWNER_RERANK_MODEL = "qwen3-reranker-0.6b-q8_0.gguf"
_OWNER_RERANK_TIMEOUT_SECONDS = 6.0
_OWNER_SOURCE_CANDIDATE_LIMIT = 20
# Two canonical containers plus the independently authorized conversation source.
_OWNER_QWEN_POOL_LIMIT = 3 * _OWNER_SOURCE_CANDIDATE_LIMIT
_FAMILY_RERANK_FAILURE_FALLBACK_LIMIT = 3
_DEFAULT_RERANKER_INPUT_TOKEN_BUDGET = 8192
# llama.cpp /rerank evaluates every (query, document) pair independently.  A
# valid UTF-8 byte is a conservative upper bound on tokenizer output, so these
# reserves protect each pair without imposing an incorrect aggregate cap on a
# request containing up to forty documents.
_OWNER_RERANKER_QUERY_BYTE_LIMIT = 1024
_OWNER_RERANKER_TEMPLATE_TOKEN_RESERVE = 768
_DEFAULT_CONTEXT_CHAR_BUDGET = 12000
_DEFAULT_CONTEXT_BYTE_BUDGET = 24000
_OWNER_EXACT_DOCUMENT_MAX_BYTES = 65536
_OWNER_EXACT_DATE_DOCUMENT_LIMIT = 4
_ELLIPTICAL_HISTORY_MESSAGES = 2
_ELLIPTICAL_HISTORY_CHAR_BUDGET = 320
_DIRECT_PERSONAL_RECALL_RE = re.compile(
    r"(?:\b(?:what|where|when|who|which|did|was|were)\b[?!.\s\w'-]{0,100}"
    r"\b(?:i|my|me)\b[?!.\s\w'-]{0,60}"
    r"\b(?:eat|ate|have|had|go|went|do|did|wear|wore|watch|watched|meet|met)\b|"
    r"\b(?:i|my|me)\b[?!.\s\w'-]{0,100}"
    r"\b(?:ate|had|went|did|wore|watched|met)\b)",
    re.IGNORECASE,
)


class _EvidenceProvenance(str, Enum):
    """Trusted local classification used by construction and reranking."""

    CANONICAL_DOCUMENT = "canonical_document"
    USER_CONVERSATION = "user_conversation"


_ROLE_BLOCK_RE = re.compile(
    r"\[role: (user|assistant)\]\n([\s\S]*?)\n\[\1:end\]",
)


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
_ELLIPTICAL_QUERY_RE = re.compile(
    r"^(?:and\s+|but\s+)?(?:did|do|does|was|were|is|are|can|could|would|should|"
    r"what|where|when|why|how)\b[^?!.]{0,120}\b(?:it|that|this|they|them|there)\b",
    re.IGNORECASE,
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


def _contextual_retrieval_query(query: str, history: Optional[List[Dict[str, Any]]]) -> str:
    """Ground an elliptical query in one bounded completed turn, without a model."""
    text = str(query or "").strip()
    if not _ELLIPTICAL_QUERY_RE.search(text) or not isinstance(history, list):
        return text
    selected: list[tuple[str, str]] = []
    remaining = _ELLIPTICAL_HISTORY_CHAR_BUDGET
    for message in reversed(history):
        if len(selected) >= _ELLIPTICAL_HISTORY_MESSAGES:
            break
        role = str(message.get("role") or "") if isinstance(message, dict) else ""
        content = message.get("content") if isinstance(message, dict) else None
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        clean = _CONTEXT_STRIP_RE.sub("", _CONTAINERS_STRIP_RE.sub("", content)).strip()
        if not clean:
            continue
        clean = clean[-remaining:]
        selected.append((role, clean))
        remaining -= len(clean)
        if remaining <= 0:
            break
    if not selected:
        return text
    prior = "\n".join(f"{role.title()}: {content}" for role, content in reversed(selected))
    return f"Previous conversation context:\n{prior}\nCurrent question: {text}"


def _empty_direct_recall_guidance(query: str, retrieval_context: Optional[dict]) -> str:
    """Return bounded fallback policy after an exhaustive direct-memory miss."""
    if not _DIRECT_PERSONAL_RECALL_RE.search(str(query or "")):
        return ""
    temporal = retrieval_context if isinstance(retrieval_context, dict) else {}
    if not temporal and not re.search(
        r"\b(?:today|yesterday|last\s+(?:night|week|month|year)|ago|on\s+\w+)\b",
        query,
        re.IGNORECASE,
    ):
        return ""
    return (
        "Direct personal-memory lookup completed successfully but found no matching evidence. "
        "Do not repeat the memory search, reload skills, or re-resolve the date. For this simple "
        "recall question, answer that no record was found. You may use at most one targeted "
        "fallback lookup only when a specific, directly relevant personal source is already known; "
        "do not browse broadly or chain exploratory tools. If that lookup has no evidence, stop."
    )


def _build_temporal_filters(retrieval_context: Optional[dict]) -> Optional[dict]:
    """Build bounded filters exclusively from trusted host context."""
    context = retrieval_context if isinstance(retrieval_context, dict) else {}
    if not _valid_trusted_temporal_scope(context):
        return None
    clauses: list[dict] = []
    for raw in context.get("event_date", ()):
        try:
            value = date.fromisoformat(raw).isoformat()
        except (TypeError, ValueError):
            continue
        clauses.append({"OR": [
            {"key": "event_date", "value": value},
            {"key": "eventDate", "value": value},
        ]})
    for bounds in context.get("event_date_ranges", ()):
        try:
            start, end = (date.fromisoformat(value) for value in bounds)
        except (TypeError, ValueError):
            continue
        if end < start or (end - start).days > 365:
            return None
        clauses.append({"AND": [
            {"filterType": "numeric", "key": "event_date_ordinal", "value": str(start.toordinal()), "numericOperator": ">="},
            {"filterType": "numeric", "key": "event_date_ordinal", "value": str(end.toordinal()), "numericOperator": "<="},
        ]})
    clauses = clauses[:100]  # at most 200 leaf conditions (provider limit)
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"OR": clauses}


def _conversation_has_explicit_user_content(text: str) -> bool:
    """Accept only a complete production role stream containing user text."""
    position = 0
    has_user_content = False
    for match in _ROLE_BLOCK_RE.finditer(text):
        if text[position:match.start()].strip():
            return False
        role, content = match.groups()
        if role == "user" and content.strip():
            has_user_content = True
        position = match.end()
    return has_user_content and not text[position:].strip()


def _role_delimited_evidence_text(text: str) -> str:
    """Separate user evidence from assistant context for downstream models."""
    users, assistants = [], []
    for role, content in _ROLE_BLOCK_RE.findall(text):
        cleaned = content.strip()
        if not cleaned:
            continue
        (users if role == "user" else assistants).append(cleaned)
    sections = ["[user-authored evidence]\n" + "\n\n".join(users)]
    if assistants:
        sections.append("[assistant context only; not evidence]\n" + "\n\n".join(assistants))
    return "\n\n".join(sections)


def _owner_capture_custom_id(item: dict) -> bool:
    """Recognize only IDs emitted by the two Owner capture entry points."""
    raw_metadata = item.get("metadata")
    metadata: dict = raw_metadata if isinstance(raw_metadata, dict) else {}
    # ``_source_custom_id`` is populated by the SDK-result normalizer from the
    # parent document returned by the exact search call.  Do not consult
    # arbitrary result metadata for this identity: extracted memories have
    # opaque IDs and metadata is user-controlled input at ingestion time.
    value = item.get("_source_custom_id")
    if not isinstance(value, str):
        return False
    hermes = re.fullmatch(r"hermes-owner-conversation:([^\s:]+)", value)
    if hermes:
        return metadata.get("session_id") == hermes.group(1)
    owner_app = re.fullmatch(r"jarvis-owner-app:([^\s:]+):([^\s:]+)", value)
    return bool(
        owner_app
        and metadata.get("session_id") == owner_app.group(1)
        and metadata.get("request_id") == owner_app.group(2)
    )


def _evidence_provenance(
    item: dict, *, schema_v4_ready: bool = False,
    trusted_owner_conversation_source: bool = False,
) -> Optional[_EvidenceProvenance]:
    """Classify evidence from provider metadata plus validated capture shape."""
    if _is_canonical_result(item, schema_v4_ready=schema_v4_ready):
        return _EvidenceProvenance.CANONICAL_DOCUMENT
    metadata = item.get("metadata") or {}
    text = str(item.get("memory") or "")
    if not trusted_owner_conversation_source or metadata.get("type") != "owner_conversation":
        return None
    # Raw captures retain cryptographically-unavailable but structurally exact
    # role parsing. Extracted memories have lost those delimiters, so require
    # every marker written by our Owner capture boundary plus its custom ID.
    if _conversation_has_explicit_user_content(text):
        return _EvidenceProvenance.USER_CONVERSATION
    if (
        text.strip()
        and metadata.get("authority") == "non-authoritative"
        and metadata.get("provenance") == "user-authored role-delimited statement"
        and _owner_capture_custom_id(item)
    ):
        return _EvidenceProvenance.USER_CONVERSATION
    return None


def _valid_trusted_temporal_scope(context: dict) -> bool:
    """Reject malformed or over-broad host-generated temporal scope."""
    if not isinstance(context, dict):
        return False
    for raw in context.get("event_date", ()):
        try:
            date.fromisoformat(raw)
        except (TypeError, ValueError):
            return False
    for bounds in context.get("event_date_ranges", ()):
        try:
            if len(bounds) != 2:
                return False
            start, end = (date.fromisoformat(value) for value in bounds)
        except (TypeError, ValueError):
            return False
        # Inclusive ranges are bounded to one calendar year (365/366 days).
        if end < start or (end - start).days > 365:
            return False
    return True


def _scope_owner_dated_event_results(
    query: str, results: list[dict], *, retrieval_context: Optional[dict] = None,
) -> list[dict]:
    """Prefer normalized event-date metadata, retaining legacy note support."""
    context = retrieval_context or {}
    requested: set[str] = set()
    for value in context.get("event_date", ()):
        try:
            requested.add(date.fromisoformat(value).isoformat())
        except (TypeError, ValueError):
            continue
    for bounds in context.get("event_date_ranges", ()):
        try:
            start, end = (date.fromisoformat(value) for value in bounds)
        except (TypeError, ValueError):
            continue
        if end < start or (end - start).days > 365:
            return []
        requested.update((start + timedelta(days=offset)).isoformat()
                         for offset in range((end - start).days + 1))
    if not requested and ("event_date" in context or "event_date_ranges" in context):
        return []
    if not requested:
        return results
    dated = [
        item for item in results
        if requested.intersection(_event_dates(item))
    ]
    if not dated:
        return []
    meal = re.search(r"\b(dinner|lunch|breakfast)\b", query, re.IGNORECASE)
    if meal:
        date_pattern = "|".join(map(re.escape, sorted(requested)))
        header = re.compile(
            rf"(?mi)^#+\s*(?:{date_pattern})[^\n]*\b{re.escape(meal.group(1))}\b"
        )
        meal_matches = [
            item for item in dated
            if header.search(str(item.get("memory") or ""))
            or ((item.get("metadata") or {}).get("source") == "conversation"
                and re.search(rf"\b{re.escape(meal.group(1))}\b", str(item.get("memory") or ""), re.IGNORECASE))
        ]
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


def _event_dates(item: dict) -> set[str]:
    metadata = item.get("metadata") or {}
    raw = metadata.get("event_date", metadata.get("eventDate"))
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        values = []
    normalized = set()
    for value in values:
        try:
            normalized.add(date.fromisoformat(str(value)).isoformat())
        except ValueError:
            continue
    if normalized:
        return normalized
    # Legacy notes predate indexed metadata. Bound the scan and accept only a
    # markdown heading or an explicit event-date/date field.
    memory = str(item.get("memory") or "")[:16384]
    found = re.findall(
        r"(?mi)^(?:#{1,6}\s*|[-*]\s*(?:event[ _-]?date|date)\s*:\s*)(\d{4}-\d{2}-\d{2})\b",
        memory,
    )
    valid = set()
    for value in found:
        try:
            valid.add(date.fromisoformat(value).isoformat())
        except ValueError:
            continue
    return valid


def _structured_fact_receipt(item: dict) -> Optional[tuple[str, str]]:
    """Read a conflict identity only from stable provider metadata."""
    metadata = item.get("metadata") or {}
    subject, key, value = (metadata.get(name) for name in ("fact_subject", "fact_key", "fact_value"))
    if not all(isinstance(part, str) and part.strip() for part in (subject, key, value)):
        return None
    return (
        f"{str(subject).strip().casefold()}|{str(key).strip().casefold()}",
        str(value).strip().casefold(),
    )


def _suppress_direct_conversation_conflicts(items: list[dict]) -> list[dict]:
    """Canonical wins only on a proven same-key/different-value conflict."""
    canonical_receipts = {
        receipt for item in items if _is_canonical_result(item)
        if (receipt := _structured_fact_receipt(item))
    }
    result = []
    for item in items:
        receipt = _structured_fact_receipt(item)
        if not _is_canonical_result(item) and receipt:
            if any(key == receipt[0] and value != receipt[1] for key, value in canonical_receipts):
                continue
        result.append(item)
    return result


def _utf8_prefix(text: str, byte_limit: int) -> str:
    """Return a valid-UTF-8 prefix no larger than byte_limit."""
    return text.encode("utf-8")[:max(0, byte_limit)].decode("utf-8", "ignore")


def _bounded_owner_reranker_payload(
    query: str, candidates: list[dict], *, token_budget: int = _DEFAULT_RERANKER_INPUT_TOKEN_BUDGET,
) -> tuple[dict, bytes]:
    """Build one fair, deterministic request within the measured model input.

    This is intentionally separate from the final injected-context budget.
    We use a conservative UTF-8 byte ceiling when a compatible local tokenizer
    is not already loaded, avoiding a new dependency and per-turn latency.
    """
    pair_budget = max(1, int(token_budget))
    bounded_query = _utf8_prefix(str(query), min(_OWNER_RERANKER_QUERY_BYTE_LIMIT, pair_budget))
    query_bytes = len(bounded_query.encode("utf-8"))
    documents = []
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        provenance = candidate.get("provenance")
        provenance_text = str(getattr(provenance, "value", provenance or "unknown"))
        envelope = f"[candidate-id: {candidate_id}]\n[provenance: {provenance_text}]\n[text]\n"
        envelope_bytes = len(envelope.encode("utf-8"))
        text_limit = pair_budget - query_bytes - _OWNER_RERANKER_TEMPLATE_TOKEN_RESERVE - envelope_bytes
        if text_limit < 0:
            raise ValueError("reranker query and metadata envelope exceed per-pair input budget")
        documents.append(envelope + _utf8_prefix(str(candidate.get("text") or ""), text_limit))
    payload = {"model": _OWNER_RERANK_MODEL, "query": bounded_query,
               "documents": documents, "top_n": len(candidates)}
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return payload, body


def _call_owner_reranker(
    query: str, candidates: list[dict], *, timeout: Optional[float] = None,
    input_token_budget: int = _DEFAULT_RERANKER_INPUT_TOKEN_BUDGET,
) -> dict:
    """Rank bounded evidence with the dedicated non-generative Qwen reranker.

    Candidate text is sent only as a document. Authority remains trusted local
    metadata and is applied after scoring, so document instructions cannot
    promote a conversation record into the canonical partition.
    """
    original_candidates = candidates
    payload, serialized_payload = _bounded_owner_reranker_payload(
        query, candidates, token_budget=input_token_budget,
    )
    request = urllib.request.Request(
        _OWNER_RERANK_URL,
        data=serialized_payload,
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
        if (index < 0 or index >= len(candidates) or not isinstance(score, (int, float))
                or isinstance(score, bool) or not math.isfinite(float(score))):
            return {}
        seen_indexes.add(index)
        candidate = candidates[index]
        candidate_id = str(candidate["id"])
        authority = "canonical" if candidate.get("authority") == "canonical" else "conversation"
        scored.append((authority, candidate_id, float(score), index))
    if len(seen_indexes) != len(candidates):
        return {}

    # Qwen's cross-encoder scores are ranking values, not calibrated
    # probabilities (valid best matches can be far below 0.5). Deterministic
    # authority/date/person/venue/authorship gates establish eligibility. Owner
    # root and Family Shared canonical evidence have equal authority after ACL.
    # Conversation evidence is globally score-ranked; canonical wins only for
    # a proven structured conflict and as the deterministic tie-breaker.
    eligible = sorted(
        (item for item in scored if item[0] == "canonical" or
         candidates[item[3]].get("provenance") is _EvidenceProvenance.USER_CONVERSATION),
        key=lambda item: (-item[2], 0 if item[0] == "canonical" else 1, item[3]),
    )
    selected_ids = [item[1] for item in eligible]
    selected_set = set(selected_ids)
    return {
        "selected_ids": selected_ids,
        "rejected_ids": [str(candidate["id"]) for candidate in original_candidates if str(candidate["id"]) not in selected_set],
        "sufficient": bool(selected_ids),
        "scores": [item[2] for item in eligible],
    }


def _is_capture_worthy_owner_statement(text: str) -> bool:
    """Conservatively retain declarative user evidence, not requests or probes."""
    normalized = " ".join((text or "").strip().split())
    if len(normalized) < _MIN_CAPTURE_LENGTH or "?" in normalized:
        return False
    lowered = normalized.lower()
    if re.search(
        r"\b(?:remember|memorize|forget)\b|\bignore\s+(?:all\s+)?(?:previous|prior|above)\b|"
        r"\b(?:system|developer)\s+(?:prompt|message|instruction)s?\b|"
        r"\b(?:prompt|operational)\s+instructions?\b|\bdo\s+not\s+(?:follow|obey|remember)\b",
        lowered,
    ):
        return False
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
        "canonical_document_search_mode": "documents",
        "entity_context": _DEFAULT_ENTITY_CONTEXT,
        "api_timeout": _DEFAULT_API_TIMEOUT,
        "prefetch_timeout": _DEFAULT_PREFETCH_TIMEOUT,
        "temporal_filters_schema_v4_ready": False,
        "context_char_budget": _DEFAULT_CONTEXT_CHAR_BUDGET,
        "context_byte_budget": _DEFAULT_CONTEXT_BYTE_BUDGET,
        "reranker_input_token_budget": _DEFAULT_RERANKER_INPUT_TOKEN_BUDGET,
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
    config["temporal_filters_schema_v4_ready"] = _as_bool(
        config.get("temporal_filters_schema_v4_ready"), False
    )
    for key, default in (("context_char_budget", _DEFAULT_CONTEXT_CHAR_BUDGET),
                         ("context_byte_budget", _DEFAULT_CONTEXT_BYTE_BUDGET),
                         ("reranker_input_token_budget", _DEFAULT_RERANKER_INPUT_TOKEN_BUDGET)):
        try:
            config[key] = max(1024, int(config.get(key, default)))
        except (TypeError, ValueError):
            config[key] = default
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
    # Canonical reads never inherit the conversation search mode.
    config["canonical_document_search_mode"] = "documents"
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


def _verified_v4_import_ready(hermes_home: str, *, storage_path: str | None = None,
                              vector_path: str | None = None, key_path: str | None = None,
                              source_root: str | None = None) -> bool:
    """Require current separate storage and vector receipts; never legacy aggregate state."""
    from .readiness_receipts import readiness_from_paths
    root = Path(hermes_home) / "readiness"
    storage = Path(storage_path) if storage_path else root / "storage-reconciliation.json"
    vector = Path(vector_path) if vector_path else root / "vector-readiness.json"
    key = Path(key_path) if key_path else root / "receipt-hmac.key"
    canonical_root = Path(source_root) if source_root else Path(
        "~/Documents/ObsidianVault/Personal/Hermes"
    ).expanduser()
    return readiness_from_paths(storage, vector, key_path=key, source_root=canonical_root)


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


def _valid_v4_canonical_path(metadata: dict) -> bool:
    """Validate the indexed path and its ACL classification as one unit."""
    scope = canonical_scope_from_path(metadata.get("relative_path"))
    return bool(scope and metadata.get("visibility") == (
        "family_shared" if scope == _FAMILY_CANONICAL_CONTAINER else "owner_private"
    ))


def _is_canonical_result(item: dict, *, schema_v4_ready: bool = False) -> bool:
    if not isinstance(item, dict):
        return False
    raw_metadata = item.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    source_container = item.get("_source_container")
    scope = canonical_scope_from_path(metadata.get("relative_path"))
    custom_id = item.get("_source_custom_id")
    exact = (
        scope in {_OWNER_CANONICAL_CONTAINER, _FAMILY_CANONICAL_CONTAINER}
        and item.get("_source_container") == scope
        and isinstance(custom_id, str) and bool(custom_id)
        and custom_id == "obsidian-" + hashlib.sha256(metadata["relative_path"].encode()).hexdigest()
        and metadata.get("index_schema_version") == 4
        and not isinstance(metadata.get("index_schema_version"), bool)
        and metadata.get("authority") == "canonical"
        and metadata.get("source") == "obsidian"
        and metadata.get("identity_scope") == "owner"
        and metadata.get("canonical_root") == "owner"
        and _valid_v4_canonical_path(metadata)
    )
    if not exact:
        logger.warning("supermemory_acl outcome=rejected schema_required=4 action=discard")
    return exact


def _visible_canonical_results(
    items: list, *, family: bool, authenticated_family: bool = False,
    schema_v4_ready: bool = True,
) -> list:
    """Apply canonical ACLs; this Owner provider has no family authentication."""
    if family and not authenticated_family:
        return []
    return [item for item in items or [] if _is_canonical_result(
        item, schema_v4_ready=schema_v4_ready,
    ) and (
        not family or (item.get("metadata") or {}).get("visibility") == "family_shared"
    )]


def _authoritative_search_results(search_results: list, *, schema_v4_ready: bool = False) -> list:
    """Fail closed: Owner facts must come from canonical Obsidian documents."""
    return [item for item in search_results or [] if _is_canonical_result(
        item, schema_v4_ready=schema_v4_ready,
    )]


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
        # Canonical chain names can likewise extend a possessive brand
        # ("Ike's Love & Sandwiches") that people naturally shorten to
        # "Ike's".  Derive aliases from the canonical name itself rather than
        # maintaining a venue-specific alias table. A leading written initial
        # may also be spoken as its English letter name ("J." -> "Jay").
        variants = [name]
        if " - " in name:
            variants.append(name.split(" - ", 1)[0])
        possessive_brand = re.match(r"^(.+?['’]s)(?:\s|$)", name, re.IGNORECASE)
        if possessive_brand:
            variants.append(possessive_brand.group(1))
        spoken_letters = {
            "a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee",
            "f": "ef", "g": "gee", "h": "aitch", "i": "eye", "j": "jay",
            "k": "kay", "l": "el", "m": "em", "n": "en", "o": "oh",
            "p": "pee", "q": "cue", "r": "ar", "s": "ess", "t": "tee",
            "u": "you", "v": "vee", "w": "double you", "x": "ex",
            "y": "why", "z": "zee",
        }
        for value in list(variants):
            initial = re.match(r"^([A-Za-z])\.\s*(.+)$", value)
            if initial:
                variants.append(f"{spoken_letters[initial.group(1).casefold()]} {initial.group(2)}")
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


def _exact_restaurant_parent(items: list[dict]) -> Optional[dict]:
    """Return one unambiguous schema-v4 restaurant search parent."""
    parents: dict[tuple[str, str], dict] = {}
    for item in items or []:
        metadata = item.get("metadata") or {}
        path = str(metadata.get("relative_path") or "")
        if not re.search(r"/Food/Restaurants/[^/]+\.md$", path, re.IGNORECASE):
            continue
        document_id = str(item.get("_parent_document_id") or "")
        if not document_id or not _is_canonical_result(item, schema_v4_ready=True):
            continue
        parents[(document_id, path)] = item
    return next(iter(parents.values())) if len(parents) == 1 else None


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
    family_context: bool = False,
    char_budget: int = _DEFAULT_CONTEXT_CHAR_BUDGET,
    byte_budget: int = _DEFAULT_CONTEXT_BYTE_BUDGET,
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
            if owner_context or family_context:
                memory = re.sub(
                    r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]",
                    lambda match: (match.group(2) or match.group(1).rsplit("/", 1)[-1]).strip(),
                    str(memory),
                )
                authority_label = (
                    "canonical" if _is_canonical_result(item)
                    else "user-authored conversation"
                )
                prefix_bits = [f"[authority: {authority_label}]"]
            else:
                prefix_bits = []
            similarity = item.get("similarity")
            updated = item.get("updated_at") or item.get("updatedAt") or ""
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
    elif family_context:
        intro += (
            "This is read-only canonical Family Shared evidence for an authenticated Family requester. "
            "It contains no Owner-private or conversational memory. Use only explicit facts below, preserve "
            "person attribution and relationship direction, and never infer that a described person is the requester. "
            "Canonical Family Shared evidence is authoritative. Prefer a terse direct answer, state when evidence is "
            "missing or ambiguous, and never emit Obsidian wikilinks. "
        )
    intro += "Do not force memories into the conversation."
    body = "\n\n".join(sections)
    opening = f"<supermemory-context>\n{intro}\n\n"
    closing = "\n</supermemory-context>"
    body = body[:max(0, char_budget - len(opening) - len(closing))]
    available_bytes = max(0, byte_budget - len(opening.encode()) - len(closing.encode()))
    if len(body.encode()) > available_bytes:
        body = body.encode()[:available_bytes].decode("utf-8", "ignore")
    context = opening + body.rstrip() + closing
    _stage_receipt("injection", selected=len(search), limit=max_results,
                   chars=len(context), bytes=len(context.encode()))
    return context


def _clean_text_for_capture(text: str) -> str:
    text = _CONTEXT_STRIP_RE.sub("", text or "")
    text = _CONTAINERS_STRIP_RE.sub("", text)
    return text.strip()


def _is_trivial_message(text: str) -> bool:
    return bool(_TRIVIAL_RE.match((text or "").strip()))


class _SupermemoryClient:
    def __init__(self, api_key: str, timeout: float, container_tag: str,
                 search_mode: str = "hybrid", base_url: str = "",
                 canonical_document_search_mode: str = "documents"):
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
        self._canonical_document_search_mode = canonical_document_search_mode
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
                        filters: Optional[dict] = None,
                        timeout: Optional[float] = None) -> list[dict]:
        tag = container_tag or self._container_tag
        mode = search_mode or self._search_mode
        kwargs: dict[str, Any] = {"q": query, "container_tag": tag}
        if filters:
            kwargs["filters"] = filters
        kwargs["limit"] = limit
        if mode in _VALID_SEARCH_MODES:
            kwargs["search_mode"] = mode
        if timeout is not None:
            kwargs["timeout"] = max(0.001, timeout)
        response = self._client.search.memories(**kwargs)
        results = []
        for item in (getattr(response, "results", None) or []):
            # Extracted results use two live SDK shapes. Most expose ``memory``
            # directly; aggregated results expose the extracted text in
            # ``chunk`` and retain the capture custom ID only on their sole
            # parent document. Normalize both here, while the source container
            # is known from this call, rather than teaching admission about SDK
            # nesting or trusting a metadata claim about its source.
            documents = getattr(item, "documents", None) or []
            source_custom_id = ""
            parent = None
            if len(documents) == 1:
                parent = documents[0]
                source_custom_id = str(getattr(parent, "id", "") or "")
            result_metadata = getattr(item, "metadata", None)
            parent_metadata = getattr(parent, "metadata", None) if parent is not None else None
            if (isinstance(result_metadata, dict) and isinstance(parent_metadata, dict)
                    and result_metadata != parent_metadata):
                logger.warning("supermemory_search outcome=rejected reason=parent_metadata_disagreement")
                continue
            normalized_metadata = (
                result_metadata if isinstance(result_metadata, dict)
                else parent_metadata if isinstance(parent_metadata, dict) else None
            )
            result_updated_at = getattr(item, "updated_at", None) or getattr(item, "updatedAt", None)
            parent_updated_at = (
                getattr(parent, "updated_at", None) or getattr(parent, "updatedAt", None)
                if parent is not None else None
            )
            if result_updated_at and parent_updated_at and result_updated_at != parent_updated_at:
                logger.warning("supermemory_search outcome=rejected reason=parent_timestamp_disagreement")
                continue
            results.append({
                "id": getattr(item, "id", ""),
                "memory": (
                    getattr(item, "memory", "")
                    or getattr(item, "chunk", "")
                    or getattr(item, "content", "")
                    or ""
                ),
                "similarity": getattr(item, "similarity", None),
                "updated_at": result_updated_at or parent_updated_at,
                "metadata": normalized_metadata,
                "_source_container": tag,
                "_source_custom_id": source_custom_id,
            })
        return results

    def search_documents(self, query: str, *, limit: int = 5,
                         container_tag: Optional[str] = None,
                         filters: Optional[dict] = None,
                         timeout: Optional[float] = None) -> list[dict]:
        """Search canonical chunks through the generic v4 endpoint."""
        tag = container_tag or self._container_tag
        started = time.monotonic()
        deadline = started + timeout if timeout is not None else None
        response = search_documents_v4(
            self._client, query, container_tag=tag, limit=limit, filters=filters,
            timeout=timeout, search_mode=getattr(self, "_canonical_document_search_mode", "documents"),
        )
        raw = _v4_field(response, "results", []) or []
        results = []
        for item in raw[:limit]:
            if not canonical_chunk_preauthorized(item, tag):
                continue
            parents = _v4_field(item, "documents")
            item_metadata = _v4_field(item, "metadata")
            summary_custom_id = (
                _v4_field(parents[0], "custom_id", _v4_field(parents[0], "customId"))
                if isinstance(parents, list) and len(parents) == 1 else None
            )
            if not isinstance(item_metadata, dict):
                continue
            expected_custom_id = "obsidian-" + hashlib.sha256(
                item_metadata["relative_path"].encode()
            ).hexdigest()
            # Current v4 summaries omit customId.  When a provider does return
            # one, a mismatch is conclusive and must reject before lookup.
            if summary_custom_id is not None and summary_custom_id != expected_custom_id:
                continue
            parent_id = (
                _v4_field(parents[0], "id")
                if isinstance(parents, list) and len(parents) == 1 else ""
            )
            hydrated = None
            cache = getattr(self, "_canonical_parent_cache", None)
            if cache is None:
                cache = self._canonical_parent_cache = {}
            if isinstance(parent_id, str) and parent_id:
                if parent_id in cache:
                    hydrated = cache[parent_id]
                elif deadline is None or time.monotonic() < deadline:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    try:
                        hydrated = self.get_document(parent_id, timeout=remaining)
                    except Exception:
                        hydrated = None
                    cache[parent_id] = hydrated
            normalized = normalize_document_chunk(
                item, tag, hydrated, allow_summary_proof=hydrated is None,
            )
            if normalized is not None:
                results.append(normalized)
        _stage_receipt("normalize", received=len(raw), eligible=len(results),
                       rejected=min(len(raw), limit) - len(results), limit=limit,
                       elapsed_ms=round((time.monotonic() - started) * 1000))
        return results

    def begin_turn(self) -> None:
        """Drop parent proofs at turn boundaries; dedupe them within a turn."""
        self._canonical_parent_cache = {}

    def get_document(self, document_id: str, *, timeout: Optional[float] = None) -> dict:
        """Fetch one document by its provider-issued parent ID."""
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = max(0.001, timeout)
        document = self._client.documents.get(document_id, **kwargs)
        return {
            "id": str(getattr(document, "id", "") or ""),
            "custom_id": str(getattr(document, "custom_id", None) or getattr(document, "customId", None) or ""),
            "content": str(getattr(document, "content", "") or ""),
            "metadata": getattr(document, "metadata", None),
            "container_tags": list(getattr(document, "container_tags", None) or getattr(document, "containerTags", None) or []),
            "task_type": getattr(document, "task_type", None) or getattr(document, "taskType", None),
            "status": getattr(document, "status", None),
            "updated_at": getattr(document, "updated_at", None) or getattr(document, "updatedAt", None),
        }


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
        # Safe pre-initialize defaults are required because MemoryManager
        # inspects tool schemas when the provider is added, before initialize_all.
        self._audience = "owner"
        self._family_mobile_reader = False
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
        self._audience = os.environ.get("HERMES_MEMORY_AUDIENCE", "owner").strip().casefold()
        if self._audience not in {"owner", "family"}:
            self._audience = "owner"
        self._family_mobile_reader = (
            self._audience == "family" and kwargs.get("platform") == "api"
        )

        # Resolve container tag: env var > config > default.
        # Supports {identity} template for profile-scoped containers.
        env_tag = os.environ.get("SUPERMEMORY_CONTAINER_TAG", "").strip()
        raw_tag = env_tag or self._config["container_tag"]
        identity = kwargs.get("agent_identity", "default")
        self._container_tag = _sanitize_tag(raw_tag.replace("{identity}", identity))
        # Provider-level namespace separation is the primary Family boundary;
        # metadata/path admission below remains defense in depth.
        if self._family_mobile_reader and self._container_tag != _FAMILY_CANONICAL_CONTAINER:
            self._family_mobile_reader = False

        self._auto_recall = self._config["auto_recall"]
        self._auto_capture = self._config["auto_capture"] and self._audience != "family"
        self._max_recall_results = self._config["max_recall_results"]
        self._profile_frequency = self._config["profile_frequency"]
        self._capture_mode = self._config["capture_mode"]
        self._search_mode = self._config["search_mode"]
        self._entity_context = self._config["entity_context"]
        self._api_timeout = self._config["api_timeout"]
        self._prefetch_timeout = self._config["prefetch_timeout"]
        self._context_char_budget = self._config["context_char_budget"]
        self._context_byte_budget = self._config["context_byte_budget"]
        self._reranker_input_token_budget = self._config["reranker_input_token_budget"]
        measured_chars = kwargs.get("runtime_context_chars")
        if isinstance(measured_chars, int) and measured_chars > 0:
            self._context_char_budget = min(self._context_char_budget, max(1024, measured_chars // 8))
        self._temporal_filters_schema_v4_ready = bool(
            self._config["temporal_filters_schema_v4_ready"]
            and _verified_v4_import_ready(
                self._hermes_home,
                storage_path=self._config.get("storage_reconciliation_receipt_path"),
                vector_path=self._config.get("vector_readiness_receipt_path"),
                key_path=self._config.get("readiness_receipt_key_path"),
                source_root=self._config.get("canonical_source_root"),
            )
        )
        # Base URL: config > SUPERMEMORY_BASE_URL env var > api.supermemory.ai.
        # Supports self-hosted Supermemory servers.
        self._base_url = _resolve_base_url(self._config["base_url"])
        self._enable_custom_containers = self._config["enable_custom_container_tags"]
        self._custom_containers = self._config["custom_containers"]
        self._custom_container_instructions = self._config["custom_container_instructions"]
        self._allowed_containers = [self._container_tag] + list(self._custom_containers)

        self._session_turns = []

        agent_context = kwargs.get("agent_context", "")
        self._write_enabled = (
            self._audience != "family"
            and agent_context not in {"cron", "flush", "subagent"}
        )
        self._active = bool(self._api_key) and (
            self._audience != "family" or self._family_mobile_reader
        )
        self._client = None
        if self._active:
            try:
                self._client = _SupermemoryClient(
                    api_key=self._api_key,
                    timeout=self._api_timeout,
                    container_tag=self._container_tag,
                    search_mode=self._search_mode,
                    base_url=self._base_url,
                    canonical_document_search_mode=self._config["canonical_document_search_mode"],
                )
            except Exception:
                logger.warning("Supermemory initialization failed", exc_info=True)
                self._active = False
                self._client = None

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = max(turn_number, 0)
        if self._client is not None and hasattr(self._client, "begin_turn"):
            self._client.begin_turn()

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        if self._audience == "family":
            return "# Family Shared memory\nRead-only canonical Family Shared evidence is prefetched automatically. No memory tools or conversational capture are available."
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

    def allows_automatic_context_without_tools(self) -> bool:
        """Allow only authenticated Family mobile read-only evidence mode."""
        return bool(
            self._active
            and self._family_mobile_reader
            and self._audience == "family"
            and not self._auto_capture
            and not self._write_enabled
            and not self.get_tool_schemas()
        )

    def _rerank_owner_candidates(
        self, query: str, items: list[dict], *, deadline: Optional[float] = None,
        trusted_conversation_items: Optional[list[dict]] = None,
    ) -> list[dict]:
        candidates = []
        by_id = {}
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()
        trusted_conversation_objects = {id(item) for item in (trusted_conversation_items or [])}
        for index, item in enumerate(items):
            text = str(item.get("memory") or "").strip()
            metadata = item.get("metadata") or {}
            provenance = _evidence_provenance(
                item, schema_v4_ready=self._temporal_filters_schema_v4_ready,
                trusted_owner_conversation_source=id(item) in trusted_conversation_objects,
            )
            if not text or provenance is None:
                continue
            candidate_id = str(item.get("id") or f"candidate-{index}")
            candidate_text = (
                _role_delimited_evidence_text(text)
                if (metadata.get("type") == "owner_conversation"
                    and _conversation_has_explicit_user_content(text)) else text
            )
            text_identity = " ".join(candidate_text.split()).casefold()
            if candidate_id in seen_ids or text_identity in seen_texts:
                continue
            candidate = {
                "id": candidate_id,
                "authority": "canonical" if provenance is _EvidenceProvenance.CANONICAL_DOCUMENT else "non-authoritative",
                "provenance": provenance,
                "timestamp": item.get("updated_at") or item.get("updatedAt") or "",
                "text": candidate_text,
                "metadata": metadata,
            }
            candidates.append(candidate)
            by_id[candidate_id] = (
                {**item, "memory": candidate_text}
                if candidate_text != text else item
            )
            seen_ids.add(candidate_id)
            seen_texts.add(text_identity)
        # Deterministic authority/entity/venue gates have already run. The
        # scorer may order eligible evidence, but its score cannot make an
        # otherwise ineligible record authoritative.
        # The independently bounded v3/v4 sources are deliberately combined
        # without quotas or pre-Qwen lexical admission. One request gives every
        # unique candidate identical pointwise scoring semantics.
        def family_failure_fallback(outcome: str, elapsed_ms: int) -> list[dict]:
            """Retain only revalidated Family Shared evidence when scoring fails."""
            if not self.allows_automatic_context_without_tools():
                selected_items: list[dict] = []
            else:
                approved = _visible_canonical_results(
                    items, family=True, authenticated_family=True,
                    schema_v4_ready=self._temporal_filters_schema_v4_ready,
                )
                approved_object_ids = {id(item) for item in approved}
                selected_items = [
                    by_id[candidate["id"]] for candidate in candidates
                    if id(by_id[candidate["id"]]) in approved_object_ids
                ][:_FAMILY_RERANK_FAILURE_FALLBACK_LIMIT]
            logger.warning(
                "supermemory_prefetch stage=reranker outcome=%s candidates=%d selected=%d "
                "score_min=na score_max=na elapsed_ms=%d fallback=family_acl",
                outcome, len(candidates), len(selected_items), elapsed_ms,
            )
            _stage_receipt("selection", outcome="error", eligible=len(candidates),
                           selected=len(selected_items), elapsed_ms=elapsed_ms)
            return selected_items

        _stage_receipt("qwen_pool", received=len(items), eligible=len(candidates),
                       rejected=len(items) - len(candidates), limit=_OWNER_QWEN_POOL_LIMIT)
        # Overflow means a source broke its bounded contract. Do not silently
        # drop eligible evidence before the common scorer.
        if len(candidates) > _OWNER_QWEN_POOL_LIMIT:
            _stage_receipt("qwen_pool", outcome="overflow", eligible=len(candidates))
            return []
        if len(candidates) <= 1:
            outcome = "selected" if candidates else "insufficient"
            selected_items = [by_id[candidates[0]["id"]]] if candidates else []
            logger.warning(
                "supermemory_prefetch stage=reranker outcome=%s candidates=%d selected=%d "
                "score_min=na score_max=na elapsed_ms=0 bypass=single",
                outcome, len(candidates), len(selected_items),
            )
            _stage_receipt("selection", eligible=len(candidates), selected=len(selected_items),
                           limit=self._max_recall_results)
            return selected_items
        started = time.monotonic()
        logger.warning("owner reranker request model=%s candidates=%d", _OWNER_RERANK_MODEL, len(candidates))
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return family_failure_fallback("deadline", 0)
        try:
            result = _call_owner_reranker(
                query, candidates, timeout=remaining,
                input_token_budget=self._reranker_input_token_budget,
            )
        except Exception:
            return family_failure_fallback(
                "exception", round((time.monotonic() - started) * 1000),
            )
        allowed_keys = {"selected_ids", "rejected_ids", "sufficient", "scores"}
        if not isinstance(result, dict) or not {"selected_ids", "rejected_ids", "sufficient"}.issubset(result) or not set(result).issubset(allowed_keys):
            return family_failure_fallback(
                "invalid", round((time.monotonic() - started) * 1000),
            )
        selected = result.get("selected_ids")
        rejected = result.get("rejected_ids")
        sufficient = result.get("sufficient")
        if not isinstance(selected, list) or not isinstance(rejected, list) or not isinstance(sufficient, bool):
            return family_failure_fallback(
                "invalid", round((time.monotonic() - started) * 1000),
            )
        combined = selected + rejected
        expected = set(by_id)
        if (not all(isinstance(value, str) for value in combined)
                or len(combined) != len(set(combined)) or set(combined) != expected):
            return family_failure_fallback(
                "invalid", round((time.monotonic() - started) * 1000),
            )
        if not sufficient or not selected:
            _stage_receipt("selection", eligible=len(candidates), selected=0, limit=self._max_recall_results)
            logger.warning("supermemory_prefetch stage=reranker outcome=insufficient candidates=%d selected=0 score_min=na score_max=na elapsed_ms=%d", len(candidates), round((time.monotonic() - started) * 1000))
            return []
        selected_items = _suppress_direct_conversation_conflicts(
            [by_id[candidate_id] for candidate_id in selected]
        )
        selected_items = selected_items[:self._max_recall_results]
        _stage_receipt("selection", eligible=len(candidates), selected=len(selected_items),
                       limit=self._max_recall_results)
        logger.warning(
            "supermemory_prefetch stage=reranker outcome=selected candidates=%d selected=%d score_min=%s score_max=%s elapsed_ms=%d",
            len(candidates), len(selected_items),
            min(result.get("scores") or [0]), max(result.get("scores") or [0]),
            round((time.monotonic() - started) * 1000),
        )
        return selected_items

    def _hydrate_exact_restaurant(
        self, items: list[dict], *, deadline: float,
    ) -> list[dict]:
        """Replace one exact venue chunk with its verified bounded parent."""
        candidate = _exact_restaurant_parent(items)
        if candidate is None or self._client is None:
            return items
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return items
        parent_id = str(candidate["_parent_document_id"])
        try:
            document = self._client.get_document(parent_id, timeout=remaining)
        except Exception:
            logger.warning("supermemory_prefetch stage=hydrate outcome=error")
            return items
        metadata = document.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        content = str(document.get("content") or "")
        prefix = re.match(r"\A\[canonical-identity\]\n[\s\S]*?\n\[/canonical-identity\]\n\n", content)
        source = content[prefix.end():] if prefix else ""
        source_bytes = source.encode("utf-8")
        expected_custom_id = "obsidian-" + hashlib.sha256(
            str(metadata.get("relative_path") or "").encode("utf-8")
        ).hexdigest()
        trusted = {
            **candidate, "metadata": metadata,
        }
        # The get endpoint normalizes away one final LF. Restore it only when
        # both the indexed byte count and SHA-256 prove that exact transform.
        expected_bytes = metadata.get("content_bytes")
        expected_hash = metadata.get("content_sha256")
        if (
            isinstance(expected_bytes, int) and expected_bytes == len(source_bytes) + 1
            and expected_hash == hashlib.sha256(source_bytes + b"\n").hexdigest()
        ):
            content += "\n"
            source_bytes += b"\n"
        valid = (
            parent_id in {document.get("id"), document.get("custom_id")}
            and document.get("custom_id") == expected_custom_id
            and (
                (_FAMILY_CANONICAL_CONTAINER if metadata.get("visibility") == "family_shared"
                 else _OWNER_CANONICAL_CONTAINER) in document.get("container_tags", [])
            )
            and document.get("task_type") == "superrag"
            and document.get("status") == "done"
            and metadata == (candidate.get("metadata") or {})
            and _is_canonical_result(trusted, schema_v4_ready=True)
            and isinstance(expected_bytes, int)
            and expected_bytes == len(source_bytes)
            and expected_hash == hashlib.sha256(source_bytes).hexdigest()
            and len(content.encode("utf-8")) <= _OWNER_EXACT_DOCUMENT_MAX_BYTES
            and time.monotonic() <= deadline
        )
        if not valid:
            logger.warning("supermemory_prefetch stage=hydrate outcome=rejected")
            return items
        hydrated = {
            **candidate, "id": parent_id, "memory": content,
            "updated_at": document.get("updated_at") or candidate.get("updated_at"),
        }
        return [hydrated if item is candidate else item for item in items]

    def _hydrate_exact_date_restaurants(
        self, retrieval_context: Optional[dict], *, deadline: float,
    ) -> list[dict]:
        """Read verified restaurant parents when the search index misses a date.

        This is deliberately not a temporal search-filter fallback.  It uses a
        host-resolved single date to locate bounded schema-v4 files under the
        importer's exact Family Shared restaurant root.
        """
        if (self._client is None or not isinstance(retrieval_context, dict)
                or not _valid_trusted_temporal_scope(retrieval_context)):
            return []
        dates = tuple(retrieval_context.get("event_date", ()))
        if len(dates) != 1 or (retrieval_context or {}).get("event_date_ranges"):
            return []
        try:
            receipt = json.loads(
                (Path(self._hermes_home) / "obsidian-supermemory-import.json").read_text()
            )
            root = Path(str(receipt["root"])).expanduser().resolve(strict=True)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return []
        restaurant_root = (root / "Jarvis/Family Shared/Food/Restaurants").resolve()
        if restaurant_root.parent.parent.parent.parent != root:
            return []
        needle = dates[0].encode("ascii")
        matches: list[tuple[Path, bytes]] = []
        try:
            for path in sorted(restaurant_root.glob("*.md")):
                raw = path.read_bytes()
                if needle in raw:
                    matches.append((path, raw))
                    if len(matches) > _OWNER_EXACT_DATE_DOCUMENT_LIMIT:
                        return []
        except OSError:
            return []

        hydrated: list[dict] = []
        for path, raw in matches:
            if time.monotonic() >= deadline or len(raw) > _OWNER_EXACT_DOCUMENT_MAX_BYTES:
                return []
            relative_path = path.relative_to(root).as_posix()
            custom_id = "obsidian-" + hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
            try:
                source = raw.decode("utf-8")
            except UnicodeDecodeError:
                return []
            if not re.match(r"\A---\s*\n[\s\S]{0,4096}?\nschema_version:\s*4\s*$", source,
                            re.MULTILINE):
                return []
            content = (
                "[canonical-identity]\n"
                f"canonical_path: {relative_path}\nentity_type: restaurant\n"
                f"entity_name: {path.stem}\n[/canonical-identity]\n\n{source}"
            )
            metadata = {
                "index_schema_version": 4, "source": "obsidian", "authority": "canonical",
                "identity_scope": "owner", "canonical_root": "owner",
                "visibility": "family_shared", "relative_path": relative_path,
                "content_bytes": len(raw), "content_sha256": hashlib.sha256(raw).hexdigest(),
            }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            try:
                document = self._client.get_document(custom_id, timeout=remaining)
            except Exception:
                logger.warning("supermemory_prefetch stage=exact_date_hydrate outcome=rejected")
                return []
            parent_metadata = document.get("metadata")
            parent_content = str(document.get("content") or "")
            parent_prefix = re.match(
                r"\A\[canonical-identity\]\n[\s\S]*?\n\[/canonical-identity\]\n\n",
                parent_content,
            )
            parent_source = parent_content[parent_prefix.end():] if parent_prefix else ""
            valid_parent = (
                canonical_scope_from_path(relative_path) == _FAMILY_CANONICAL_CONTAINER
                and canonical_chunk_preauthorized(
                    {"metadata": metadata}, _FAMILY_CANONICAL_CONTAINER,
                )
                and document.get("id") in {custom_id, document.get("custom_id")}
                and document.get("custom_id") == custom_id
                and document.get("container_tags") == [_FAMILY_CANONICAL_CONTAINER]
                and document.get("task_type") == "superrag"
                and document.get("status") == "done"
                and parent_metadata == metadata
                and parent_source.encode("utf-8") == raw
                and time.monotonic() <= deadline
            )
            if not valid_parent:
                logger.warning("supermemory_prefetch stage=exact_date_hydrate outcome=rejected")
                return []
            hydrated.append({
                "id": custom_id, "memory": content, "metadata": metadata,
                "updated_at": "",
                "_source_container": _FAMILY_CANONICAL_CONTAINER,
                "_source_custom_id": custom_id,
            })
        if hydrated:
            logger.info(
                "supermemory_prefetch stage=exact_date_hydrate outcome=selected candidates=%d",
                len(hydrated),
            )
        return hydrated

    def _parallel_owner_retrieval(
        self, query: str, retrieval_query: str, recall_query: str, deadline: float,
        retrieval_context: Optional[dict] = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Run independent read stages under one monotonic deadline.

        Every production HTTP call receives the remaining budget. Pending
        futures are cancelled and executor shutdown never waits past the outer
        deadline. Late results are ignored, so they cannot leak into a later turn.
        """
        client = self._client
        assert client is not None
        temporal_filters = (
            _build_temporal_filters(retrieval_context)
            if self._temporal_filters_schema_v4_ready else None
        )
        stages: dict[str, tuple[bool, Any]] = {
            "profile": (False, lambda timeout: client.get_profile(
                query=recall_query[:200], timeout=timeout, augment_search=False,
            )),
            "canonical_private": (True, lambda timeout: client.search_documents(
                recall_query[:511], limit=_OWNER_SOURCE_CANDIDATE_LIMIT, container_tag=_OWNER_CANONICAL_CONTAINER,
                **({"filters": temporal_filters} if temporal_filters else {}),
                timeout=timeout,
            )),
            "canonical_shared": (True, lambda timeout: client.search_documents(
                recall_query[:511], limit=_OWNER_SOURCE_CANDIDATE_LIMIT, container_tag=_FAMILY_CANONICAL_CONTAINER,
                **({"filters": temporal_filters} if temporal_filters else {}),
                timeout=timeout,
            )),
            "conversation": (False, lambda timeout: client.search_memories(
                retrieval_query, limit=_OWNER_SOURCE_CANDIDATE_LIMIT, container_tag=_OWNER_CONVERSATION_CONTAINER,
                **({"filters": temporal_filters} if temporal_filters else {}),
                search_mode=self._search_mode, timeout=timeout,
            )),
        }

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
                    _stage_receipt(name, outcome=outcome, elapsed_ms=elapsed_ms,
                                   received=len(value) if isinstance(value, list) else 0)
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
            _stage_receipt(name, outcome="deadline", elapsed_ms=elapsed_ms)
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
        results += self._client.search_documents(
            recall_query[:511], limit=20, container_tag=_FAMILY_CANONICAL_CONTAINER,
        )
        results = _authoritative_search_results(
            results, schema_v4_ready=self._temporal_filters_schema_v4_ready,
        )
        results = _scope_owner_named_person_results(owner_query, results)
        results, _named_restaurant = _scope_owner_restaurant_results(owner_query, results)
        results = _rank_owner_canonical_results(owner_query, results)
        conversations = self._client.search_memories(
            owner_query, limit=20, container_tag=_OWNER_CONVERSATION_CONTAINER,
            search_mode=self._search_mode,
        )
        results = self._rerank_owner_candidates(
            owner_query, results + conversations,
            trusted_conversation_items=conversations,
        )
        return _scope_owner_person_sections(owner_query, results)[:limit]

    def prefetch(
        self, query: str, *, session_id: str = "", deadline: Optional[float] = None,
        retrieval_context: Optional[dict] = None,
        retrieval_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not self._active or not self._auto_recall or not self._client or not query.strip():
            return ""
        temporal_scope = retrieval_context if self._temporal_filters_schema_v4_ready else None
        if retrieval_context and not self._temporal_filters_schema_v4_ready:
            logger.info("temporal_filter_not_ready schema_required=4 action=unfiltered")
        if temporal_scope and not _valid_trusted_temporal_scope(temporal_scope):
            logger.warning("supermemory_prefetch stage=temporal outcome=invalid action=discard")
            return ""
        try:
            family_reader = self._audience == "family"
            canonical_owner = self._container_tag == _OWNER_CANONICAL_CONTAINER and not family_reader
            retrieval_query = _contextual_retrieval_query(query, retrieval_history)
            recall_query = _owner_canonical_query(retrieval_query) if canonical_owner else retrieval_query
            values: dict[str, Any] = {}
            outcomes: dict[str, str] = {}
            include_profile = self._turn_count <= 1 or (self._turn_count % self._profile_frequency == 0)
            configured_deadline = time.monotonic() + self._prefetch_timeout
            if deadline is not None:
                configured_deadline = min(configured_deadline, deadline)
            deadline = configured_deadline - _PREFETCH_FORMAT_MARGIN
            if deadline <= time.monotonic():
                return ""
            if family_reader:
                raw = self._client.search_documents(
                    retrieval_query[:511], limit=_OWNER_SOURCE_CANDIDATE_LIMIT,
                    container_tag=_FAMILY_CANONICAL_CONTAINER,
                    timeout=max(0.001, deadline - time.monotonic()),
                )
                family_results = _visible_canonical_results(
                    list(raw or []), family=True, authenticated_family=True,
                    schema_v4_ready=self._temporal_filters_schema_v4_ready,
                )
                family_results = _scope_owner_named_person_results(retrieval_query, family_results)
                family_results = _scope_owner_dated_event_results(
                    retrieval_query, family_results, retrieval_context=temporal_scope,
                )
                restaurant_results, named_restaurant = _scope_owner_restaurant_results(
                    retrieval_query, family_results,
                )
                if named_restaurant:
                    hydrated_restaurant = self._hydrate_exact_restaurant(
                        restaurant_results, deadline=deadline,
                    )
                    scoped_ids = {id(item) for item in restaurant_results}
                    family_results = hydrated_restaurant + [
                        item for item in family_results if id(item) not in scoped_ids
                    ]
                search_results = self._rerank_owner_candidates(
                    retrieval_query, family_results, deadline=deadline,
                )
                return _format_prefetch_context(
                    static_facts=[], dynamic_facts=[], search_results=search_results,
                    max_results=self._max_recall_results, family_context=True,
                    char_budget=self._context_char_budget,
                    byte_budget=self._context_byte_budget,
                )
            if canonical_owner:
                values, outcomes = self._parallel_owner_retrieval(
                    query, retrieval_query, recall_query, deadline, temporal_scope,
                )
                if any(outcomes.get(name) != "ok" for name in ("canonical_private", "canonical_shared")):
                    logger.warning(
                        "supermemory_prefetch stage=canonical outcome=%s required=true action=discard",
                        "error",
                    )
                    return ""
                profile = values.get("profile") or {"static": [], "dynamic": [], "search_results": []}
                profile_results = list(profile.get("search_results") or [])
                profile_results += list(values.get("canonical_private") or [])
                profile_results += list(values.get("canonical_shared") or [])
                profile_results += list(values.get("fallback") or [])
            else:
                # Preserve the common provider's historical one-call fast path.
                profile = self._client.get_profile(
                    query=recall_query[:200], timeout=deadline - time.monotonic(),
                )
                profile_results = profile["search_results"]
            search_results = _authoritative_search_results(
                profile_results,
                schema_v4_ready=self._temporal_filters_schema_v4_ready,
            )
            named_restaurant = False
            if canonical_owner:
                search_results = _scope_owner_named_person_results(retrieval_query, search_results)
                search_results, _named_restaurant = _scope_owner_restaurant_results(retrieval_query, search_results)
                if _named_restaurant:
                    search_results = self._hydrate_exact_restaurant(search_results, deadline=deadline)
                search_results = _rank_owner_canonical_results(retrieval_query, search_results)
                conversation_results = list(values.get("conversation") or [])
                candidates = _scope_owner_dated_event_results(
                    retrieval_query, search_results + conversation_results,
                    retrieval_context=temporal_scope,
                )
                if (not candidates and retrieval_context and not temporal_scope
                        and re.search(r"\b(?:dinner|lunch|breakfast|brunch|ate|eat|meal)\b",
                                      retrieval_query, re.IGNORECASE)):
                    candidates = self._hydrate_exact_date_restaurants(
                        retrieval_context, deadline=deadline,
                    )
                search_results = self._rerank_owner_candidates(
                    retrieval_query, candidates, deadline=deadline,
                    trusted_conversation_items=conversation_results,
                )
                search_results = _scope_owner_person_sections(retrieval_query, search_results)
                if (
                    not search_results
                    and outcomes.get("canonical_private") == "ok"
                    and outcomes.get("canonical_shared") == "ok"
                    and outcomes.get("conversation") == "ok"
                ):
                    guidance = _empty_direct_recall_guidance(query, retrieval_context)
                    if guidance:
                        logger.info(
                            "supermemory_prefetch outcome=empty_direct_recall action=bounded_fallback"
                        )
                        return guidance
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
                max_results=self._max_recall_results,
                owner_context=canonical_owner,
                char_budget=self._context_char_budget,
                byte_budget=self._context_byte_budget,
            )
            return context
        except Exception:
            _stage_receipt("prefetch", outcome="error")
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
        if self._audience == "family":
            return []
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
        if getattr(self, "_audience", "owner") == "family":
            return tool_error("Memory tools are unavailable")
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
