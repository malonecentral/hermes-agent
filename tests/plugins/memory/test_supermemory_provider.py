import json
import os
import stat
import threading

import pytest

from plugins.memory.supermemory import (
    SupermemoryMemoryProvider,
    _clean_text_for_capture,
    _format_connection_summary,
    _format_prefetch_context,
    _load_supermemory_config,
    _probe_supermemory_connection,
    _save_supermemory_config,
)


class FakeClient:
    def __init__(self, api_key: str, timeout: float, container_tag: str, search_mode: str = "hybrid",
                 base_url: str = ""):
        self.api_key = api_key
        self.profile_queries = []
        self.timeout = timeout
        self.container_tag = container_tag
        self.search_mode = search_mode
        self.base_url = base_url
        self.add_calls = []
        self.search_calls = []
        self.search_results = []
        self.profile_response = {"static": [], "dynamic": [], "search_results": []}
        self.ingest_calls = []
        self.forgotten_ids = []
        self.forget_by_query_response = {"success": True, "message": "Forgot"}

    def add_memory(self, content, metadata=None, *, entity_context="",
                   container_tag=None, custom_id=None, task_type=None):
        self.add_calls.append({
            "content": content,
            "metadata": metadata,
            "entity_context": entity_context,
            "container_tag": container_tag,
            "custom_id": custom_id,
            "task_type": task_type,
        })
        return {"id": "mem_123"}

    def search_memories(self, query, *, limit=5, container_tag=None, search_mode=None):
        self.search_calls.append({"query": query, "container_tag": container_tag, "search_mode": search_mode})
        return self.search_results

    def get_profile(self, query=None, *, container_tag=None):
        self.profile_queries.append(query)
        return self.profile_response

    def forget_memory(self, memory_id, *, container_tag=None):
        self.forgotten_ids.append(memory_id)

    def forget_by_query(self, query, *, container_tag=None):
        return self.forget_by_query_response

    def ingest_conversation(self, session_id, messages, metadata=None):
        self.ingest_calls.append({"session_id": session_id, "messages": messages, "metadata": metadata})


@pytest.fixture
def provider(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    monkeypatch.setattr(
        "plugins.memory.supermemory._call_owner_reranker",
        lambda query, candidates: {
            "selected_ids": [candidate["id"] for candidate in candidates],
            "rejected_ids": [],
            "sufficient": True,
        },
    )
    p = SupermemoryMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    return p


def test_is_available_false_without_api_key(monkeypatch):
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    p = SupermemoryMemoryProvider()
    assert p.is_available() is False


def test_load_and_save_config_round_trip(tmp_path):
    _save_supermemory_config({"container_tag": "demo-tag", "auto_capture": False}, str(tmp_path))
    cfg = _load_supermemory_config(str(tmp_path))
    # container_tag is kept raw — sanitization happens in initialize() after template resolution
    assert cfg["container_tag"] == "demo-tag"
    assert cfg["auto_capture"] is False
    assert cfg["auto_recall"] is True


def test_clean_text_for_capture_strips_injected_context():
    text = "hello\n<supermemory-context>ignore me</supermemory-context>\nworld"
    assert _clean_text_for_capture(text) == "hello\nworld"


def test_format_prefetch_context_deduplicates_overlap():
    result = _format_prefetch_context(
        static_facts=["Jordan prefers short answers"],
        dynamic_facts=["Jordan prefers short answers", "Uses Hermes"],
        search_results=[{"memory": "Uses Hermes", "similarity": 0.9}],
        max_results=10,
    )
    assert result.count("Jordan prefers short answers") == 1
    assert result.count("Uses Hermes") == 1
    assert "<supermemory-context>" in result


def test_owner_prefetch_renders_wikilinks_as_plain_facts():
    result = _format_prefetch_context(
        static_facts=[], dynamic_facts=[],
        search_results=[{"memory": "Dennis likes [[Dishes/Deli Sandwiches|B.L.T. on SDB]] at [[Zipp's]].", "similarity": 0.9}],
        max_results=1, owner_context=True,
    )
    assert "Dennis likes B.L.T. on SDB at Zipp's." in result
    assert "[[" not in result
    assert "Dishes/Deli Sandwiches" not in result


def test_prefetch_includes_profile_on_first_turn(provider):
    provider._client.profile_response = {
        "static": ["Jordan prefers short answers"],
        "dynamic": ["Current project is Supermemory provider"],
        "search_results": [{"memory": "Working on Hermes memory provider", "similarity": 0.88}],
    }
    provider.on_turn_start(1, "start")
    result = provider.prefetch("what am I working on?")
    assert "User Profile (Persistent)" in result
    assert "Recent Context" in result
    assert "Relevant Memories" in result
    assert "authenticated Owner/requester is Dennis" not in result


def test_non_owner_prefetch_retains_user_conversation_evidence(provider):
    provider._client.profile_response = {
        "static": [],
        "dynamic": [],
        "search_results": [
            {"id": "user", "memory": "Jordan likes tea.", "metadata": {"type": "user_conversation"}},
        ],
    }
    result = provider.prefetch("What does Jordan like?")
    assert "Jordan likes tea" in result


def test_owner_primary_prefetch_uses_only_canonical_documents(provider):
    provider._container_tag = "owner_primary"
    provider._client.profile_response = {
        "static": ["Assistant-derived profile claim"],
        "dynamic": ["Assistant-derived recent claim"],
        "search_results": [
            {"id": "bad", "memory": "[role: assistant] Wrong wife", "metadata": {"type": "conversation"}},
            {"id": "good", "memory": "Dennis's wife is Courtnee.", "metadata": {"source": "obsidian", "authority": "canonical"}},
        ],
    }
    provider.on_turn_start(1, "start")
    result = provider.prefetch("Who is my wife?")
    assert "Courtnee" in result
    assert "authenticated Owner/requester is Dennis" in result
    assert "Assistant-derived" not in result
    assert "Wrong wife" not in result


def test_owner_primary_prefetch_resolves_speaker_relative_parent_query(provider):
    provider._container_tag = "owner_primary"
    provider._client.profile_response = {
        "static": [],
        "dynamic": [],
        "search_results": [
            {
                "id": "collision",
                "memory": "Dennis William Malone is Dennis Malone’s paternal uncle; Phillip D. Malone is his sibling.",
                "metadata": {
                    "source": "obsidian",
                    "authority": "canonical",
                    "relative_path": "Jarvis/Family Shared/People/Dennis William Malone.md",
                },
            },
            {
                "id": "father",
                "memory": "- Phillip D. Malone is Dennis Malone’s father.",
                "metadata": {"source": "obsidian", "authority": "canonical"},
            }
        ],
    }

    # Exact fourth-turn sequence from the production regression. Prior recall
    # blocks stay in message history, so each turn must remain a one-result
    # current-query lookup rather than growing static family context.
    result = ""
    for turn, query in enumerate(
        ["Who am I?", "Who is my wife?", "What is my favorite order at Ike's?", "Who is my dad?"],
        start=1,
    ):
        provider.on_turn_start(turn, query)
        result = provider.prefetch(query)

    assert provider._client.profile_queries[-1] == (
        "Who is my dad? Authenticated requester: Dennis Malone. "
        "Resolve Dennis Malone parent relationship: father or dad."
    )
    assert "Phillip D. Malone is Dennis Malone’s father" in result
    assert result.count("## Relevant Memories") == 1


def test_owner_primary_fails_closed_without_canonical_evidence(provider):
    provider._container_tag = "owner_primary"
    provider._client.profile_response = {
        "static": ["Profile claim"],
        "dynamic": [],
        "search_results": [
            {"memory": "Unmarked claim: somebody is Dennis Malone’s father."},
            {"memory": "[role: assistant] Invented father", "metadata": {"type": "conversation"}},
        ],
    }
    assert provider.prefetch("Who is my dad?") == ""


def test_owner_parent_ranking_rejects_reversed_indirect_and_negated_claims():
    from plugins.memory.supermemory import _rank_owner_canonical_results

    canonical = {"source": "obsidian", "authority": "canonical"}
    correct = {"memory": "- Phillip is Dennis Malone’s father.", "metadata": canonical}
    distractors = [
        {"memory": "- Dennis Malone is Alex’s father.", "metadata": canonical},
        {"memory": "- Dennis Malone’s wife’s father is Robert.", "metadata": canonical},
        {"memory": "- Dennis Malone is not the father of Alex.", "metadata": canonical},
        {"memory": "- Robert is Dennis Malone’s father’s brother.", "metadata": canonical},
        {"memory": "- It is false that Robert is Dennis Malone’s father.", "metadata": canonical},
        {"memory": "- Robert D. Malone is Dennis Malone’s father, but that claim is false.", "metadata": canonical},
        {"memory": "- It is untrue that Robert is Dennis Malone’s father.", "metadata": canonical},
        {"memory": "- Robert is Dennis Malone’s father, which is incorrect.", "metadata": canonical},
        {
            "memory": "- Father: Not Robert",
            "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/People/Dennis Malone.md"},
        },
    ]
    ranked = _rank_owner_canonical_results("Who is my dad?", distractors + [correct])
    assert ranked[0] is correct


def test_owner_parents_plural_ranks_direct_parent_evidence():
    from plugins.memory.supermemory import _rank_owner_canonical_results

    canonical = {"source": "obsidian", "authority": "canonical"}
    correct = {"memory": "- Phillip is Dennis Malone’s father.", "metadata": canonical}
    unrelated = {"memory": "Canonical family retrieval policy.", "metadata": canonical}
    assert _rank_owner_canonical_results("Who are my parents?", [unrelated, correct])[0] is correct


def test_owner_parent_query_does_not_rewrite_indirect_relationship():
    from plugins.memory.supermemory import _owner_canonical_query

    query = "Who is my wife's father?"
    assert _owner_canonical_query(query) == query
    query = "Who is my father's brother?"
    assert _owner_canonical_query(query) == query
    query = "Who is my father-in-law?"
    assert _owner_canonical_query(query) == query
    assert "Resolve Dennis Malone parent relationship: father or dad." in _owner_canonical_query("What is my dad's name?")


def test_owner_mother_in_law_query_ranks_direct_profile_without_rewrite():
    from plugins.memory.supermemory import _owner_canonical_query, _rank_owner_canonical_results

    query = "Tell me about my mother-in-law"
    assert _owner_canonical_query(query) == query
    canonical = {"source": "obsidian", "authority": "canonical"}
    generic = {"memory": "Generic household summary", "metadata": canonical}
    courtnee = {
        "memory": "## Parents\n- Mother: [[Mary Pat Thompson]]",
        "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/People/Courtnee Malone.md"},
    }
    mary = {
        "memory": "# Mary Pat Thompson\n- Son-in-law: Dennis Malone\n- She enjoyed reading and crocheting.",
        "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/People/Mary Pat Thompson.md"},
    }
    assert _rank_owner_canonical_results(query, [generic, courtnee, mary]) == [mary, courtnee, generic]


@pytest.mark.parametrize("query", [
    "What does Courtnee normally get at Zips?",
    "What does Dennis get at Zipp's?",
    "What does the family think about Zipp’s?",
])
def test_owner_named_restaurant_recall_keeps_only_exact_note_and_reciprocal_dishes(provider, query):
    provider._container_tag = "owner_primary"
    canonical = {"source": "obsidian", "authority": "canonical"}
    zipps = {"memory": "restaurant: Zipp's\n### Courtnee\n- Mozzarella Sticks — prefers ranch.", "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/Food/Restaurants/Zipp's.md"}}
    mozzarella = {"memory": "# Mozzarella Sticks\n- [[Restaurants/Zipp's]] — Courtnee likes them with ranch.", "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/Food/Dishes/Mozzarella Sticks.md"}}
    parlay = {"memory": "restaurant: Parlay\nCourtnee ordered the Honey Hot Chicken Sandwich.", "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/Food/Restaurants/Parlay.md"}}
    chicken = {"memory": "# Chicken Sandwiches\n- [[Restaurants/Parlay]] — Courtnee rated it 3/5.", "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/Food/Dishes/Chicken Sandwiches.md"}}
    provider._client.profile_response = {"static": [], "dynamic": [], "search_results": [parlay, chicken, mozzarella, zipps]}

    result = provider.prefetch(query)

    assert "prefers ranch" in result
    assert "Mozzarella Sticks" in result
    assert "Honey Hot Chicken Sandwich" not in result
    assert "3/5" not in result
    assert result.index("restaurant: Zipp's") < result.index("# Mozzarella Sticks")


def test_owner_recall_keeps_only_requested_person_section(provider):
    provider._container_tag = "owner_primary"
    canonical = {"source": "obsidian", "authority": "canonical"}
    provider._client.profile_response = {"static": [], "dynamic": [], "search_results": [{
        "id": "ikes",
        "memory": "restaurant: Ike's\n### Dennis\n- Madison Bumgarner on sourdough.\n### Lauren\n- Ike's Reuben.",
        "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/Food/Restaurants/Ike's.md"},
    }]}

    result = provider.prefetch("What is my favorite order at Ike's?")

    assert "Madison Bumgarner on sourdough" in result
    assert "Ike's Reuben" not in result


def test_owner_named_parlay_recall_excludes_zipps_person_collision(provider):
    provider._container_tag = "owner_primary"
    canonical = {"source": "obsidian", "authority": "canonical"}
    provider._client.profile_response = {"static": [], "dynamic": [], "search_results": [
        {"memory": "restaurant: Zipp's\nCourtnee likes mozzarella sticks with ranch.", "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/Food/Restaurants/Zipp's.md"}},
        {"memory": "# Chicken Sandwiches\n- [[Restaurants/Parlay]] — Courtnee rated the Honey Hot Chicken Sandwich 3/5.", "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/Food/Dishes/Chicken Sandwiches.md"}},
        {"memory": "restaurant: Parlay\nCourtnee liked the sweet heat but disliked the breading and bun.", "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/Food/Restaurants/Parlay.md"}},
    ]}

    result = provider.prefetch("What did Courtnee think about Parlay?")

    assert "sweet heat" in result
    assert "Honey Hot Chicken Sandwich" in result
    assert "mozzarella sticks" not in result


def test_owner_named_restaurant_query_targets_canonical_venue_records():
    from plugins.memory.supermemory import _owner_canonical_query

    assert _owner_canonical_query("What does Courtnee normally get at Zips?") == (
        "What does Courtnee normally get at Zips? Canonical restaurant venue: Zips. "
        "Prefer the exact Food/Restaurants note and reciprocal Food/Dishes records for Zips; exclude other venues."
    )
    # Ambiguous "about <name>" text is not forced into restaurant scope.
    assert _owner_canonical_query("What did Courtnee think about Parlay?") == "What did Courtnee think about Parlay?"
    assert _owner_canonical_query("Tell me about Philip B Malone") == "Tell me about Philip B Malone"


def test_owner_explicit_person_scope_requires_matching_canonical_name():
    from plugins.memory.supermemory import _scope_owner_named_person_results

    canonical = {"source": "obsidian", "authority": "canonical"}
    phillip = {"memory": "Full name: Phillip Daniel Malone", "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/People/Phillip D. Malone.md"}}
    noise = {"memory": "Unrelated family relationship index", "metadata": {**canonical, "relative_path": "Jarvis/Family Shared/People/Malone Family Relationships.md"}}
    assert _scope_owner_named_person_results("Tell me about Phillip D Malone", [noise, phillip]) == [phillip]
    assert _scope_owner_named_person_results("Tell me about Philip B Malone", [noise, phillip]) == []
    assert _scope_owner_named_person_results("Who is my dad?", [noise, phillip]) == [noise, phillip]


def test_sync_turn_buffers_short_messages(provider):
    provider.sync_turn("ok", "sure", session_id="session-1")
    assert len(provider._client.add_calls) == 1


def test_sync_turn_writes_only_user_claims_with_stable_id(provider):
    messages = [
        {"role": "system", "content": "private system prompt"},
        {"role": "user", "content": "First ordinary user message"},
        {"role": "assistant", "content": "First ordinary assistant reply"},
        {"role": "tool", "content": "private tool output"},
    ]
    provider.sync_turn("ignored", "ignored", session_id="session-1", messages=messages)
    first = provider._client.add_calls[-1]
    assert first["custom_id"] == "hermes-session:session-1"
    assert first["task_type"] == "memory"
    assert first["entity_context"] == provider._entity_context
    assert "private system prompt" not in first["content"]
    assert "private tool output" not in first["content"]
    assert "[role: user]\nFirst ordinary user message\n[user:end]" in first["content"]
    assert "First ordinary assistant reply" not in first["content"]

    messages += [
        {"role": "user", "content": "Second message"},
        {"role": "assistant", "content": "<supermemory-context>injected</supermemory-context>Second reply"},
    ]
    provider.sync_turn("ignored", "ignored", session_id="session-1", messages=messages)
    second = provider._client.add_calls[-1]
    assert second["custom_id"] == first["custom_id"]
    assert first["content"] in second["content"]
    assert "injected" not in second["content"]
    assert "Second reply" not in second["content"]
    assert second["metadata"] == {
        "type": "user_conversation",
        "session_id": "session-1",
        "message_count": 2,
    }


def test_owner_primary_routes_automatic_capture_to_conversations(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"container_tag": "owner_primary", "auto_capture": True}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("owner-session", hermes_home=str(tmp_path), platform="cli")
    assert p._auto_capture is True
    p.sync_turn(
        "Dennis prefers concise status updates.",
        "I will keep updates concise.",
        session_id="owner-session",
    )
    assert len(p._client.add_calls) == 1
    call = p._client.add_calls[0]
    assert call["container_tag"] == "owner_conversations"
    assert call["container_tag"] != "owner_primary"
    assert call["custom_id"] == "hermes-owner-conversation:owner-session"
    assert call["task_type"] == "memory"
    assert "[role: user]" in call["content"]
    assert "[role: assistant-context]" not in call["content"]
    assert "I will keep updates concise" not in call["content"]
    assert call["metadata"]["message_count"] == 1
    assert call["metadata"]["provenance"] == "user-authored role-delimited statement"
    assert call["metadata"]["authority"] == "non-authoritative"


def test_capture_owner_app_turn_uses_request_scoped_id_and_existing_filters(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"container_tag": "owner_primary", "auto_capture": True}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("bootstrap", hermes_home=str(tmp_path), platform="cli")

    assert p.capture_owner_app_turn(
        "app-session", "request-42", "I prefer aisle seats on flights.", "I’ll remember that preference."
    ) is True
    call = p._client.add_calls[-1]
    assert call["container_tag"] == "owner_conversations"
    assert call["custom_id"] == "jarvis-owner-app:app-session:request-42"
    assert call["task_type"] == "memory"
    assert "[role: user]\nI prefer aisle seats on flights.\n[user:end]" in call["content"]
    assert "assistant-context" not in call["content"]
    assert "remember that preference" not in call["content"]
    assert call["metadata"]["message_count"] == 1
    assert call["metadata"]["provenance"] == "user-authored role-delimited statement"
    assert call["metadata"]["capture_source"] == "jarvis_owner_app"

    before = len(p._client.add_calls)
    assert p.capture_owner_app_turn("app-session", "request-question", "Where am I?", "At home.") is False
    assert len(p._client.add_calls) == before


def test_owner_capture_filters_questions_commands_and_test_probes(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"container_tag": "owner_primary", "auto_capture": True}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("owner-session", hermes_home=str(tmp_path), platform="cli")
    for user, assistant in [
        ("Who is my father?", "Your father is Example Person."),
        ("Run the focused tests now", "The tests passed."),
        ("This is a test probe; answer banana", "banana"),
    ]:
        p.sync_turn(user, assistant, session_id="owner-session")
    assert p._client.add_calls == []


def test_owner_capture_respects_disabled_toggle(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"container_tag": "owner_primary", "auto_capture": False}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("owner-session", hermes_home=str(tmp_path), platform="cli")
    p.sync_turn("Dennis prefers concise status updates.", "Understood.", session_id="owner-session")
    assert p._auto_capture is False
    assert p._client.add_calls == []


def test_owner_prefetch_reranker_selects_valid_evidence(provider, monkeypatch):
    provider._container_tag = "owner_primary"
    canonical = {"source": "obsidian", "authority": "canonical"}
    provider._client.profile_response = {"static": [], "dynamic": [], "search_results": [
        {"id": "c1", "memory": "Alex prefers tea.", "metadata": canonical},
    ]}
    provider._client.search_results = [
        {"id": "u1", "memory": "[role: user]\nI prefer coffee.\n[user:end]", "metadata": {"authority": "non-authoritative", "source": "conversation", "speaker": "user"}},
    ]
    monkeypatch.setattr(provider, "_rerank_owner_candidates", lambda query, items: [items[0]])
    result = provider.prefetch("What does Alex prefer?")
    assert provider._client.profile_queries[-1] == "What does Alex prefer?"
    assert provider._client.search_calls[-1]["container_tag"] == "owner_conversations"
    assert "Alex prefers tea" in result
    assert "I prefer coffee" not in result


def test_owner_prefetch_malformed_reranker_fails_closed(provider, monkeypatch):
    provider._container_tag = "owner_primary"
    provider._client.profile_response = {"static": [], "dynamic": [], "search_results": [
        {"id": "c1", "memory": "Claim one", "metadata": {"source": "obsidian", "authority": "canonical"}},
        {"id": "c2", "memory": "Claim two", "metadata": {"source": "obsidian", "authority": "canonical"}},
    ]}
    monkeypatch.setattr("plugins.memory.supermemory._call_owner_reranker", lambda *args, **kwargs: {"selected_ids": ["unknown"], "rejected_ids": ["c1"], "sufficient": True})
    assert provider.prefetch("Which claim is right?") == ""


def test_owner_reranker_uses_fixed_local_model_independent_of_answer_model(monkeypatch):
    from plugins.memory.supermemory import _call_owner_reranker, _OWNER_RERANK_SYSTEM

    assert "usual, normal, favorite, or repeated" in _OWNER_RERANK_SYSTEM
    assert "Next-time" in _OWNER_RERANK_SYSTEM
    assert "may-order" in _OWNER_RERANK_SYSTEM
    assert "another person's" in _OWNER_RERANK_SYSTEM

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"message": {"content": json.dumps({
                "selected_ids": ["c1"], "rejected_ids": ["c2"], "sufficient": True,
            })}}).encode()

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("plugins.memory.supermemory.urllib.request.urlopen", fake_urlopen)
    result = _call_owner_reranker("Choose", [{"id": "c1"}, {"id": "c2"}])
    assert result["selected_ids"] == ["c1"]
    assert captured["payload"]["model"] == "qwen3.5:4b"
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"] == {
        "num_ctx": 16384,
        "temperature": 0,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
        "presence_penalty": 0,
        "num_predict": 300,
    }
    assert captured["url"] == "http://mcomen.malonecentral.com:11434/api/chat"
    assert captured["timeout"] == 4.0


def test_sync_turn_fallback_accumulates_turns_and_isolates_sessions(provider):
    provider.sync_turn("session one user", "session one reply", session_id="session-1")
    provider.on_session_switch("session-2")
    provider.sync_turn("session two user", "session two reply", session_id="session-2")
    assert [call["custom_id"] for call in provider._client.add_calls] == [
        "hermes-session:session-1",
        "hermes-session:session-2",
    ]
    assert "session one" not in provider._client.add_calls[-1]["content"]


def test_resumed_session_uses_same_custom_id(provider):
    provider.sync_turn("one", "reply one", session_id="session-1")
    provider.initialize("session-1", hermes_home=provider._hermes_home, platform="cli")
    provider.sync_turn("two", "reply two", session_id="session-1")
    assert provider._client.add_calls[-1]["custom_id"] == "hermes-session:session-1"


def test_on_session_end_does_not_duplicate_completed_turn_capture(provider):
    messages = [
        {"role": "system", "content": "skip"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    provider.sync_turn("hello", "hi there", session_id="session-1", messages=messages)
    assert len(provider._client.add_calls) == 1
    provider.on_session_end(messages)
    provider.shutdown()
    assert len(provider._client.add_calls) == 1
    assert provider._client.ingest_calls == []
    assert provider._session_turns == []


def test_auto_capture_disabled_blocks_all_automatic_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"auto_capture": False}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("disabled-1", hermes_home=str(tmp_path), platform="cli")
    p.sync_turn("user", "assistant", session_id="disabled-1")
    p.on_session_end([{"role": "user", "content": "user"}])
    p.on_session_switch("disabled-2")
    p.shutdown()
    assert p._client.add_calls == []
    assert p._client.ingest_calls == []


def test_explicit_approved_store_is_not_blocked_by_auto_capture(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"auto_capture": False}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    result = json.loads(p.handle_tool_call("supermemory_store", {"content": "Approved explicit fact"}))
    assert result["saved"] is True
    assert len(p._client.add_calls) == 1


def test_merge_metadata_stamps_sm_source():
    # sm_source routes Hermes writes into the "Hermes" Space in the Supermemory
    # app (functional routing, not telemetry) — must always be present.
    from plugins.memory.supermemory import _SupermemoryClient

    client = _SupermemoryClient.__new__(_SupermemoryClient)
    merged = client._merge_metadata({"type": "explicit_memory"})
    assert merged["sm_source"] == "hermes"
    assert merged["type"] == "explicit_memory"

    # Legacy "source" is migrated into "type" when type is absent.
    merged2 = client._merge_metadata({"source": "conversation_turn"})
    assert merged2["sm_source"] == "hermes"
    assert merged2["type"] == "conversation_turn"
    assert "source" not in merged2


def test_shutdown_joins_threads_without_duplicate_capture(provider, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_add_memory(content, metadata=None, *, entity_context="",
                        container_tag=None, custom_id=None, task_type=None):
        started.set()
        release.wait(timeout=1)
        provider._client.add_calls.append({
            "content": content,
            "metadata": metadata,
            "entity_context": entity_context,
        })
        return {"id": "mem_slow"}

    monkeypatch.setattr(provider._client, "add_memory", slow_add_memory)

    provider.sync_turn(
        "Please remember this request in long-term memory",
        "Absolutely, I will keep that in long-term memory.",
        session_id="session-1",
    )
    assert provider._sync_thread is None
    assert len(provider._client.add_calls) == 1

    # on_memory_write still runs on a background thread.
    provider.on_memory_write("add", "memory", "Jordan likes concise docs")
    assert started.wait(timeout=1)
    assert provider._write_thread is not None

    release.set()
    provider.shutdown()

    # All tracked threads joined and cleared.
    assert provider._sync_thread is None
    assert provider._write_thread is None
    assert provider._prefetch_thread is None
    # Explicit memory write went through.
    assert len(provider._client.add_calls) == 2
    assert provider._client.ingest_calls == []


def test_store_tool_returns_saved_payload(provider):
    result = json.loads(provider.handle_tool_call("supermemory_store", {"content": "Jordan likes concise docs"}))
    assert result["saved"] is True
    assert result["id"] == "mem_123"


def test_search_tool_formats_results(provider):
    provider._client.search_results = [
        {"id": "m1", "memory": "Jordan likes concise docs", "similarity": 0.92}
    ]
    result = json.loads(provider.handle_tool_call("supermemory_search", {"query": "concise docs"}))
    assert result["count"] == 1
    assert result["results"][0]["similarity"] == 92


def test_forget_tool_by_id(provider):
    result = json.loads(provider.handle_tool_call("supermemory_forget", {"id": "m1"}))
    assert result == {"forgotten": True, "id": "m1"}
    assert provider._client.forgotten_ids == ["m1"]


def test_profile_tool_formats_sections(provider):
    provider._client.profile_response = {
        "static": ["Jordan prefers concise docs"],
        "dynamic": ["Working on Supermemory provider"],
        "search_results": [],
    }
    result = json.loads(provider.handle_tool_call("supermemory_profile", {}))
    assert result["static_count"] == 1
    assert result["dynamic_count"] == 1
    assert "User Profile (Persistent)" in result["profile"]


def test_handle_tool_call_returns_error_when_unconfigured(monkeypatch):
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    p = SupermemoryMemoryProvider()
    result = json.loads(p.handle_tool_call("supermemory_search", {"query": "x"}))
    assert "error" in result


# -- Identity template tests --------------------------------------------------


def test_identity_template_resolved_in_container_tag(monkeypatch, tmp_path):
    """container_tag with {identity} resolves to profile-scoped tag."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"container_tag": "hermes-{identity}"}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli", agent_identity="coder")
    assert p._container_tag == "hermes_coder"


def test_container_tag_env_var_override(monkeypatch, tmp_path):
    """SUPERMEMORY_CONTAINER_TAG env var overrides config."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setenv("SUPERMEMORY_CONTAINER_TAG", "env-override")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._container_tag == "env_override"


# -- Search mode tests --------------------------------------------------------


def test_invalid_search_mode_falls_back_to_default(monkeypatch, tmp_path):
    """Invalid search_mode falls back to 'hybrid'."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"search_mode": "invalid_mode"}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._search_mode == "hybrid"


# -- Base URL tests -------------------------------------------------------------


def test_base_url_defaults_to_cloud(monkeypatch, tmp_path):
    """Without config or env override, the client targets api.supermemory.ai."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.delenv("SUPERMEMORY_BASE_URL", raising=False)
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._base_url == "https://api.supermemory.ai"
    assert p._client.base_url == "https://api.supermemory.ai"


def test_client_passes_custom_base_url_to_sdk(monkeypatch):
    """SDK operations use the normalized custom base URL."""
    import sys
    import types

    from plugins.memory.supermemory import _SupermemoryClient

    captured = {}

    class StubSupermemory:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = types.ModuleType("supermemory")
    module.Supermemory = StubSupermemory
    monkeypatch.setitem(sys.modules, "supermemory", module)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *args, **kwargs: None)

    client = _SupermemoryClient(
        api_key="test-key",
        timeout=1.0,
        container_tag="hermes",
        base_url="http://localhost:6767/",
    )

    assert client._base_url == "http://localhost:6767"
    assert captured["base_url"] == "http://localhost:6767"


def test_add_memory_passes_explicit_task_type_to_documents_add():
    from plugins.memory.supermemory import _SupermemoryClient
    client = _SupermemoryClient.__new__(_SupermemoryClient)
    client._container_tag = "hermes"
    captured = {}
    class Docs:
        def add(self, **kwargs):
            captured.update(kwargs)
            return type("Result", (), {"id": "doc-1"})()
    client._client = type("SDK", (), {"documents": Docs()})()
    client.add_memory("conversation", task_type="memory")
    assert captured["task_type"] == "memory"


# -- Multi-container tests ----------------------------------------------------


def test_multi_container_disabled_by_default(provider):
    """Multi-container is off by default; schemas have no container_tag param."""
    assert provider._enable_custom_containers is False
    schemas = provider.get_tool_schemas()
    for s in schemas:
        assert "container_tag" not in s["parameters"]["properties"]


def test_get_config_schema_minimal():
    """get_config_schema only returns the API key field."""
    p = SupermemoryMemoryProvider()
    schema = p.get_config_schema()
    assert len(schema) == 1
    assert schema[0]["key"] == "api_key"
    assert schema[0]["secret"] is True


def test_probe_supermemory_connection_missing_key(tmp_path):
    status = _probe_supermemory_connection("", str(tmp_path))
    assert status["ok"] is False
    assert status["error"] == "SUPERMEMORY_API_KEY not set"
    assert status["container_tag"] == "hermes"


def _stub_supermemory_importable(monkeypatch):
    """Make ``__import__("supermemory")`` succeed without the real package.

    ``_probe_supermemory_connection`` guards on ``__import__("supermemory")``
    before using the (mocked) client, so tests that mock ``_SupermemoryClient``
    must also satisfy that import guard — otherwise they only pass in an
    environment where the optional ``supermemory`` package happens to be
    installed (and fail on a clean checkout / CI). Mirrors the inverse stub in
    ``test_is_available_false_when_import_missing``.
    """
    import builtins
    import types

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "supermemory":
            return types.ModuleType("supermemory")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_post_setup_writes_config_and_prints_summary(monkeypatch, tmp_path, capsys):
    config: dict = {"memory": {}}
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "")
    monkeypatch.setattr(
        "hermes_cli.memory_setup._prompt",
        lambda label, secret=True, default=None: "new-api-key",
    )
    monkeypatch.setattr(
        "plugins.memory.supermemory._probe_supermemory_connection",
        lambda api_key, hermes_home, **kwargs: {
            "ok": True,
            "container_tag": "hermes",
            "profile_facts": 3,
            "auto_recall": True,
            "auto_capture": True,
        },
    )

    saved: dict = {}

    def fake_save_config(cfg):
        saved.update(cfg)

    monkeypatch.setattr("hermes_cli.config.save_config", fake_save_config)

    SupermemoryMemoryProvider().post_setup(str(tmp_path), config)

    assert config["memory"]["provider"] == "supermemory"
    assert saved["memory"]["provider"] == "supermemory"
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "SUPERMEMORY_API_KEY=new-api-key" in env_text

    out = capsys.readouterr().out
    assert "✓ Connected" in out
    assert "3 profile facts" in out
    assert "Memory provider: supermemory" in out


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits not enforced on Windows")
def test_save_config_sets_owner_only_permissions(tmp_path):
    """supermemory.json must be written with 0o600 so API key is not world-readable."""
    _save_supermemory_config({"api_key": "sm-test-key"}, str(tmp_path))
    config_file = tmp_path / "supermemory.json"
    assert config_file.exists()
    mode = stat.S_IMODE(config_file.stat().st_mode)
    assert mode == 0o600, f"Expected 0o600 (owner-only), got {oct(mode)}"
