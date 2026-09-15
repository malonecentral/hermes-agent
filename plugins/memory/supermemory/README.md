# Supermemory Memory Provider

Semantic long-term memory with profile recall, semantic search, explicit memory tools, and full-session conversation ingest (one ingest per session) for richer profiles.

## Requirements

- `pip install supermemory`
- Hosted: API key from [app.supermemory.ai/integrations?connect=hermes](http://app.supermemory.ai/integrations?connect=hermes)
- Self-hosted: a running [Supermemory local](https://supermemory.ai/docs/self-hosting/overview) server and the API key it prints on first boot

## Setup

```bash
hermes memory setup    # select "supermemory"
```

Or manually:

```bash
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=***' >> ~/.hermes/.env
```

For a fully self-hosted setup, start Supermemory local and note the API key it
prints on first boot:

```bash
npx supermemory local
```

Before running `hermes memory setup`, add the local endpoint to
`$HERMES_HOME/supermemory.json`:

```json
{
  "base_url": "http://localhost:6767"
}
```

Then run `hermes memory setup` and enter the local server's API key. Configuring
the endpoint first ensures the setup connection probe also stays local.

## Config

Config file: `$HERMES_HOME/supermemory.json`

| Key | Default | Description |
|-----|---------|-------------|
| `base_url` | `https://api.supermemory.ai` | API endpoint for hosted or self-hosted Supermemory. Takes priority over `SUPERMEMORY_BASE_URL`. |
| `container_tag` | `hermes` | Container tag used for search and writes. Supports `{identity}` template for profile-scoped tags (e.g. `hermes-{identity}` → `hermes-coder`). |
| `auto_recall` | `true` | Inject relevant memory context before turns |
| `auto_capture` | `true` | Store cleaned user-assistant turns after each response |
| `temporal_filters_schema_v4_ready` | `false` | Request temporal filtering; it activates only when the local import receipt is schema v4, reconciled, has the expected backend count, and every eligible row is terminal `done`/v4 with zero failures or pending work. A v3 or incomplete receipt stays off. |
| `context_char_budget` | `12000` | Maximum final memory-envelope characters; runtime context measurement may lower it. |
| `context_byte_budget` | `24000` | Maximum UTF-8 bytes in the final memory envelope. Truncation is deterministic and preserves the envelope and authority labels. |
| `reranker_input_token_budget` | `8192` | Measured Qwen reranker context. The request builder reserves 768 tokens for the runner template and bounds each query/document pair by UTF-8 bytes (safe for byte-fallback/pathological Unicode). This is independent of the final context budget. |
| `max_recall_results` | `5` | Max recalled items to format into context (hard configuration clamp: 20); no authority-specific quota is applied. |
| `profile_frequency` | `50` | Include profile facts on first turn and every N turns |
| `capture_mode` | `all` | Skip tiny or trivial turns by default |
| `search_mode` | `hybrid` | Conversation search mode: `hybrid`, `memories`, or `documents`. |
| `canonical_document_search_mode` | `documents` | Canonical retrieval uses `/v4/search` in document-only mode; other values normalize to `documents`. Independent of conversation search. |
| `entity_context` | built-in default | Extraction guidance passed to Supermemory |
| `api_timeout` | `5.0` | Timeout for SDK and ingest requests |
| `owner_canonical_container` | `owner_primary` | Validated Owner canonical-document topology tag. Contract-only in A1; current retrieval behavior is unchanged. |
| `owner_explicit_container` | `owner_primary` | Validated Owner explicit-memory topology tag. Contract-only in A1. |
| `family_shared_container` | `family_shared` | Validated Family Shared canonical-document topology tag. Contract-only in A1. |
| `owner_conversation_container` | `owner_conversations` | Validated Owner conversation topology tag. Contract-only in A1. |
| `routing_projection_enabled` | `false` | A2 feature flag. When enabled, retrieval uses the validated projected containers below; when disabled, the legacy routing path is unchanged. |
| `requester_conversation_projection` | `false` | Enables requester-specific Family conversation routing when trusted identity projection and a namespace key are configured. |
| `requester_conversation_capture` | `false` | Reserved compatibility flag; the plugin always forces it off. Family requester conversation capture is owned by the private executor. |
| `requester_identity_server` | empty | MCP server name whose request-local `jarvisRequester.person_id` projection is supplied by the authenticated registry integration. Display names and local registry files are never accepted. |
| `requester_conversation_namespace_key` | empty | At least 32 UTF-8 bytes used as the HMAC-SHA256 namespace key for opaque stable container tags. Keep `$HERMES_HOME/supermemory.json` owner-only. Missing/invalid keys disable the capability. |

With `routing_projection_enabled: true`, Owner recall independently queries the
projected private-canonical, Family Shared canonical, explicit-memory, and Owner
conversation containers. Only rows validated against the container queried and
their expected source shape enter the common Qwen reranker. Explicit memories
remain non-authoritative and are admitted after scoring only when they clear the
same post-Qwen relevance threshold as other noncanonical evidence and carry
locally assigned `explicit_memory` provenance; provider response text or
metadata cannot create that trust. Family projected recall remains canonical
Family Shared plus an optional authenticated requester-conversation projection.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SUPERMEMORY_API_KEY` | API key (required) |
| `SUPERMEMORY_BASE_URL` | Compatibility fallback for the API endpoint when `base_url` is not configured |
| `SUPERMEMORY_CONTAINER_TAG` | Override container tag (takes priority over config file) |

Base URL precedence is `supermemory.json` → `SUPERMEMORY_BASE_URL` →
`https://api.supermemory.ai`. Hermes resolves it once and uses the same endpoint
for SDK operations, setup/status probes, and full-session conversation ingest.

## Tools

Kebab-case names are registered for the agent; snake_case aliases remain supported.

| Tool | Alias | Description |
|------|-------|-------------|
| `supermemory-save` | `supermemory_store` | Store an explicit memory |
| `supermemory-search` | `supermemory_search` | Search memories by semantic similarity |
| `supermemory-forget` | `supermemory_forget` | Forget a memory by ID or best-match query |
| `supermemory-profile` | `supermemory_profile` | Retrieve persistent profile and recent context |

## Source attribution

All Supermemory API calls send `x-sm-source: hermes`, and document writes stamp
`metadata.sm_source: hermes`. This is a **functional routing key, not telemetry**:
it groups Hermes-written memories into a dedicated "Hermes" Space in the
Supermemory app, so you can filter, browse, and bulk-manage them per source agent
(alongside Codex, Claude Code, etc.) from the Supermemory UI.

## Behavior

When enabled, Hermes can:

- prefetch relevant memory context before each turn
- retrieve up to 20 candidates independently per authorized source. Legacy routing searches two canonical containers plus Owner conversation (at most 60 candidates). A2 projected routing additionally searches the explicit-memory container (at most 80 candidates). After source-shape, provenance, ACL, entity, temporal, and deduplication guards, one Qwen request scores the full eligible pool. The existing single-candidate bypass, Qwen score behavior, `max_recall_results`, and final context byte/character budgets are preserved.
- emit structured `supermemory_stage` receipts containing stage names, outcomes, counts, limits, and durations; receipts contain no queries, content, credentials, or source/requester/session identifiers.
- buffer the full conversation and ingest it as **one session** at session end (or on `/reset`, branch, compression, or shutdown)
- ingest the full session to the conversations endpoint for richer profile/graph updates
- route every SDK, probe, and conversation-ingest request through the configured hosted or self-hosted endpoint
- expose explicit tools for search, store, forget, and profile access

The session is written once via the conversations endpoint, which drives Supermemory's entity extraction and profile building while keeping a clean, retrievable full transcript.

## Profile-Scoped Containers

Use `{identity}` in the `container_tag` to scope memories per Hermes profile:

```json
{
  "container_tag": "hermes-{identity}"
}
```

For a profile named `coder`, this resolves to `hermes-coder`. The default profile resolves to `hermes-default`. Without `{identity}`, all profiles share the same container.

## Multi-Container Mode

For advanced setups (e.g. OpenClaw-style multi-workspace), you can enable custom container tags so the agent can read/write across multiple named containers:

```json
{
  "container_tag": "hermes",
  "enable_custom_container_tags": true,
  "custom_containers": ["project-alpha", "project-beta", "shared-knowledge"],
  "custom_container_instructions": "Use project-alpha for coding tasks, project-beta for research, and shared-knowledge for team-wide facts."
}
```

When enabled:
- `supermemory-search`, `supermemory-save`, `supermemory-forget`, and `supermemory-profile` accept an optional `container_tag` parameter
- The tag must be in the whitelist: primary container + `custom_containers`
- Automatic operations (turn sync, prefetch, memory write mirroring, session ingest) always use the **primary** container only
- Custom container instructions are injected into the system prompt

## Support

- [Supermemory Discord](https://supermemory.link/discord)
- [support@supermemory.com](mailto:support@supermemory.com)
- [supermemory.ai](https://supermemory.ai)
