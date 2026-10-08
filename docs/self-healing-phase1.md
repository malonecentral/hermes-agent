# Phase 1 incident observations

Collection is **off by default**. To record observations, set in the active
profile's `config.yaml`:

```yaml
self_healing:
  enabled: true
```

The effective managed configuration wins over user settings; an administrator's
`self_healing.enabled: false` disables capture even when the user enables it.
The config read does not initialize Hermes home or back up malformed user config.

This enables recording only. This increment observes exceptions caught by
`ToolRegistry.dispatch`, not returned error objects, provider failures elsewhere,
process-wide exceptions, or deliberate failure injection. Existing logging,
sanitization and tool-error returns are unchanged. Recording failures are silent
and never change tool behavior.

The separate ledger is at `get_hermes_home()/self-healing/incidents.db`, not
`state.db`. Every operation closes its own connection. Writes atomically append
an occurrence and update the aggregate; schema initialization is serialized.
SQLite's existing/default journal mode is retained (WAL is not forced). Dispatch
capture uses a 50 ms SQLite busy budget for writer contention and silently drops
the observation on timeout rather than delaying tool-error handling for seconds.
Direct `record()` callers retain the normal 2-second write budget.
New ledger directories/files use 0700/0600 permissions. Unsafe existing symlinks,
non-regular files, hardlinks, foreign-owned or group/world-writable targets are
rejected, never chmodded. Filesystem permissions remain platform-dependent.

Only explicit scalar metadata is accepted: tool, component, operation, exception
type, stable code, bounded message and correlation ID. Dispatch currently supplies
its task ID when available and a static source. No arguments, results, prompts,
tracebacks, locals, exception objects or arbitrary payload dictionaries enter the
ledger API. All scalar metadata is force-redacted (including URL credentials)
before truncation, hashing or SQL binding, even if normal logging redaction is
off. Redaction failure omits the observation. The ledger adds no logging; existing
registry exception logs are outside its privacy boundary and remain unchanged.
Redaction is pattern-based, not a guarantee of removing all personal information;
exception messages may contain residual sensitive prose. Protect the database as
private diagnostic data. There is no automatic retention/deletion in Phase 1.

Versioned canonical SHA-256 fingerprints exclude timestamps and correlation IDs,
normalize whitespace, UUIDs, PIDs, temporary paths and addresses, and preserve
stable identifying fields. Classification is descriptive only: ordered external,
network, upstream, rate-limit, auth, cancellation and deliberate-test deferrals;
positively established local/internal/schema/config/import/protocol facts can be
classified for investigation by the pure classifier. **Dispatch provides no such
provenance**, so ambiguous errors go to review. Words inside messages never
establish provenance. No classification grants execution authority.

Read an existing ledger without creating or migrating anything:

```sh
python -m agent.incident_report --limit 100 --examples 3
```

Uses the active `HERMES_HOME`, outputs JSON, orders newest aggregates first with
fingerprint tie-breaking, and bounds results to 1–1000 aggregates and 0–10
occurrences per aggregate. A missing database prints `[]`. Unsupported schema
versions are rejected rather than migrated or overwritten.

Reporting is an **offline/quiescent read**, not an online snapshot: stop ledger
writers before reporting. The connection uses `mode=ro&immutable=1`, so SQLite
does not create or modify database sidecars or acquire database locks. WAL-mode
databases are rejected even when checkpointed and no sidecars remain; immutable
reads cannot safely include active WAL data. Existing journal/WAL/SHM files also
cause refusal, and changed database write metadata discards the report. No
checkpoint, journal recovery, mode conversion, chmod or cleanup is attempted.
Filesystem access-time accounting from reads is outside this guarantee.
SQLite errors exit with status 1 and a bounded diagnostic containing only the
exception class, never its message, database contents or a traceback.

Phase 1 has **no** agents, queues, subprocess/network calls, repository edits,
repairs, merges, deployment, scheduling or service changes. Reporting is manual;
no primary CLI rewrite or automated consumer is included.
