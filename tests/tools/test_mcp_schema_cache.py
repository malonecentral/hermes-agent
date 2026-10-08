"""Unit tests for the on-disk MCP schema cache (tools/mcp_schema_cache.py).

The module landed in #56832's extraction without its tests; these cover the
fingerprint keying, read/write round-trip, and invalidation behavior.
"""

import tools.mcp_schema_cache as msc


class TestConfigFingerprint:
    def test_stable_for_same_config(self):
        cfg = {"command": "npx", "args": ["-y", "@playwright/mcp"]}
        assert msc.config_fingerprint(cfg) == msc.config_fingerprint(dict(cfg))

    def test_changes_when_connection_config_changes(self):
        base = {"command": "npx", "args": ["-y", "@playwright/mcp"]}
        assert msc.config_fingerprint(base) != msc.config_fingerprint(
            {**base, "args": ["-y", "@playwright/mcp", "--headless"]}
        )
        assert msc.config_fingerprint(base) != msc.config_fingerprint(
            {**base, "command": "uvx"}
        )
        assert msc.config_fingerprint(base) != msc.config_fingerprint(
            {**base, "tools": {"include": ["a"]}}
        )

    def test_ignores_non_connection_keys(self):
        base = {"command": "npx", "args": []}
        assert msc.config_fingerprint(base) == msc.config_fingerprint(
            {**base, "timeout": 5, "enabled": True, "lazy": True}
        )

    def test_changes_when_schema_or_auth_relevant_config_changes(self):
        base = {
            "url": "https://example.test/mcp",
            "headers": {"Authorization": "Bearer first"},
            "env": {"SERVER_TOOL_VERSION": "1"},
        }
        assert msc.config_fingerprint(base) != msc.config_fingerprint(
            {**base, "headers": {"Authorization": "Bearer second"}}
        )
        assert msc.config_fingerprint(base) != msc.config_fingerprint(
            {**base, "env": {"SERVER_TOOL_VERSION": "2"}}
        )

    def test_cache_format_version_invalidates_old_fingerprints(self, monkeypatch):
        cfg = {"url": "https://example.test/mcp"}
        before = msc.config_fingerprint(cfg)
        monkeypatch.setattr(msc, "_CACHE_FORMAT_VERSION", msc._CACHE_FORMAT_VERSION + 1)
        assert before != msc.config_fingerprint(cfg)


class TestCacheRoundTrip:
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(msc, "_cache_path", lambda: tmp_path / "cache.json")

    def test_write_then_read_with_matching_fingerprint(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        tools = [{"name": "t1", "description": "d", "inputSchema": {"type": "object"}}]
        msc.write_cache_entry("srv", "fp1", tools=tools, utility_tools=[])
        entry = msc.get_cached_entry("srv", "fp1")
        assert entry is not None
        assert msc.tools_from_cache_entry(entry) == tools
        assert msc.utility_tools_from_cache_entry(entry) == []
        assert msc.has_cached_entry("srv", "fp1")

    def test_fingerprint_mismatch_returns_none(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        msc.write_cache_entry("srv", "fp1", tools=[], utility_tools=[])
        assert msc.get_cached_entry("srv", "OTHER") is None
        assert not msc.has_cached_entry("srv", "OTHER")

    def test_entry_without_server_ttl_expires_at_safe_default(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        monkeypatch.setattr(msc.time, "time", lambda: 1_000.0)
        msc.write_cache_entry("srv", "fp1", tools=[], utility_tools=[])
        monkeypatch.setattr(
            msc.time,
            "time",
            lambda: 1_000.0 + msc._DEFAULT_CACHE_MAX_AGE_SECONDS + 1,
        )
        assert msc.get_cached_entry("srv", "fp1") is None

    def test_missing_server_returns_none(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        assert msc.get_cached_entry("nope", "fp") is None

    def test_clear_cache_entry(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        msc.write_cache_entry("srv", "fp1", tools=[], utility_tools=[])
        msc.clear_cache_entry("srv")
        assert msc.get_cached_entry("srv", "fp1") is None

    def test_corrupt_cache_file_is_tolerated(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        (tmp_path / "cache.json").write_text("{not json", encoding="utf-8")
        assert msc.get_cached_entry("srv", "fp") is None
        # And writes recover the file.
        msc.write_cache_entry("srv", "fp", tools=[], utility_tools=[])
        assert msc.has_cached_entry("srv", "fp")

    def test_malformed_entry_shapes_are_tolerated(self):
        assert msc.tools_from_cache_entry({"tools": "nope"}) == []
        assert msc.utility_tools_from_cache_entry({}) == []


class TestCacheFileLocation:
    def test_cache_lives_under_hermes_home_cache_dir_with_0600(
        self, monkeypatch, tmp_path
    ):
        # Real path (no _cache_path monkeypatch): HERMES_HOME/cache/…, 0o600,
        # matching the discovery-cache precedent in tools/registry.py.
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
        path = msc._cache_path()
        assert path == tmp_path / "cache" / "mcp_schema_cache.json"
        msc.write_cache_entry("srv", "fp", tools=[], utility_tools=[])
        assert path.exists()
        assert (path.stat().st_mode & 0o777) == 0o600


class TestWriteRefresh:
    def test_identical_payload_refreshes_bounded_default_ttl(self, monkeypatch, tmp_path):
        monkeypatch.setattr(msc, "_cache_path", lambda: tmp_path / "cache.json")
        saves = []
        real_save = msc._save_all

        def _counting_save(data):
            saves.append(1)
            real_save(data)

        monkeypatch.setattr(msc, "_save_all", _counting_save)
        tools = [{"name": "t1", "description": "d", "inputSchema": {}}]
        msc.write_cache_entry("srv", "fp1", tools=tools, utility_tools=[])
        assert len(saves) == 1
        # Identical payload was live-reconfirmed, so refresh written_at.
        msc.write_cache_entry("srv", "fp1", tools=list(tools), utility_tools=[])
        assert len(saves) == 2
        # Changed payload → rewrite.
        msc.write_cache_entry("srv", "fp2", tools=tools, utility_tools=[])
        assert len(saves) == 3
