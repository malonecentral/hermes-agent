from agent.memory_manager import inject_memory_provider_tools


def test_readonly_memory_toolset_exposes_only_provider_lookup_schemas():
    provider = type("Provider", (), {"name": "fixture"})()
    schemas = [
        {"name": "supermemory_search", "parameters": {}},
        {"name": "supermemory_profile", "parameters": {}},
        {"name": "supermemory_store", "parameters": {}},
        {"name": "supermemory_forget", "parameters": {}},
    ]
    manager = type("Manager", (), {
        "providers": [provider], "get_all_tool_schemas": lambda self: schemas,
    })()
    agent = type("Agent", (), {})()
    agent._memory_manager = manager
    agent.tools = []
    agent.valid_tool_names = set()
    agent.enabled_toolsets = ["memory_readonly"]
    agent.disabled_toolsets = []

    assert inject_memory_provider_tools(agent) == 2
    assert [tool["function"]["name"] for tool in agent.tools] == [
        "supermemory_search", "supermemory_profile",
    ]