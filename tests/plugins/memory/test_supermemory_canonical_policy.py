import hashlib

import pytest

from plugins.memory.supermemory.canonical_policy import (
    FAMILY_CONTAINER,
    OWNER_CONTAINER,
    canonical_scope_from_path,
    canonical_visibility_from_path,
    stable_custom_id_from_path,
)


@pytest.mark.parametrize(
    ("path", "scope"),
    [
        ("Root Note.md", OWNER_CONTAINER),
        ("Jarvis/Fact.md", OWNER_CONTAINER),
        ("Jarvis/Deep/Fact.md", OWNER_CONTAINER),
        ("Skills/Memory/SKILL.md", OWNER_CONTAINER),
        ("Jarvis/Family Shared/Fact.md", FAMILY_CONTAINER),
        ("Jarvis/Family Shared/People/Zoë.md", FAMILY_CONTAINER),
        ("Jarvis/Release: Notes.md", OWNER_CONTAINER),
        ("Jarvis/100% ready.md", OWNER_CONTAINER),
    ],
)
def test_approved_canonical_paths_have_exact_scope(path, scope):
    assert canonical_scope_from_path(path) == scope


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "..",
        "./Root.md",
        "../Root.md",
        "Jarvis/./Root.md",
        "Jarvis/../Root.md",
        "Jarvis//Root.md",
        "Jarvis/Root.md/",
        "Jarvis\\Root.md",
        "/Root.md",
        "//server/Root.md",
        "C:/Root.md",
        "C:Root.md",
        "Root.txt",
        "Root.MD",
        "Root.md ",
        " Root.md",
        "Jarvis",
        "Jarvis/Family Shared",
        "People/Owner.md",
        "Other/Secret.md",
        "Family Shared/Fact.md",
        "JarvisX/Fact.md",
        "SkillsX/Fact.md",
        "skills/Fact.md",
        "Ｊarvis/Fact.md",
        "Jarvis／Family Shared/Fact.md",
    ],
)
def test_noncanonical_paths_are_rejected(path):
    assert canonical_scope_from_path(path) is None
    assert canonical_visibility_from_path(path) is None
    assert stable_custom_id_from_path(path) is None


@pytest.mark.parametrize(
    "path",
    [
        "Root\x00.md",
        "Jarvis/%2e%2e/Fact.md",
        "Jarvis/%2E%2E/Fact.md",
        "Jarvis/%2FFamily Shared/Fact.md",
        "Jarvis/%2fFamily Shared/Fact.md",
        "Root%2FChild.md",
        "Root%5cChild.md",
        "Root%5CChild.md",
        "Root%41.md",  # Conservative: reject every syntactic byte escape.
        "Root%aF.md",
        "Jarvis/%252e%252e/Fact.md",
        "Jarvis/%252E%252F/Fact.md",
        "Jarvis/%255cFamily Shared/Fact.md",
        "Jarvis/%255CFamily Shared/Fact.md",
        "Root／Child.md",  # U+FF0F fullwidth solidus
        "Root＼Child.md",  # U+FF3C fullwidth reverse solidus
        "Root⁄Child.md",  # U+2044 fraction slash
        "Root∕Child.md",  # U+2215 division slash
        "Root⧵Child.md",  # U+29F5 reverse solidus operator
        "Root﹨Child.md",  # U+FE68 small reverse solidus
        "Root╱Child.md",  # U+2571 box-drawing diagonal
        "Root╲Child.md",  # U+2572 box-drawing diagonal
        "Root⧸Child.md",  # U+29F8 big solidus
        "Root⧹Child.md",  # U+29F9 big reverse solidus
        "Root∖Child.md",  # U+2216 slash-like set minus
    ],
)
def test_malformed_or_separator_lookalike_paths_derive_no_trusted_values(path):
    assert canonical_scope_from_path(path) is None
    assert canonical_visibility_from_path(path) is None
    assert stable_custom_id_from_path(path) is None


@pytest.mark.parametrize(
    "path",
    [
        "C:Root.md",
        "c:Root.md",
        "Z:/Root.md",
        "z:\\Root.md",
        "//server/share/Root.md",
        "\\\\server\\share\\Root.md",
        "//?/C:/Root.md",
        "\\\\?\\C:\\Root.md",
        "//./C:/Root.md",
        "\\\\.\\C:\\Root.md",
        "\\??\\C:\\Root.md",
    ],
)
def test_windows_qualified_relative_unc_and_device_paths_derive_no_trusted_values(path):
    assert canonical_scope_from_path(path) is None
    assert canonical_visibility_from_path(path) is None
    assert stable_custom_id_from_path(path) is None


@pytest.mark.parametrize("path", [None, 0, b"Root.md", [], object()])
def test_non_string_paths_are_rejected(path):
    assert canonical_scope_from_path(path) is None


def test_visibility_is_derived_only_from_exact_scope():
    assert canonical_visibility_from_path("Root.md") == "owner_private"
    assert canonical_visibility_from_path("Jarvis/Fact.md") == "owner_private"
    assert canonical_visibility_from_path("Skills/Fact.md") == "owner_private"
    assert canonical_visibility_from_path("Jarvis/Family Shared/Fact.md") == "family_shared"


@pytest.mark.parametrize(
    "path", ["Jarvis/Family Sharedness/Fact.md", "Jarvis/Family SharedX/Fact.md"]
)
def test_family_prefix_siblings_remain_owner_scoped(path):
    assert canonical_scope_from_path(path) == OWNER_CONTAINER
    assert canonical_visibility_from_path(path) == "owner_private"


def test_stable_custom_id_is_obsidian_prefixed_sha256_of_utf8_relative_path():
    path = "Jarvis/Family Shared/People/Zoë.md"
    expected = "obsidian-" + hashlib.sha256(path.encode("utf-8")).hexdigest()
    assert stable_custom_id_from_path(path) == expected
    assert len(expected) == len("obsidian-") + 64


def test_stable_identity_is_path_sensitive_and_not_unicode_normalized():
    composed = "Jarvis/Café.md"
    decomposed = "Jarvis/Cafe\u0301.md"
    assert canonical_scope_from_path(composed) == OWNER_CONTAINER
    assert canonical_scope_from_path(decomposed) == OWNER_CONTAINER
    assert stable_custom_id_from_path(composed) != stable_custom_id_from_path(decomposed)


@pytest.mark.parametrize(
    "path",
    [
        "Résumé.md",
        "Jarvis/日本語/事実.md",
        "Skills/Mémoire/Zoë.md",
        "Jarvis/Family Shared/Crème brûlée.md",
        "Jarvis/絵文字 🚀.md",
        "Jarvis/Math π + ∑ = ∞.md",
        "Jarvis/Punctuation – ‘quoted’ (draft)!.md",
        "Jarvis/Colon: valid; percent% literal.md",
    ],
)
def test_legitimate_unicode_names_are_valid_and_hash_exact_utf8_bytes(path):
    assert canonical_scope_from_path(path) is not None
    expected = "obsidian-" + hashlib.sha256(path.encode("utf-8")).hexdigest()
    assert stable_custom_id_from_path(path) == expected
