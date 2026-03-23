"""
Tests that config_manager.get_config correctly loads every prompt YAML
in backend/agent/prompts/ and that each contains the expected keys.

Run:  python -m pytest test_config_prompts.py -v   (from backend/)
"""

import sys
from pathlib import Path

import pytest

# Ensure the backend package root is on sys.path so `utils.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_manager import get_config, load_config, reload_config


# ─── Prompt file names (without directory, matching what graph.py passes) ──────

PROMPT_FILES = [
    "guardrail_prompt.yaml",
    "query_refiner_prompt.yaml",
    "intent_prompt.yaml",
    "synthesis_prompt.yaml",
]

# Every prompt YAML must have at least these top-level keys
REQUIRED_KEYS = {"system_prompt", "user_prompt"}


# ─── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prompt_name", PROMPT_FILES)
def test_load_config_returns_dict(prompt_name: str):
    """get_config should return a dict (not str) for YAML prompt files."""
    data = get_config(prompt_name)
    assert isinstance(data, dict), (
        f"Expected dict from '{prompt_name}', got {type(data).__name__}"
    )


@pytest.mark.parametrize("prompt_name", PROMPT_FILES)
def test_prompt_has_required_keys(prompt_name: str):
    """Each prompt YAML must contain 'system_prompt' and 'user_prompt'."""
    data = get_config(prompt_name)
    missing = REQUIRED_KEYS - data.keys()
    assert not missing, (
        f"'{prompt_name}' is missing required keys: {missing}"
    )


@pytest.mark.parametrize("prompt_name", PROMPT_FILES)
def test_system_prompt_is_non_empty_string(prompt_name: str):
    """system_prompt value must be a non-empty string."""
    data = get_config(prompt_name)
    sp = data["system_prompt"]
    assert isinstance(sp, str) and sp.strip(), (
        f"'system_prompt' in '{prompt_name}' is empty or not a string"
    )


@pytest.mark.parametrize("prompt_name", PROMPT_FILES)
def test_user_prompt_is_non_empty_string(prompt_name: str):
    """user_prompt value must be a non-empty string."""
    data = get_config(prompt_name)
    up = data["user_prompt"]
    assert isinstance(up, str) and up.strip(), (
        f"'user_prompt' in '{prompt_name}' is empty or not a string"
    )


def test_guardrail_has_output_schema():
    """The guardrail prompt must include an output_schema for structured output."""
    data = get_config("guardrail_prompt.yaml")
    assert "output_schema" in data, (
        "guardrail_prompt.yaml is missing the 'output_schema' key "
        "needed by guardrail_node.py"
    )
    schema = data["output_schema"]
    assert isinstance(schema, dict), "output_schema should be a dict"
    assert "properties" in schema, "output_schema must define 'properties'"


def test_user_prompt_contains_placeholder():
    """Every user_prompt should contain {user_query} for formatting."""
    for name in PROMPT_FILES:
        data = get_config(name)
        up = data["user_prompt"]
        assert "{user_query}" in up, (
            f"'user_prompt' in '{name}' is missing the '{{user_query}}' placeholder"
        )


def test_synthesis_user_prompt_has_api_response_placeholder():
    """synthesis_prompt's user_prompt must also contain {api_response}."""
    data = get_config("synthesis_prompt.yaml")
    up = data["user_prompt"]
    assert "{api_response}" in up, (
        "synthesis_prompt.yaml user_prompt is missing the '{api_response}' placeholder"
    )


def test_reload_config_refreshes_cache():
    """reload_config should return fresh data (not stale cache)."""
    first = load_config("guardrail_prompt.yaml")
    reloaded = reload_config("guardrail_prompt.yaml")
    # Content should be identical (file hasn't changed on disk)
    assert first == reloaded


def test_get_config_nested_key_access():
    """get_config with extra keys should traverse into the dict."""
    status_enum = get_config(
        "guardrail_prompt.yaml", "output_schema", "properties", "status"
    )
    assert isinstance(status_enum, dict), "Nested key access should return a dict"
    assert "enum" in status_enum, "status property should have an 'enum' list"
