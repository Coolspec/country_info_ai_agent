"""
Unified configuration & file manager.

Supported file types: .yaml / .yml, .json, .txt
Searches two directories:
  - ``app/configs/``  (configs, .env)
  - ``agent/prompts/``      (prompt templates)
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Directories ───────────────────────────────────────────────────────────────

_BACKEND_DIR = Path(__file__).resolve().parent.parent          # backend/
_CONFIGS_DIR = _BACKEND_DIR / "app" / "configs"
_PROMPTS_DIR = _BACKEND_DIR / "agent" / "prompts"

_SEARCH_DIRS = [_CONFIGS_DIR, _PROMPTS_DIR]

# Supported extensions (order matters for resolution)
_SUPPORTED_EXTENSIONS = [".yaml", ".yml", ".json", ".txt"]

_env_loaded = False

# ─── Config mode & cache ──────────────────────────────────────────────────────

# Set CONFIG_MODE=dev in your .env or environment to always
# reload configs fresh from disk on every request.
# In prod mode (default), configs are cached with a TTL.

_CACHE_TTL_SECONDS = 300  # 5 minutes for prod mode

# Cache structure: { name: (timestamp, data) }
_config_cache: dict[str, tuple[float, Any]] = {}


def _get_mode() -> str:
    """Return 'dev' or 'prod' based on CONFIG_MODE env var."""
    return os.getenv("CONFIG_MODE", "prod").lower()


# ─── Environment Variable Helpers ─────────────────────────────────────────────

def load_env(env_path: str | Path | None = None, *, override: bool = False) -> None:
    """
    Load the ``.env`` file into the process environment.

    Idempotent — subsequent calls are no-ops unless *override* is ``True``.

    Args:
        env_path: Optional custom path. Defaults to ``backend/app/configs/.env``.
        override: If ``True``, reload the file even if already loaded and
                  overwrite existing env vars.
    """

    if os.getenv("CONFIG_MODE", "prod").lower() == "prod":
        logger.info("Production mode enabled. Skipping .env loading.")
        return

    global _env_loaded
    if _env_loaded and not override:
        return

    path = Path(env_path) if env_path else _CONFIGS_DIR / ".env"

    if path.is_file():
        load_dotenv(dotenv_path=path, override=override)
        logger.info("Loaded .env from %s", path)
    else:
        logger.warning(".env file not found at %s", path)

    _env_loaded = True


def get_env(variable: str, *, required: bool = True, default: str | None = None) -> str | None:
    """
    Get an environment variable (sourced from ``.env`` or the system).

    Automatically calls :func:`load_env` on first use.

    Args:
        variable: Name of the environment variable.
        required: If ``True``, raise when missing.
        default: Fallback value.
    """
    load_env()
    value = os.getenv(variable, default)
    if value is None and required:
        logger.error("Missing required env var: %s", variable)
        raise Exception(f"Required environment variable not set: '{variable}'")
    return value


# ─── Unified File Loader ──────────────────────────────────────────────────────

def load_config(name: str) -> dict[str, Any] | str:
    """
    Load a config or prompt file by name.

    In **dev** mode (``CONFIG_MODE=dev``), always reads fresh from disk.
    In **prod** mode (default), caches the result for ``_CACHE_TTL_SECONDS``.

    Resolution order: searches ``app/configs/`` then ``prompts/``,
    trying the exact name first, then appending each supported extension.

    Returns:
        - ``dict`` for YAML / JSON files.
        - ``str``  for TXT files (and prompts).

    Raises:
        Exception: If the file is not found or cannot be parsed.
    """
    mode = _get_mode()

    # In prod mode, return cached data if still fresh
    if mode == "prod" and name in _config_cache:
        cached_time, cached_data = _config_cache[name]
        if (time.time() - cached_time) < _CACHE_TTL_SECONDS:
            return cached_data

    path = _resolve_path(name)
    suffix = path.suffix.lower()
    logger.debug("Loading file: %s (mode=%s)", path, mode)

    try:
        raw = path.read_text(encoding="utf-8")

        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(raw)
            result = data if data is not None else {}

        elif suffix == ".json":
            result = json.loads(raw)

        elif suffix == ".txt":
            result = raw

        else:
            raise Exception(
                f"Unsupported file type '{suffix}' for '{name}'. "
                f"Supported: {', '.join(_SUPPORTED_EXTENSIONS)}"
            )

    except (yaml.YAMLError, json.JSONDecodeError) as e:
        logger.error("Parse error in '%s': %s", name, e)
        raise Exception(f"Failed to parse '{name}': {e}") from e

    logger.info("Loaded '%s' successfully. (mode=%s)", name, mode)

    # Cache in prod mode only
    if mode == "prod":
        _config_cache[name] = (time.time(), result)

    return result


def get_config(name: str, *keys: str, default: Any = None) -> Any:
    """
    Load a file and optionally traverse nested keys (for YAML / JSON).

    Examples::

        # Full config dict
        cfg = get_config("inference_config")

        # Nested value
        api_key = get_config("inference_config", "grok", "api-key")

        # Prompt text (no keys needed)
        prompt = get_config("system_prompt")

    Args:
        name: Filename (with or without extension).
        *keys: Sequence of keys to traverse into the loaded dict.
        default: Fallback value if a key is not found.

    Returns:
        The value at the key path, the full dict, or the raw text.
    """
    data = load_config(name)

    if not keys:
        return data

    if not isinstance(data, dict):
        logger.warning("Cannot traverse keys on non-dict file '%s'.", name)
        return default

    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data


def reload_config(name: str) -> dict[str, Any] | str:
    """Clear the cache entry for *name* and reload from disk."""
    _config_cache.pop(name, None)
    return load_config(name)

# ─── Internal Helpers ──────────────────────────────────────────────────────────

def _resolve_path(name: str) -> Path:
    """
    Resolve *name* to an absolute file path.

    Searches each directory in ``_SEARCH_DIRS`` for:
      1. Exact filename match
      2. Filename + each supported extension
    """
    for directory in _SEARCH_DIRS:
        # Exact match
        candidate = directory / name
        if candidate.is_file():
            return candidate
        # Try supported extensions
        for ext in _SUPPORTED_EXTENSIONS:
            candidate = directory / f"{name}{ext}"
            if candidate.is_file():
                return candidate

    raise Exception(
        f"File not found: '{name}' (searched {', '.join(str(d) for d in _SEARCH_DIRS)})"
    )