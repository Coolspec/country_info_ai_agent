"""
LLM provider factory with automatic fallback.

Reads ``fallback_order`` from ``inference_config.yaml`` and tries each
provider in sequence. Each provider gets **1 retry** before the next
provider is attempted.

Usage in nodes::

    from utils.llm_provider import get_llm_client

    llm = get_llm_client()                         # plain chat
    llm = get_llm_client(tools=country_tools)      # with tool binding
    llm = get_llm_client(schema=my_schema)         # with structured output
"""

from typing import Any

from langchain_cerebras import ChatCerebras
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from utils.config_manager import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Provider name → LangChain client class ───────────────────────────────────

_CLIENT_MAP: dict[str, type] = {
    "groq": ChatGroq,
    "cerebras": ChatCerebras,
    "google": ChatGoogleGenerativeAI,
}

_MAX_RETRIES_PER_PROVIDER = 1


def _build_client(provider: str, agent_name: str | None = None) -> Any:
    """Instantiate a LangChain chat client for *provider*.

    If *agent_name* is given, agent-specific ``model`` / ``temperature``
    overrides are looked up under ``provider.<agent_name>`` in the config.
    Falls back to the provider's global defaults when an override is absent.
    """
    cfg = get_config("inference_config", provider)
    if cfg is None:
        raise ValueError(f"No config found for provider '{provider}'")

    client_cls = _CLIENT_MAP.get(provider)
    if client_cls is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {', '.join(_CLIENT_MAP)}"
        )

    # Resolve agent-specific overrides, falling back to provider globals
    agent_cfg = cfg.get(agent_name, {}) if agent_name else {}
    model = agent_cfg.get("model", cfg["model"])
    temperature = agent_cfg.get("temperature", cfg.get("temperature", 0))

    # logger.info("Building client for provider '%s' with model '%s' and temperature %.1f", provider, model, temperature)

    return client_cls(
        model=model,
        temperature=temperature,
    )


def get_llm_clients(
    *,
    tools: list | None = None,
    schema: dict | None = None,
    schema_method: str = "json_schema",
    strict: bool | None = None,
    agent_name: str | None = None,
) -> list[Any]:
    """Return a list of ready-to-invoke LLM clients in fallback order.

    If *tools* is provided, each client is wrapped via ``bind_tools``.
    If *schema* is provided, each client is wrapped via ``with_structured_output``.
    """
    global_order: list[str] = get_config("inference_config", "fallback_order")
    if not global_order:
        raise ValueError("inference_config.yaml must define 'fallback_order'")

    agent_fallback_order = get_config("inference_config", "agent_fallback_order") or {}
    order = agent_fallback_order.get(agent_name, global_order) if agent_name else global_order

    clients: list[Any] = []
    for provider in order:
        try:
            client = _build_client(provider, agent_name=agent_name)
            if tools:
                client = client.bind_tools(tools)
            elif schema:
                cfg = get_config("inference_config", provider) or {}
                agent_cfg = cfg.get(agent_name, {}) if agent_name else {}
                cfg_strict = agent_cfg.get("strict", cfg.get("strict", True))
                
                final_strict = cfg_strict if strict is None else strict

                client = client.with_structured_output(
                    schema, method=schema_method, strict=final_strict,
                )
            clients.append((provider, client))
        except Exception as exc:
            logger.warning("Skipping provider '%s' (init failed): %s", provider, exc)

    if not clients:
        raise RuntimeError("All LLM providers failed to initialise.")

    return clients


async def ainvoke_with_fallback(
    messages: list,
    *,
    tools: list | None = None,
    schema: dict | None = None,
    schema_method: str = "json_schema",
    strict: bool | None = None,
    agent_name: str | None = None,
) -> Any:
    """Invoke the LLM with automatic fallback across providers.

    Tries each provider in ``fallback_order``. Each provider gets up to
    ``_MAX_RETRIES_PER_PROVIDER`` retries before moving to the next.

    Returns the raw LLM response (AIMessage or structured dict).

    Raises:
        RuntimeError: If every provider exhausts its retries.
    """
    clients = get_llm_clients(
        tools=tools, schema=schema,
        schema_method=schema_method, strict=strict,
        agent_name=agent_name,
    )

    last_exc: Exception | None = None

    for provider, client in clients:
        for attempt in range(_MAX_RETRIES_PER_PROVIDER + 1):
            try:
                logger.info(
                    "Calling provider '%s' (attempt %d/%d)",
                    provider, attempt + 1, _MAX_RETRIES_PER_PROVIDER + 1,
                )
                response = await client.ainvoke(messages)
                logger.info("Provider '%s' succeeded.", provider)
                return response
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Provider '%s' attempt %d failed: %s",
                    provider, attempt + 1, exc,
                )

        logger.error("Provider '%s' exhausted all retries.", provider)

    raise RuntimeError(
        f"All LLM providers failed. Last error: {last_exc}"
    )
