"""
Guardrail node — classifies user queries for relevancy and safety.

Runs FIRST in the graph. Uses structured output to produce a
``GuardrailResponse`` (status / category / rationale).
If out-of-scope or a violation, populates ``final_answer`` with a
polite refusal and sets ``is_relevant = False`` so the graph short-circuits.
"""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from utils.llm_provider import ainvoke_with_fallback
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Refusal templates ─────────────────────────────────────────────────────────

_REFUSAL_OUT_OF_SCOPE = (
    "I'm sorry, but I can only help with questions about countries — "
    "such as capitals, currencies, populations, languages, and regions. "
    "Your question appears to be outside my area of expertise."
)

_REFUSAL_VIOLATION = (
    "I'm unable to process that request. "
    "If you have a genuine question about countries, I'd be happy to help!"
)


async def run_guardrail(state: AgentState) -> dict[str, Any]:
    """Evaluate the user query for relevancy and safety.

    Returns a partial state update with ``is_relevant`` and, when the
    query is rejected, a pre-filled ``final_answer``.
    """
    # ── Pull config from state ───────────────────────────────────────────
    config: dict[str, Any] = state["guardrail_agent_config"]
    system_prompt: str = config["system_prompt"]
    user_prompt_tpl: str = config["user_prompt"]
    schema: dict[str, Any] = config["output_schema"]

    user_query: str = state["user_query"]
    user_prompt = user_prompt_tpl.format(user_query=user_query)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    logger.info("Guardrail evaluating query: %s", user_query)

    try:
        result: dict[str, Any] = await ainvoke_with_fallback(
            messages, schema=schema, agent_name="guardrail",
        )
        logger.info("Guardrail result: %s", json.dumps(result))
    except Exception as exc:
        logger.error("Guardrail LLM call failed: %s", exc)
        # Fail-open: let the query through so the user isn't silently blocked
        return {
            "is_relevant": True,
            "guardrail_classification": {"status": "SAFE_IN_SCOPE", "rationale": "Fail-open due to error"},
            "guardrail_rationale": "Guardrail evaluation failed, failing open."
        }

    status = result.get("status", "OUT_OF_SCOPE")
    rationale = result.get("rationale", "")

    if status == "SAFE_IN_SCOPE":
        logger.info("Guardrail PASSED — query is relevant.")
        return {
            "is_relevant": True,
            "guardrail_classification": result,
            "guardrail_rationale": rationale
        }

    # Out of scope or violation → short-circuit with a polite refusal
    refusal = (
        _REFUSAL_VIOLATION if status == "VIOLATION" else _REFUSAL_OUT_OF_SCOPE
    )
    logger.info("Query rejected (%s): %s", status, rationale)
    return {
        "is_relevant": False,
        "final_answer": refusal,
        "guardrail_classification": result,
        "guardrail_rationale": rationale
    }
