"""
Synthesis node — converts raw API data into a user-friendly answer.

Uses an LLM with a strict anti-hallucination prompt to produce a
natural-language response based *only* on the data in ``api_response``.
"""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from utils.llm_provider import ainvoke_with_fallback
from utils.logger import get_logger

logger = get_logger(__name__)


def _format_api_response(api_response: Any) -> str:
    """Produce a readable string from the raw API response."""
    if isinstance(api_response, str):
        return api_response
    try:
        return json.dumps(api_response, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(api_response)


async def synthesize_answer(state: AgentState) -> dict[str, Any]:
    """Synthesise a final natural-language answer from the API response.

    Returns a partial state update with the ``final_answer`` string.
    """
    # ── Pull config from state ───────────────────────────────────────────
    config: dict[str, Any] = state["synthesis_agent_config"]
    system_prompt: str = config["system_prompt"]
    user_prompt_tpl: str = config["user_prompt"]

    # Prefer the refined query produced by the refiner node;
    # fall back to the raw user_query if it was not set.
    user_query: str = state.get("refined_query") or state["user_query"]
    api_response_raw = state.get("api_response", "No data available.")
    api_response_str = _format_api_response(api_response_raw)

    user_prompt = user_prompt_tpl.format(
        user_query=user_query,
        api_response=api_response_str,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    logger.info("Synthesis node producing answer for: %s", user_query)

    try:
        response = await ainvoke_with_fallback(messages, agent_name="synthesis")
        answer = response.content
    except Exception as exc:
        logger.error("Synthesis LLM call failed: %s", exc)
        answer = (
            "I'm sorry, I was unable to generate a response at this time. "
            "Please try again later."
        )

    logger.info("Synthesis complete (length=%d chars).", len(answer))
    logger.info("Synthesis answer: %s", answer)
    return {"final_answer": answer}