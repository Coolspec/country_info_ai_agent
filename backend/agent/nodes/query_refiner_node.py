"""
Query Refiner node — rewrites the raw user query into a clearer form.

Runs AFTER the guardrail (only for relevant queries) and BEFORE intent
classification.  Stores the cleaned-up question in ``refined_query``
so the synthesizer can reference it later.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from utils.llm_provider import ainvoke_with_fallback
from utils.logger import get_logger

logger = get_logger(__name__)


async def refine_query(state: AgentState) -> dict[str, Any]:
    """Rewrite the raw user query into an explicit, search-friendly form.

    Returns a partial state update with the ``refined_query`` string.
    """
    # ── Pull config from state ───────────────────────────────────────────
    config: dict[str, Any] = state["refiner_agent_config"]
    system_prompt: str = config["system_prompt"]
    user_prompt_tpl: str = config["user_prompt"]

    user_query: str = state["user_query"]
    user_prompt = user_prompt_tpl.format(user_query=user_query)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    logger.info("Refiner node processing query: %s", user_query)

    try:
        response = await ainvoke_with_fallback(messages, agent_name="refiner")
        refined = response.content.strip()
    except Exception as exc:
        logger.error("Refiner LLM call failed: %s", exc)
        # Fallback — pass the original query through unchanged
        refined = user_query

    logger.info("Refined query: %s", refined)
    return {"refined_query": refined}
