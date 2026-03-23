"""
Intent / tool-binding node — selects the correct country tool and arguments.

Uses ``bind_tools`` + ``ainvoke`` (same pattern as ``async_tool_call_test.py``)
to let the LLM decide which REST Countries tool to invoke and with what params.
Populates ``tool_calls`` in the agent state.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes.tool_node import country_tools
from agent.state import AgentState
from utils.llm_provider import ainvoke_with_fallback
from utils.logger import get_logger

logger = get_logger(__name__)


async def extract_intent(state: AgentState) -> dict[str, Any]:
    """Bind country tools to the LLM and extract tool-call intent.

    Returns a partial state update with the ``tool_calls`` list.
    Each entry is a dict with ``name``, ``args``, and ``id`` keys
    (matching LangChain's ``tool_calls`` structure).
    """
    # ── Pull config from state ───────────────────────────────────────────
    config: dict[str, Any] = state["intent_agent_config"]
    system_prompt: str = config["system_prompt"]
    user_prompt_tpl: str = config["user_prompt"]

    user_query: str = state["user_query"]
    user_prompt = user_prompt_tpl.format(user_query=user_query)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    logger.info("Intent node processing query: %s", user_query)

    try:
        response = await ainvoke_with_fallback(messages, tools=country_tools, agent_name="intent")
    except Exception as exc:
        logger.error("Intent LLM call failed: %s", exc)
        return {
            "tool_calls": [],
            "final_answer": (
                "I'm sorry, I encountered an issue while processing your request. "
                "Please try again."
            ),
        }

    if response.tool_calls:
        for tc in response.tool_calls:
            logger.info(
                "Intent tool call: %s(%s)",
                tc["name"], tc.get("args", {}),
            )
        logger.info(
            "Intent resolved %d tool call(s): %s",
            len(response.tool_calls),
            [tc["name"] for tc in response.tool_calls],
        )
        return {"tool_calls": response.tool_calls}

    # LLM responded with text instead of a tool call — fallback
    logger.warning("LLM did not produce a tool call; responded with text.")
    return {
        "tool_calls": [],
        "final_answer": response.content or (
            "I wasn't able to determine which data to look up. "
            "Could you rephrase your question?"
        ),
    }
