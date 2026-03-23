"""
LangGraph state-graph definition for the Country Info AI Agent.

Flow:
    User Query ──► Guardrail
                    ├─ (irrelevant / violation) ──► END  (final_answer pre-filled)
                    └─ (relevant) ──► Refiner ──► Intent ──► Tools ──► Synthesis ──► END

Exported symbol: ``app`` (the compiled graph).
"""

import asyncio
import json
from typing import Any

from langgraph.graph import END, StateGraph

from agent.nodes.guardrail_node import run_guardrail
from agent.nodes.query_refiner_node import refine_query
from agent.nodes.intent_node import extract_intent
from agent.nodes.synthesis_node import synthesize_answer
from agent.nodes.tool_node import country_tools
from agent.state import AgentState
from utils.config_manager import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

_TOOL_MAP: dict[str, Any] = {t.name: t for t in country_tools}

async def _run_tools(state: AgentState) -> dict[str, Any]:
    """Execute the tool(s) selected by the intent node.

    Reads ``tool_calls`` from the state, looks each one up in
    ``_TOOL_MAP``, awaits the async tool function, and writes the
    combined result into ``api_response``.
    """
    tool_calls: list[dict[str, Any]] = state.get("tool_calls", [])

    if not tool_calls:
        logger.warning("Tool executor invoked with empty tool_calls.")
        return {"api_response": "Error: No tool was selected for execution."}

    # Restrict concurrent tool executions to avoid overwhelming APIs
    semaphore = asyncio.Semaphore(4)

    async def _execute_single_tool(tc: dict[str, Any]) -> Any:
        name = tc.get("name", "")
        args = tc.get("args", {})
        tool_fn = _TOOL_MAP.get(name)

        if tool_fn is None:
            logger.error("Unknown tool requested: %s", name)
            return f"Error: Unknown tool '{name}'."

        async with semaphore:
            logger.info("Executing tool '%s' with args %s", name, json.dumps(args))
            try:
                return await tool_fn.ainvoke(input=args)
            except Exception as exc:
                logger.error("Tool '%s' raised an error: %s", name, exc)
                return f"Error executing tool '{name}': {exc}"

    # Execute all tools concurrently and preserve order
    results = list(await asyncio.gather(*[_execute_single_tool(tc) for tc in tool_calls]))

    # If there's only a single result, unwrap it for cleaner downstream use
    api_response = results[0] if len(results) == 1 else results
    return {"api_response": api_response}

def _route_after_guardrail(state: AgentState) -> str:
    """Return the next node name based on the guardrail's verdict."""
    if state.get("is_relevant", False):
        return "refiner"
    return END

def _route_after_intent(state: AgentState) -> str:
    """Skip tool execution if intent node already set a final_answer (fallback)."""
    if state.get("final_answer"):
        return END
    if not state.get("tool_calls"):
        return END
    return "tools"

_builder = StateGraph(AgentState)

_builder.add_node("guardrail", run_guardrail)
_builder.add_node("refiner", refine_query)
_builder.add_node("intent", extract_intent)
_builder.add_node("tools", _run_tools)
_builder.add_node("synthesis", synthesize_answer)

_builder.set_entry_point("guardrail")

_builder.add_conditional_edges("guardrail", _route_after_guardrail)
_builder.add_edge("refiner", "intent")
_builder.add_conditional_edges("intent", _route_after_intent)

_builder.add_edge("tools", "synthesis")
_builder.add_edge("synthesis", END)

app = _builder.compile()

def build_initial_state(user_query: str) -> AgentState:
    return AgentState(
        user_query=user_query,
        is_relevant=False,
        refined_query="",
        tool_calls=[],
        api_response="",
        final_answer="",
        guardrail_agent_config=get_config("guardrail_prompt.yaml"),
        refiner_agent_config=get_config("query_refiner_prompt.yaml"),
        intent_agent_config=get_config("intent_prompt.yaml"),
        synthesis_agent_config=get_config("synthesis_prompt.yaml"),
    )