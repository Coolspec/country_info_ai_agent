"""
Agent state definition for the LangGraph country info agent.

Carries data between nodes and includes sub-agent task prompts
so each node can retrieve its instructions from the state dict.
"""

from typing import Any, TypedDict, Union


class AgentState(TypedDict):
    """Shared state flowing through every node in the graph."""

    #User input
    user_query: str

    #Guardrail output
    is_relevant: bool
    guardrail_classification: dict[str, Any]
    guardrail_rationale: str

    #Query refiner output
    refined_query: str

    #Intent / tool-binding output
    tool_calls: list[dict[str, Any]]

    #Tool execution output
    api_response: Union[str, dict, list]

    #Final synthesised answer
    final_answer: str

    #Task prompts (injected at graph init from YAML configs)
    guardrail_agent_config: dict[str, Any]
    refiner_agent_config: dict[str, Any]
    intent_agent_config: dict[str, Any]
    synthesis_agent_config: dict[str, Any]
