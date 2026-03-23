from typing import Any

from schemas.chat import ChatRequest, ChatResponse
from agent.graph import app, build_initial_state
from utils.logger import get_logger

logger = get_logger(__name__)

async def process_chat(request: ChatRequest) -> ChatResponse:
    """
    Executes the LangGraph pipeline for a specific user query and returns
    a formatted ChatResponse populated with the final graph state.
    """
    logger.info("Processing chat request for query: %s", request.query)
    
    # Create the initial state dictionary dynamically
    initial_state = build_initial_state(request.query)
    
    try:
        # Run the agent tracking process asynchronously
        final_state = await app.ainvoke(initial_state)
    except Exception as exc:
        logger.error("LangGraph processing failed unexpectedly: %s", exc)
        return ChatResponse(
            answer="I am sorry, an unexpected error occurred while processing your request.",
            is_relevant=True,  # Assume true so it doesn't look like a guardrail violation
            guardrail_rationale=f"Error: {str(exc)}",
            tools_used=[]
        )

    # Extract final outputs
    answer = final_state.get("final_answer", "")
    is_relevant = final_state.get("is_relevant", False)
    guardrail_rationale = final_state.get("guardrail_rationale")
    
    # Determine the tools that were actually chosen by the LLM (if any)
    tool_calls = final_state.get("tool_calls", [])
    tools_used = [
        {"name": tc.get("name"), "args": tc.get("args")} 
        for tc in tool_calls 
        if isinstance(tc, dict) and "name" in tc and "args" in tc
    ]
    
    response = ChatResponse(
        answer=answer,
        is_relevant=is_relevant,
        guardrail_rationale=guardrail_rationale,
        tools_used=tools_used
    )
    
    logger.info("Chat processed successfully. Tools used: %s", tools_used)
    return response
