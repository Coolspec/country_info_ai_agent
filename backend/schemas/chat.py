from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's query about a country or countries.")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The final synthesized answer from the agent.")
    is_relevant: bool = Field(..., description="Whether the query was considered relevant by the guardrail.")
    guardrail_rationale: Optional[str] = Field(default=None, description="The rationale behind the guardrail classification.")
    tools_used: List[Dict[str, Any]] = Field(default_factory=list, description="A list of REST Countries tools executed by the agent to arrive at the answer.")
