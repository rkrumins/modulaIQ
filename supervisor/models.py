"""Pydantic models for the supervisor."""

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class AgentInfo(BaseModel):
    """Information about a registered agent."""
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    endpoint: str
    status: Literal["active", "inactive", "error"] = "active"
    registered_at: datetime = Field(default_factory=datetime.now)
    last_health_check: Optional[datetime] = None


class TaskRequest(BaseModel):
    """Request for task execution."""
    session_id: str
    query: str
    context: Optional[Dict[str, Any]] = None
    max_iterations: Optional[int] = None


class TaskResponse(BaseModel):
    """Response from task execution."""
    session_id: str
    response: str
    reasoning: List[str] = Field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    agents_used: List[str] = Field(default_factory=list)
    iterations: int = 0
    success: bool = True
    error: Optional[str] = None


class AgentCall(BaseModel):
    """Individual agent call within a task."""
    agent_id: str
    input_data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class ReasoningStep(BaseModel):
    """A single reasoning step in the ReAct paradigm."""
    step: int
    reasoning: str
    action: str
    observation: str
    timestamp: datetime = Field(default_factory=datetime.now)


class SupervisorState(BaseModel):
    """State of the supervisor during task execution."""
    session_id: str
    query: str
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    agent_calls: List[AgentCall] = Field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 10
    context: Dict[str, Any] = Field(default_factory=dict)
    final_response: Optional[str] = None
    is_complete: bool = False
    error: Optional[str] = None
