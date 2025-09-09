"""Pydantic models for the supervisor."""

from typing import Dict, List, Any, Optional, Literal, Set
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


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


class TaskType(str, Enum):
    """Types of tasks that can be executed."""
    RESEARCH = "research"
    CODE = "code"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    DIRECT_LLM = "direct_llm"


class TaskStatus(str, Enum):
    """Status of a task in the execution graph."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionNode(BaseModel):
    """A node in the execution graph representing a task."""
    node_id: str
    task_type: TaskType
    description: str
    agent_id: Optional[str] = None  # None for direct LLM tasks
    input_data: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)  # Node IDs this depends on
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ExecutionPlan(BaseModel):
    """A complete execution plan with nodes and dependencies."""
    plan_id: str
    query: str
    nodes: Dict[str, ExecutionNode] = Field(default_factory=dict)
    execution_order: List[List[str]] = Field(default_factory=list)  # Parallel execution batches
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


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
    # New fields for planning system
    execution_plan: Optional[ExecutionPlan] = None
    use_planning: bool = True
