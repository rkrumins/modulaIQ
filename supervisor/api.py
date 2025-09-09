"""FastAPI server for the Supervisor agent."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supervisor.models import AgentInfo, TaskRequest, TaskResponse
from supervisor.agent_registry import AgentRegistry
from supervisor.react_supervisor import ReactSupervisor
from core.config import Config
from core.memory import MemoryManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
config: Config = None
agent_registry: AgentRegistry = None
memory_manager: MemoryManager = None
supervisor: ReactSupervisor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global config, agent_registry, memory_manager, supervisor
    
    # Startup
    logger.info("Starting Supervisor API...")
    
    # Load configuration
    config = Config.load_from_yaml()
    
    # Initialize components
    agent_registry = AgentRegistry(config)
    memory_manager = MemoryManager(config)
    supervisor = ReactSupervisor(config, agent_registry, memory_manager)
    
    # Start health check loop
    await agent_registry.start_health_check_loop()
    
    logger.info("Supervisor API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Supervisor API...")
    await agent_registry.stop_health_check_loop()
    logger.info("Supervisor API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Supervisor",
    description="ReAct-based Supervisor for orchestrating AI agents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class AgentRegistrationRequest(BaseModel):
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    endpoint: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    registered_agents: int


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=str(asyncio.get_event_loop().time()),
        registered_agents=len(agent_registry.get_all_agents())
    )


@app.post("/agents/register")
async def register_agent(request: AgentRegistrationRequest):
    """Register a new agent."""
    agent_info = AgentInfo(
        agent_id=request.agent_id,
        name=request.name,
        description=request.description,
        capabilities=request.capabilities,
        endpoint=request.endpoint
    )
    
    success = await agent_registry.register_agent(agent_info)
    if success:
        return {"message": f"Agent {request.agent_id} registered successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to register agent")


@app.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str):
    """Unregister an agent."""
    success = await agent_registry.unregister_agent(agent_id)
    if success:
        return {"message": f"Agent {agent_id} unregistered successfully"}
    else:
        raise HTTPException(status_code=404, detail="Agent not found")


@app.get("/agents", response_model=List[AgentInfo])
async def list_agents():
    """List all registered agents."""
    return agent_registry.get_all_agents()


@app.get("/agents/{agent_id}", response_model=AgentInfo)
async def get_agent(agent_id: str):
    """Get information about a specific agent."""
    agent = agent_registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@app.get("/agents/capability/{capability}", response_model=List[AgentInfo])
async def get_agents_by_capability(capability: str):
    """Get agents with a specific capability."""
    return agent_registry.get_agents_by_capability(capability)


@app.post("/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute a task using the ReAct supervisor."""
    try:
        response = await supervisor.execute_task(request)
        return response
    except Exception as e:
        logger.error(f"Error executing task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    memory_manager.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    messages = memory_manager.get_messages(session_id)
    return {"session_id": session_id, "messages": messages}


if __name__ == "__main__":
    import uvicorn
    config = Config.load_from_yaml()
    uvicorn.run(
        "supervisor.api:app",
        host=config.supervisor.host,
        port=config.supervisor.port,
        reload=True
    )
