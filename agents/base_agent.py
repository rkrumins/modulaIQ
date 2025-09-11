"""Base agent class with unified API."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import Config
from core.llm_factory import LLMFactory
from core.memory import MemoryManager


logger = logging.getLogger(__name__)


class AgentRequest(BaseModel):
    """Request model for agent execution."""
    session_id: str
    input_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    """Response model for agent execution."""
    session_id: str
    response: str
    metadata: Dict[str, Any] = {}
    success: bool = True
    error: Optional[str] = None


class AgentInfo(BaseModel):
    """Information about the agent."""
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    version: str = "1.0.0"


class BaseAgent(ABC):
    """Base class for all agents with unified API."""
    
    def __init__(self, config: Config, agent_id: str, name: str, description: str, capabilities: List[str]):
        self.config = config
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.version = "1.0.0"
        
        # Initialize LLM
        self.llm = LLMFactory.create_llm(config, config.supervisor.llm_provider)
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(config)
        
        # Create FastAPI app
        self.app = FastAPI(
            title=f"{name} Agent",
            description=description,
            version=self.version
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "name": self.name,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/info", response_model=AgentInfo)
        async def get_info():
            """Get agent information."""
            return AgentInfo(
                agent_id=self.agent_id,
                name=self.name,
                description=self.description,
                capabilities=self.capabilities,
                version=self.version
            )
        
        @self.app.post("/execute", response_model=AgentResponse)
        async def execute(request: AgentRequest):
            """Execute agent task."""
            try:
                response = await self.process_request(request)
                return response
            except Exception as e:
                logger.error(f"Error in agent {self.agent_id}: {e}")
                return AgentResponse(
                    session_id=request.session_id,
                    response=f"Error processing request: {str(e)}",
                    success=False,
                    error=str(e)
                )
        
        @self.app.get("/sessions/{session_id}/history")
        async def get_session_history(session_id: str):
            """Get conversation history for a session."""
            messages = self.memory_manager.get_messages(session_id)
            return {"session_id": session_id, "messages": messages}
        
        @self.app.post("/sessions/{session_id}/clear")
        async def clear_session(session_id: str):
            """Clear conversation history for a session."""
            self.memory_manager.clear_session(session_id)
            return {"message": f"Session {session_id} cleared"}
    
    @abstractmethod
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process a request and return a response. Must be implemented by subclasses."""
        pass
    
    def get_conversation_context(self, session_id: str, limit: int = 5) -> List[Dict[str, str]]:
        """Get conversation context for the session."""
        return self.memory_manager.get_conversation_context(session_id, limit)
    
    def store_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Store a message in memory."""
        self.memory_manager.store_message(session_id, role, content, metadata)
    
    async def run(self, host: str = "localhost", port: int = 8000):
        """Run the agent server."""
        import uvicorn
        from uvicorn.config import Config as UvicornConfig
        
        logger.info(f"Starting {self.name} agent on {host}:{port}")
        
        # Create uvicorn config
        config = UvicornConfig(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        
        # Create and run server
        server = uvicorn.Server(config)
        await server.serve()
    
    def get_endpoint(self, host: str = "localhost", port: int = 8000) -> str:
        """Get the agent's endpoint URL."""
        return f"http://{host}:{port}"
