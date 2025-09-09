"""Agent registry for managing registered agents."""

import httpx
import asyncio
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

from supervisor.models import AgentInfo
from core.config import Config


logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for managing agent discovery and health checks."""
    
    def __init__(self, config: Config):
        self.config = config
        self.agents: Dict[str, AgentInfo] = {}
        self.health_check_interval = config.agent_registry.health_check_interval
        self.max_retries = config.agent_registry.max_retries
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def register_agent(self, agent_info: AgentInfo) -> bool:
        """Register a new agent."""
        try:
            # Verify agent is reachable
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{agent_info.endpoint}/health", timeout=5.0)
                if response.status_code == 200:
                    self.agents[agent_info.agent_id] = agent_info
                    logger.info(f"Successfully registered agent: {agent_info.agent_id}")
                    return True
                else:
                    logger.error(f"Agent health check failed: {agent_info.agent_id}")
                    return False
        except Exception as e:
            logger.error(f"Failed to register agent {agent_info.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
            return True
        return False
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent information by ID."""
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[AgentInfo]:
        """Get all registered agents."""
        return list(self.agents.values())
    
    def get_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """Get agents that have a specific capability."""
        return [
            agent for agent in self.agents.values()
            if capability in agent.capabilities and agent.status == "active"
        ]
    
    async def call_agent(self, agent_id: str, input_data: Dict, session_id: str = "supervisor-session") -> Dict:
        """Call an agent with input data."""
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        
        if agent.status != "active":
            raise ValueError(f"Agent is not active: {agent_id}")
        
        try:
            # Create proper agent request structure
            agent_request = {
                "session_id": session_id,
                "input_data": input_data,
                "context": {}
            }
            
            logger.info(f"Calling agent {agent_id} at {agent.endpoint}/execute")
            start_time = time.time()
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{agent.endpoint}/execute",
                    json=agent_request,
                    timeout=30.0
                )
                response.raise_for_status()
                call_duration = time.time() - start_time
                logger.info(f"Agent {agent_id} call completed in {call_duration:.2f}s")
                return response.json()
        except httpx.TimeoutException:
            logger.error(f"Timeout calling agent: {agent_id}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling agent {agent_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calling agent {agent_id}: {e}")
            raise
    
    async def health_check_agent(self, agent_id: str) -> bool:
        """Perform health check on a specific agent."""
        agent = self.get_agent(agent_id)
        if not agent:
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{agent.endpoint}/health", timeout=5.0)
                if response.status_code == 200:
                    agent.status = "active"
                    agent.last_health_check = datetime.now()
                    return True
                else:
                    agent.status = "error"
                    return False
        except Exception:
            agent.status = "error"
            return False
    
    async def health_check_all_agents(self):
        """Perform health check on all registered agents."""
        tasks = []
        for agent_id in self.agents.keys():
            tasks.append(self.health_check_agent(agent_id))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def start_health_check_loop(self):
        """Start the periodic health check loop."""
        if self._health_check_task and not self._health_check_task.done():
            return
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop_health_check_loop(self):
        """Stop the health check loop."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
    
    async def _health_check_loop(self):
        """Internal health check loop."""
        while True:
            try:
                await self.health_check_all_agents()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
