#!/usr/bin/env python3
"""Script to run all example agents."""

import asyncio
import logging
from pathlib import Path
import signal
import sys
from typing import List

from agents.examples.general_agent import GeneralAgent
from agents.examples.code_agent import CodeAgent
from agents.examples.research_agent import ResearchAgent
from agents.examples.creative_agent import CreativeAgent
from core.config import Config


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/agents.log', mode='a')
        ]
    )


class AgentManager:
    """Manager for running multiple agents."""
    
    def __init__(self):
        self.agents = []
        self.tasks = []
        self.running = True
    
    def add_agent(self, agent, port: int):
        """Add an agent to run."""
        self.agents.append((agent, port))
    
    async def register_agent_with_supervisor(self, agent, port: int):
        """Register an agent with the supervisor."""
        import httpx
        logger = logging.getLogger(__name__)
        
        try:
            async with httpx.AsyncClient() as client:
                registration_data = {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "description": agent.description,
                    "capabilities": agent.capabilities,
                    "endpoint": agent.get_endpoint(port=port)
                }
                
                response = await client.post(
                    f"http://localhost:8000/agents/register",
                    json=registration_data,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Successfully registered {agent.name} with supervisor")
                else:
                    logger.error(f"❌ Failed to register {agent.name}: {response.text}")
        except Exception as e:
            logger.error(f"❌ Error registering {agent.name} with supervisor: {e}")
    
    async def start_agents(self):
        """Start all agents."""
        logger = logging.getLogger(__name__)
        
        # Start agents first
        for agent, port in self.agents:
            task = asyncio.create_task(agent.run(port=port))
            self.tasks.append(task)
            logger.info(f"Started {agent.name} on port {port}")
        
        # Wait a moment for agents to start
        await asyncio.sleep(2)
        
        # Register agents with supervisor
        logger.info("Registering agents with supervisor...")
        registration_tasks = []
        for agent, port in self.agents:
            registration_task = asyncio.create_task(
                self.register_agent_with_supervisor(agent, port)
            )
            registration_tasks.append(registration_task)
        
        # Wait for all registrations to complete
        await asyncio.gather(*registration_tasks, return_exceptions=True)
        
        # Wait for all agent tasks
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("Agents stopped")
    
    def stop_agents(self):
        """Stop all agents."""
        self.running = False
        for task in self.tasks:
            task.cancel()


async def main():
    """Main function to run all agents."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config.load_from_yaml()
        
        # Create agent manager
        manager = AgentManager()
        
        # Add agents
        manager.add_agent(GeneralAgent(config), 8002)
        manager.add_agent(CodeAgent(config), 8003)
        manager.add_agent(ResearchAgent(config), 8004)
        manager.add_agent(CreativeAgent(config), 8005)
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            manager.stop_agents()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("Starting all agents...")
        await manager.start_agents()
        
    except Exception as e:
        logger.error(f"Failed to start agents: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
