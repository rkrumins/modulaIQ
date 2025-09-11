"""General purpose agent for handling various queries."""

import asyncio
import logging
from typing import Dict, List, Any

from agents.langgraph_agent import SimpleAgent
from core.config import Config


logger = logging.getLogger(__name__)


class GeneralAgent(SimpleAgent):
    """General purpose agent that can handle a wide variety of queries."""
    
    def __init__(self, config: Config):
        system_prompt = """You are a helpful general-purpose AI assistant. You can help with a wide variety of tasks including:

- Answering questions on various topics
- Providing explanations and clarifications
- Helping with problem-solving
- Offering suggestions and recommendations
- Assisting with analysis and reasoning

You should be helpful, accurate, and provide clear, well-structured responses. If you're unsure about something, say so rather than making up information.

Always maintain a professional and friendly tone in your responses."""
        
        super().__init__(
            config=config,
            agent_id="general",
            name="General Assistant",
            description="A general-purpose AI assistant that can help with various tasks and questions",
            capabilities=[
                "question_answering",
                "explanation",
                "problem_solving",
                "analysis",
                "recommendations",
                "general_conversation"
            ],
            system_prompt=system_prompt
        )


async def main():
    """Run the general agent."""
    config = Config.load_from_yaml()
    agent = GeneralAgent(config)
    
    # Register with supervisor
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            registration_data = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "endpoint": agent.get_endpoint(port=8002)
            }
            
            response = await client.post(
                f"http://localhost:8000/agents/register",
                json=registration_data
            )
            
            if response.status_code == 200:
                logger.info("Successfully registered with supervisor")
            else:
                logger.error(f"Failed to register with supervisor: {response.text}")
    except Exception as e:
        logger.error(f"Error registering with supervisor: {e}")
    
    # Run the agent
    await agent.run(port=8002)


if __name__ == "__main__":
    asyncio.run(main())
