"""Research and information gathering agent."""

import asyncio
import logging
from typing import Dict, List, Any

from agents.langgraph_agent import SimpleAgent
from core.config import Config


logger = logging.getLogger(__name__)


class ResearchAgent(SimpleAgent):
    """Specialized agent for research and information gathering."""
    
    def __init__(self, config: Config):
        system_prompt = """You are a specialized research assistant AI. Your expertise includes:

- Information gathering and synthesis
- Research methodology and best practices
- Data analysis and interpretation
- Fact-checking and verification
- Academic and professional research
- Market research and competitive analysis
- Technical research and documentation review

When conducting research:
1. Always cite sources when available
2. Distinguish between facts and opinions
3. Provide multiple perspectives when relevant
4. Highlight any limitations or uncertainties
5. Suggest additional research directions
6. Present findings in a clear, structured manner

You should be thorough, objective, and methodical in your approach to research tasks. Always acknowledge when information is incomplete or when additional research might be needed."""
        
        super().__init__(
            config=config,
            agent_id="research",
            name="Research Assistant",
            description="A specialized AI assistant for research, information gathering, and analysis",
            capabilities=[
                "information_gathering",
                "research_synthesis",
                "data_analysis",
                "fact_checking",
                "academic_research",
                "market_research",
                "documentation_review"
            ],
            system_prompt=system_prompt
        )


async def main():
    """Run the research agent."""
    config = Config.load_from_yaml()
    agent = ResearchAgent(config)
    
    # Register with supervisor
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            registration_data = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "endpoint": agent.get_endpoint(port=8004)
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
    await agent.run(port=8004)


if __name__ == "__main__":
    asyncio.run(main())
