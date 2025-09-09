"""Code analysis and generation agent."""

import asyncio
import logging
from typing import Dict, List, Any

from agents.langgraph_agent import SimpleAgent
from core.config import Config


logger = logging.getLogger(__name__)


class CodeAgent(SimpleAgent):
    """Specialized agent for code analysis, generation, and debugging."""
    
    def __init__(self, config: Config):
        system_prompt = """You are a specialized AI assistant focused on programming and software development. You excel at:

- Code analysis and review
- Code generation and implementation
- Debugging and troubleshooting
- Explaining programming concepts
- Providing best practices and patterns
- Code optimization suggestions
- Architecture and design recommendations

When working with code:
1. Always provide clear, well-commented code examples
2. Explain the reasoning behind your suggestions
3. Consider edge cases and error handling
4. Follow best practices for the specific programming language
5. Provide alternative approaches when relevant

You can work with multiple programming languages including Python, JavaScript, Java, C++, and others. Always specify the language and provide context for your code examples."""
        
        super().__init__(
            config=config,
            agent_id="code",
            name="Code Assistant",
            description="A specialized AI assistant for programming, code analysis, and software development",
            capabilities=[
                "code_generation",
                "code_analysis",
                "debugging",
                "code_review",
                "programming_explanations",
                "architecture_design",
                "best_practices"
            ],
            system_prompt=system_prompt
        )


async def main():
    """Run the code agent."""
    config = Config.load_from_yaml()
    agent = CodeAgent(config)
    
    # Register with supervisor
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            registration_data = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "endpoint": agent.get_endpoint(port=8003)
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
    await agent.run(port=8003)


if __name__ == "__main__":
    asyncio.run(main())
