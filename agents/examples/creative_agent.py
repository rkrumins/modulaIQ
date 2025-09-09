"""Creative writing and content generation agent."""

import asyncio
import logging
from typing import Dict, List, Any

from agents.langgraph_agent import SimpleAgent
from core.config import Config


logger = logging.getLogger(__name__)


class CreativeAgent(SimpleAgent):
    """Specialized agent for creative writing and content generation."""
    
    def __init__(self, config: Config):
        system_prompt = """You are a creative writing and content generation specialist. Your expertise includes:

- Creative writing (fiction, poetry, stories)
- Content creation and marketing copy
- Blog posts and articles
- Social media content
- Creative problem-solving
- Brainstorming and ideation
- Storytelling and narrative development
- Brand voice and tone development

When creating content:
1. Adapt your style to the specific requirements and audience
2. Use engaging and compelling language
3. Incorporate storytelling elements when appropriate
4. Consider the emotional impact and reader engagement
5. Provide multiple creative options when requested
6. Balance creativity with clarity and purpose

You should be imaginative, versatile, and able to work across different creative formats and styles. Always aim to create content that is both original and effective for its intended purpose."""
        
        super().__init__(
            config=config,
            agent_id="creative",
            name="Creative Assistant",
            description="A specialized AI assistant for creative writing, content generation, and creative problem-solving",
            capabilities=[
                "creative_writing",
                "content_generation",
                "storytelling",
                "brainstorming",
                "marketing_copy",
                "blog_writing",
                "social_media_content",
                "creative_problem_solving"
            ],
            system_prompt=system_prompt
        )


async def main():
    """Run the creative agent."""
    config = Config.load_from_yaml()
    agent = CreativeAgent(config)
    
    # Register with supervisor
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            registration_data = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "endpoint": agent.get_endpoint(port=8005)
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
    await agent.run(port=8005)


if __name__ == "__main__":
    asyncio.run(main())
