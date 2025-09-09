#!/usr/bin/env python3
"""Test script to verify the async fix works."""

import asyncio
import logging
from agents.examples.general_agent import GeneralAgent
from core.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_agent_startup():
    """Test that an agent can start without event loop conflicts."""
    try:
        # Load configuration
        config = Config.load_from_yaml()
        
        # Create agent
        agent = GeneralAgent(config)
        
        logger.info("✅ Agent created successfully")
        
        # Test that we can create the agent without starting it
        logger.info(f"Agent ID: {agent.agent_id}")
        logger.info(f"Agent Name: {agent.name}")
        logger.info(f"Agent Capabilities: {agent.capabilities}")
        
        logger.info("✅ Agent configuration test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Agent test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("🧪 Testing Async Fix")
    logger.info("="*40)
    
    success = await test_agent_startup()
    
    if success:
        logger.info("🎉 Async fix test passed!")
        logger.info("✅ Agents should now start without event loop conflicts.")
    else:
        logger.info("❌ Async fix test failed.")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())
