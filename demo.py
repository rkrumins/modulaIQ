#!/usr/bin/env python3
"""Demo script showing how to use the multi-agent system."""

import asyncio
import httpx
import json
import logging
from typing import Dict, Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentDemo:
    """Demo class for the multi-agent system."""
    
    def __init__(self, supervisor_url: str = "http://localhost:8000"):
        self.supervisor_url = supervisor_url
        self.session_id = "demo-session"
    
    async def demo_simple_query(self):
        """Demo a simple query."""
        logger.info("\n🔍 Demo 1: Simple Query")
        logger.info("=" * 50)
        
        query = "Hello! Can you help me understand what machine learning is?"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.supervisor_url}/execute",
                json={
                    "session_id": self.session_id,
                    "query": query
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Query: {query}")
                logger.info(f"Response: {result['response']}")
                logger.info(f"Agents used: {result['agents_used']}")
                logger.info(f"Iterations: {result['iterations']}")
            else:
                logger.error(f"Error: {response.status_code} - {response.text}")
    
    async def demo_code_query(self):
        """Demo a code-related query."""
        logger.info("\n💻 Demo 2: Code Query")
        logger.info("=" * 50)
        
        query = "Write a Python function to sort a list of dictionaries by a specific key"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.supervisor_url}/execute",
                json={
                    "session_id": self.session_id,
                    "query": query
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Query: {query}")
                logger.info(f"Response: {result['response']}")
                logger.info(f"Agents used: {result['agents_used']}")
            else:
                logger.error(f"Error: {response.status_code} - {response.text}")
    
    async def demo_creative_query(self):
        """Demo a creative query."""
        logger.info("\n🎨 Demo 3: Creative Query")
        logger.info("=" * 50)
        
        query = "Write a short creative story about a robot who discovers emotions"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.supervisor_url}/execute",
                json={
                    "session_id": self.session_id,
                    "query": query
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Query: {query}")
                logger.info(f"Response: {result['response']}")
                logger.info(f"Agents used: {result['agents_used']}")
            else:
                logger.error(f"Error: {response.status_code} - {response.text}")
    
    async def demo_research_query(self):
        """Demo a research query."""
        logger.info("\n🔬 Demo 4: Research Query")
        logger.info("=" * 50)
        
        query = "Research the latest developments in quantum computing and summarize the key findings"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.supervisor_url}/execute",
                json={
                    "session_id": self.session_id,
                    "query": query
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Query: {query}")
                logger.info(f"Response: {result['response']}")
                logger.info(f"Agents used: {result['agents_used']}")
            else:
                logger.error(f"Error: {response.status_code} - {response.text}")
    
    async def demo_conversation_memory(self):
        """Demo conversation memory."""
        logger.info("\n🧠 Demo 5: Conversation Memory")
        logger.info("=" * 50)
        
        # First message
        query1 = "My name is Alice and I'm a software engineer"
        
        async with httpx.AsyncClient() as client:
            response1 = await client.post(
                f"{self.supervisor_url}/execute",
                json={
                    "session_id": self.session_id,
                    "query": query1
                }
            )
            
            if response1.status_code == 200:
                logger.info(f"Message 1: {query1}")
                logger.info(f"Response: {response1.json()['response']}")
            
            # Follow-up message
            query2 = "What's my name and profession?"
            
            response2 = await client.post(
                f"{self.supervisor_url}/execute",
                json={
                    "session_id": self.session_id,
                    "query": query2
                }
            )
            
            if response2.status_code == 200:
                result2 = response2.json()
                logger.info(f"Message 2: {query2}")
                logger.info(f"Response: {result2['response']}")
                logger.info("✅ Memory test passed - the system remembered the context!")
            else:
                logger.error(f"Error: {response2.status_code} - {response2.text}")
    
    async def demo_agent_registration(self):
        """Demo agent registration."""
        logger.info("\n📋 Demo 6: Agent Registration")
        logger.info("=" * 50)
        
        async with httpx.AsyncClient() as client:
            # List registered agents
            response = await client.get(f"{self.supervisor_url}/agents")
            
            if response.status_code == 200:
                agents = response.json()
                logger.info(f"Registered agents ({len(agents)}):")
                for agent in agents:
                    logger.info(f"  - {agent['name']} ({agent['agent_id']})")
                    logger.info(f"    Capabilities: {', '.join(agent['capabilities'])}")
                    logger.info(f"    Status: {agent['status']}")
            else:
                logger.error(f"Error: {response.status_code} - {response.text}")
    
    async def demo_session_history(self):
        """Demo session history."""
        logger.info("\n📚 Demo 7: Session History")
        logger.info("=" * 50)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.supervisor_url}/sessions/{self.session_id}/history")
            
            if response.status_code == 200:
                history = response.json()
                logger.info(f"Session history for '{self.session_id}':")
                for i, message in enumerate(history['messages'], 1):
                    logger.info(f"  {i}. {message['role']}: {message['content'][:100]}...")
            else:
                logger.error(f"Error: {response.status_code} - {response.text}")
    
    async def run_all_demos(self):
        """Run all demos."""
        logger.info("🚀 Starting Multi-Agent System Demo")
        logger.info("=" * 60)
        
        demos = [
            self.demo_agent_registration,
            self.demo_simple_query,
            self.demo_code_query,
            self.demo_creative_query,
            self.demo_research_query,
            self.demo_conversation_memory,
            self.demo_session_history,
        ]
        
        for demo in demos:
            try:
                await demo()
                await asyncio.sleep(1)  # Brief pause between demos
            except Exception as e:
                logger.error(f"Demo failed: {e}")
        
        logger.info("\n🎉 Demo completed!")
        logger.info("=" * 60)


async def main():
    """Main demo function."""
    demo = MultiAgentDemo()
    
    # Wait a moment for services to start
    logger.info("⏳ Waiting for services to start...")
    await asyncio.sleep(3)
    
    # Run demos
    await demo.run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())
