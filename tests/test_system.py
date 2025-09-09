#!/usr/bin/env python3
"""Test script for the multi-agent system."""

import asyncio
import httpx
import json
import logging
from typing import Dict, Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemTester:
    """Test the multi-agent system functionality."""
    
    def __init__(self, supervisor_url: str = "http://localhost:8000"):
        self.supervisor_url = supervisor_url
        self.session_id = "test-session"
    
    async def test_health(self) -> bool:
        """Test supervisor health."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.supervisor_url}/health")
                if response.status_code == 200:
                    logger.info("✅ Supervisor health check passed")
                    return True
                else:
                    logger.error(f"❌ Supervisor health check failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"❌ Supervisor health check error: {e}")
            return False
    
    async def test_agent_registration(self) -> bool:
        """Test agent registration."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.supervisor_url}/agents")
                if response.status_code == 200:
                    agents = response.json()
                    logger.info(f"✅ Found {len(agents)} registered agents")
                    for agent in agents:
                        logger.info(f"  - {agent['name']} ({agent['agent_id']})")
                    return len(agents) > 0
                else:
                    logger.error(f"❌ Failed to get agents: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"❌ Agent registration test error: {e}")
            return False
    
    async def test_simple_query(self) -> bool:
        """Test a simple query."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.supervisor_url}/execute",
                    json={
                        "session_id": self.session_id,
                        "query": "Hello, can you help me with a simple question?"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info("✅ Simple query test passed")
                    logger.info(f"  Response: {result['response'][:100]}...")
                    logger.info(f"  Agents used: {result['agents_used']}")
                    logger.info(f"  Iterations: {result['iterations']}")
                    return True
                else:
                    logger.error(f"❌ Simple query test failed: {response.status_code}")
                    logger.error(f"  Response: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"❌ Simple query test error: {e}")
            return False
    
    async def test_code_query(self) -> bool:
        """Test a code-related query."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.supervisor_url}/execute",
                    json={
                        "session_id": self.session_id,
                        "query": "Write a Python function to calculate the factorial of a number"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info("✅ Code query test passed")
                    logger.info(f"  Response: {result['response'][:200]}...")
                    logger.info(f"  Agents used: {result['agents_used']}")
                    return True
                else:
                    logger.error(f"❌ Code query test failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"❌ Code query test error: {e}")
            return False
    
    async def test_creative_query(self) -> bool:
        """Test a creative query."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.supervisor_url}/execute",
                    json={
                        "session_id": self.session_id,
                        "query": "Write a short creative story about a robot learning to paint"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info("✅ Creative query test passed")
                    logger.info(f"  Response: {result['response'][:200]}...")
                    logger.info(f"  Agents used: {result['agents_used']}")
                    return True
                else:
                    logger.error(f"❌ Creative query test failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"❌ Creative query test error: {e}")
            return False
    
    async def test_conversation_memory(self) -> bool:
        """Test conversation memory."""
        try:
            async with httpx.AsyncClient() as client:
                # First message
                response1 = await client.post(
                    f"{self.supervisor_url}/execute",
                    json={
                        "session_id": self.session_id,
                        "query": "My name is Alice and I like programming"
                    }
                )
                
                if response1.status_code != 200:
                    logger.error("❌ First message failed")
                    return False
                
                # Follow-up message
                response2 = await client.post(
                    f"{self.supervisor_url}/execute",
                    json={
                        "session_id": self.session_id,
                        "query": "What's my name and what do I like?"
                    }
                )
                
                if response2.status_code == 200:
                    result = response2.json()
                    logger.info("✅ Conversation memory test passed")
                    logger.info(f"  Response: {result['response'][:200]}...")
                    return True
                else:
                    logger.error(f"❌ Conversation memory test failed: {response2.status_code}")
                    return False
        except Exception as e:
            logger.error(f"❌ Conversation memory test error: {e}")
            return False
    
    async def test_session_history(self) -> bool:
        """Test session history retrieval."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.supervisor_url}/sessions/{self.session_id}/history")
                
                if response.status_code == 200:
                    history = response.json()
                    logger.info("✅ Session history test passed")
                    logger.info(f"  Found {len(history['messages'])} messages in history")
                    return True
                else:
                    logger.error(f"❌ Session history test failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"❌ Session history test error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        logger.info("🚀 Starting multi-agent system tests...")
        
        tests = [
            ("Health Check", self.test_health),
            ("Agent Registration", self.test_agent_registration),
            ("Simple Query", self.test_simple_query),
            ("Code Query", self.test_code_query),
            ("Creative Query", self.test_creative_query),
            ("Conversation Memory", self.test_conversation_memory),
            ("Session History", self.test_session_history),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running {test_name}...")
            try:
                result = await test_func()
                results[test_name] = result
            except Exception as e:
                logger.error(f"❌ {test_name} failed with exception: {e}")
                results[test_name] = False
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """Print test summary."""
        logger.info("\n" + "="*50)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*50)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! The system is working correctly.")
        else:
            logger.info("⚠️  Some tests failed. Check the logs above for details.")


async def main():
    """Main test function."""
    tester = SystemTester()
    
    # Wait a moment for services to start
    logger.info("⏳ Waiting for services to start...")
    await asyncio.sleep(2)
    
    # Run tests
    results = await tester.run_all_tests()
    
    # Print summary
    tester.print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
