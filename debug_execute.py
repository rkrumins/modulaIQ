#!/usr/bin/env python3
"""Debug script to test the execute endpoint and identify 422 errors."""

import asyncio
import httpx
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_execute_endpoint():
    """Test the execute endpoint with different request formats."""
    
    # Test cases with different request formats
    test_cases = [
        {
            "name": "Basic Request",
            "data": {
                "session_id": "test-session-1",
                "query": "Hello, can you help me?"
            }
        },
        {
            "name": "Request with Context",
            "data": {
                "session_id": "test-session-2",
                "query": "What's the weather like?",
                "context": {"location": "New York"}
            }
        },
        {
            "name": "Request with Max Iterations",
            "data": {
                "session_id": "test-session-3",
                "query": "Help me write a Python function",
                "max_iterations": 5
            }
        },
        {
            "name": "Full Request",
            "data": {
                "session_id": "test-session-4",
                "query": "Create a creative story",
                "context": {"genre": "sci-fi"},
                "max_iterations": 10
            }
        }
    ]
    
    async with httpx.AsyncClient() as client:
        for test_case in test_cases:
            logger.info(f"\n🧪 Testing: {test_case['name']}")
            logger.info(f"Request data: {json.dumps(test_case['data'], indent=2)}")
            
            try:
                response = await client.post(
                    "http://localhost:8000/execute",
                    json=test_case['data'],
                    timeout=30.0
                )
                
                logger.info(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info("✅ Request successful!")
                    logger.info(f"Response: {result.get('response', '')[:100]}...")
                    logger.info(f"Agents used: {result.get('agents_used', [])}")
                elif response.status_code == 422:
                    logger.error("❌ Validation Error (422)")
                    try:
                        error_detail = response.json()
                        logger.error(f"Error details: {json.dumps(error_detail, indent=2)}")
                    except:
                        logger.error(f"Raw response: {response.text}")
                else:
                    logger.error(f"❌ Error {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    
            except Exception as e:
                logger.error(f"❌ Request failed: {e}")


async def test_supervisor_health():
    """Test supervisor health and agent registration."""
    logger.info("🔍 Checking supervisor health...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Check health
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Supervisor healthy - Registered agents: {data.get('registered_agents', 0)}")
            else:
                logger.error(f"❌ Supervisor health check failed: {response.status_code}")
                return False
                
            # Check agents
            response = await client.get("http://localhost:8000/agents", timeout=5.0)
            if response.status_code == 200:
                agents = response.json()
                logger.info(f"✅ Found {len(agents)} registered agents:")
                for agent in agents:
                    logger.info(f"  - {agent['name']} ({agent['agent_id']}) - Status: {agent['status']}")
            else:
                logger.error(f"❌ Failed to get agents: {response.status_code}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Supervisor check failed: {e}")
            return False


async def main():
    """Main debug function."""
    logger.info("🐛 Debugging Execute Endpoint")
    logger.info("="*50)
    
    # First check supervisor health
    supervisor_ok = await test_supervisor_health()
    
    if not supervisor_ok:
        logger.error("❌ Supervisor is not ready. Please start it first.")
        return
    
    # Test execute endpoint
    await test_execute_endpoint()
    
    logger.info("\n" + "="*50)
    logger.info("🔍 Debug Summary")
    logger.info("="*50)
    logger.info("If you see 422 errors, check the error details above.")
    logger.info("Common causes:")
    logger.info("- Missing required fields (session_id, query)")
    logger.info("- Invalid data types")
    logger.info("- Pydantic validation errors")


if __name__ == "__main__":
    asyncio.run(main())
