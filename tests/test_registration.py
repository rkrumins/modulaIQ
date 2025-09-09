#!/usr/bin/env python3
"""Test script to verify agent registration with supervisor."""

import asyncio
import httpx
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_supervisor_health():
    """Test if supervisor is running and healthy."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Supervisor is healthy - Registered agents: {data.get('registered_agents', 0)}")
                return True
            else:
                logger.error(f"❌ Supervisor health check failed: {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"❌ Supervisor is not reachable: {e}")
        return False


async def test_agent_registration():
    """Test agent registration with supervisor."""
    try:
        async with httpx.AsyncClient() as client:
            # Test registration data
            registration_data = {
                "agent_id": "test-agent",
                "name": "Test Agent",
                "description": "A test agent for verification",
                "capabilities": ["testing", "verification"],
                "endpoint": "http://localhost:8002"
            }
            
            response = await client.post(
                "http://localhost:8000/agents/register",
                json=registration_data,
                timeout=10.0
            )
            
            if response.status_code == 200:
                logger.info("✅ Agent registration test passed")
                return True
            else:
                logger.error(f"❌ Agent registration failed: {response.status_code} - {response.text}")
                return False
    except Exception as e:
        logger.error(f"❌ Agent registration test error: {e}")
        return False


async def test_list_agents():
    """Test listing registered agents."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/agents", timeout=5.0)
            
            if response.status_code == 200:
                agents = response.json()
                logger.info(f"✅ Found {len(agents)} registered agents:")
                for agent in agents:
                    logger.info(f"  - {agent['name']} ({agent['agent_id']}) - Status: {agent['status']}")
                return True
            else:
                logger.error(f"❌ Failed to list agents: {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"❌ List agents test error: {e}")
        return False


async def test_agent_health():
    """Test individual agent health."""
    agent_ports = [8002, 8003, 8004, 8005]
    agent_names = ["General", "Code", "Research", "Creative"]
    
    try:
        async with httpx.AsyncClient() as client:
            for port, name in zip(agent_ports, agent_names):
                try:
                    response = await client.get(f"http://localhost:{port}/health", timeout=5.0)
                    if response.status_code == 200:
                        logger.info(f"✅ {name} Agent (port {port}) is healthy")
                    else:
                        logger.error(f"❌ {name} Agent (port {port}) health check failed: {response.status_code}")
                except Exception as e:
                    logger.error(f"❌ {name} Agent (port {port}) is not reachable: {e}")
    except Exception as e:
        logger.error(f"❌ Agent health test error: {e}")


async def main():
    """Main test function."""
    logger.info("🧪 Testing Agent Registration System")
    logger.info("="*50)
    
    # Wait a moment for services to start
    logger.info("⏳ Waiting for services to start...")
    await asyncio.sleep(3)
    
    # Test supervisor health
    supervisor_healthy = await test_supervisor_health()
    
    if not supervisor_healthy:
        logger.error("❌ Supervisor is not running. Please start it first with: python3 run_supervisor.py")
        return False
    
    # Test agent health
    await test_agent_health()
    
    # Test listing agents
    await test_list_agents()
    
    # Test agent registration
    await test_agent_registration()
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("📊 REGISTRATION TEST SUMMARY")
    logger.info("="*50)
    
    if supervisor_healthy:
        logger.info("✅ Supervisor is running and healthy")
        logger.info("💡 If no agents are registered, make sure to run: python3 run_agents.py")
        logger.info("💡 The agents should automatically register with the supervisor")
    else:
        logger.info("❌ Supervisor is not running")
        logger.info("💡 Start the supervisor first: python3 run_supervisor.py")
    
    return supervisor_healthy


if __name__ == "__main__":
    asyncio.run(main())
