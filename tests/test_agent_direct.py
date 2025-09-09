#!/usr/bin/env python3
"""Test individual agents directly to identify 422 errors."""

import asyncio
import json
from agents.examples.general_agent import GeneralAgent
from core.config import Config


async def test_agent_directly():
    """Test an agent directly without the supervisor."""
    print("🧪 Testing Agent Directly")
    print("="*40)
    
    try:
        # Load configuration
        config = Config.load_from_yaml()
        
        # Create agent
        agent = GeneralAgent(config)
        print(f"✅ Created agent: {agent.name}")
        
        # Test the process_request method directly
        from agents.base_agent import AgentRequest
        
        request = AgentRequest(
            session_id="test-session",
            input_data={"query": "Hello, can you help me?"}
        )
        
        print("📤 Sending request to agent...")
        response = await agent.process_request(request)
        
        print("📥 Agent response:")
        print(f"  Success: {response.success}")
        print(f"  Response: {response.response[:100]}...")
        print(f"  Error: {response.error}")
        
        if response.success:
            print("✅ Agent is working correctly!")
            return True
        else:
            print("❌ Agent failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await test_agent_directly()
    
    if success:
        print("\n🎉 Agent test passed!")
        print("The 422 errors might be due to the agents not running.")
    else:
        print("\n❌ Agent test failed!")
        print("There might be an issue with the agent implementation.")


if __name__ == "__main__":
    asyncio.run(main())
