#!/usr/bin/env python3
"""Test script to verify the AgentCall import fix."""

import asyncio
import httpx
import json

async def test_agentcall_fix():
    """Test that the AgentCall import fix resolves the undefined error."""
    
    print("🧪 Testing AgentCall Import Fix...")
    print("=" * 50)
    
    # Test request
    test_request = {
        "session_id": "agentcall-test",
        "query": "What is 2+2?"
    }
    
    print(f"📤 Sending test request: {test_request['query']}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/execute",
                json=test_request,
                timeout=30.0
            )
            
            print(f"✅ Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Response: {result.get('response', '')[:100]}...")
                print(f"   Iterations: {result.get('iterations', 0)}")
                print(f"   Agents used: {result.get('agents_used', [])}")
                
                # Check for the specific error
                actions_taken = result.get('actions_taken', [])
                for action in actions_taken:
                    observation = action.get('observation', '')
                    if 'AgentCall' in observation and 'not defined' in observation:
                        print(f"❌ Still seeing AgentCall error: {observation}")
                        return False
                
                print("✅ No AgentCall errors found!")
                return True
            else:
                print(f"❌ Error response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_agentcall_fix())
    if success:
        print("\n🎉 AgentCall fix is working!")
    else:
        print("\n⚠️  AgentCall fix needs more work")
