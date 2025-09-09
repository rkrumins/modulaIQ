#!/usr/bin/env python3
"""Test script to verify agent selection is working correctly."""

import asyncio
import httpx
import json

async def test_agent_selection():
    """Test that the supervisor selects the correct agents for different types of queries."""
    
    print("🧪 Testing Agent Selection...")
    print("=" * 60)
    
    # Test cases with expected agents
    test_cases = [
        {
            "query": "Write a Python function to calculate Fibonacci numbers",
            "expected_agent": "code",
            "description": "Programming question"
        },
        {
            "query": "What is the capital of France?",
            "expected_agent": "research", 
            "description": "Research question"
        },
        {
            "query": "Write a creative story about a robot",
            "expected_agent": "creative",
            "description": "Creative writing question"
        },
        {
            "query": "Hello, how are you?",
            "expected_agent": "general",
            "description": "General conversation"
        }
    ]
    
    results = []
    
    async with httpx.AsyncClient() as client:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📤 Test {i}: {test_case['description']}")
            print(f"   Query: {test_case['query']}")
            print(f"   Expected agent: {test_case['expected_agent']}")
            
            try:
                response = await client.post(
                    "http://localhost:8000/execute",
                    json={
                        "session_id": f"test-{i}",
                        "query": test_case["query"]
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    agents_used = result.get("agents_used", [])
                    actions_taken = result.get("actions_taken", [])
                    
                    print(f"   ✅ Response received")
                    print(f"   Agents used: {agents_used}")
                    
                    # Check if the expected agent was used
                    if test_case["expected_agent"] in agents_used:
                        print(f"   ✅ CORRECT: Expected agent '{test_case['expected_agent']}' was used")
                        success = True
                    else:
                        print(f"   ❌ INCORRECT: Expected agent '{test_case['expected_agent']}' was NOT used")
                        print(f"   Actual agents used: {agents_used}")
                        success = False
                    
                    # Show the first action to see what agent was called
                    if actions_taken:
                        first_action = actions_taken[0]
                        if "call_agent" in first_action.get("observation", ""):
                            observation = first_action["observation"]
                            if "Agent" in observation and "responded" in observation:
                                # Extract agent name from observation
                                agent_name = observation.split("Agent ")[1].split(" responded")[0]
                                print(f"   First agent called: {agent_name}")
                    
                    results.append({
                        "test": i,
                        "description": test_case["description"],
                        "expected": test_case["expected_agent"],
                        "actual": agents_used,
                        "success": success
                    })
                    
                else:
                    print(f"   ❌ Error: {response.status_code} - {response.text}")
                    results.append({
                        "test": i,
                        "description": test_case["description"],
                        "expected": test_case["expected_agent"],
                        "actual": "error",
                        "success": False
                    })
                
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                results.append({
                    "test": i,
                    "description": test_case["description"],
                    "expected": test_case["expected_agent"],
                    "actual": "exception",
                    "success": False
                })
            
            # Small delay between tests
            if i < len(test_cases):
                print("   ⏳ Waiting 3 seconds before next test...")
                await asyncio.sleep(3)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 AGENT SELECTION TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results if r["success"])
    total_tests = len(results)
    
    print(f"✅ Successful selections: {successful_tests}/{total_tests}")
    print(f"🎯 Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    print("\n📋 DETAILED RESULTS:")
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"   {status} Test {result['test']}: {result['description']}")
        print(f"      Expected: {result['expected']}")
        print(f"      Actual: {result['actual']}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if successful_tests == total_tests:
        print("   ✅ All agent selections are working correctly!")
    elif successful_tests > total_tests // 2:
        print("   ⚠️  Most agent selections are working, but some need improvement")
    else:
        print("   ❌ Agent selection needs significant improvement")
        print("   🔧 Consider improving the system prompt or agent descriptions")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_agent_selection())
