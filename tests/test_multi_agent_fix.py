#!/usr/bin/env python3
"""Test script to verify the multi-agent fix is working correctly."""

import asyncio
import httpx
import json

async def test_multi_agent_fix():
    """Test that the multi-agent fix prevents premature termination."""
    
    print("🧪 Testing Multi-Agent Fix...")
    print("=" * 70)
    
    # Test the specific Fibonacci query that was problematic
    test_cases = [
        {
            "query": "Research the origins of Fibonacci and code this up in Python",
            "expected_agents": ["research", "code"],
            "description": "Fibonacci research + coding query (the original problem)"
        },
        {
            "query": "Write a Python function to calculate prime numbers and research their applications in cryptography",
            "expected_agents": ["code", "research"],
            "description": "Programming + research query"
        },
        {
            "query": "Create a marketing campaign for AI products and analyze the target market",
            "expected_agents": ["creative", "research"],
            "description": "Creative + research query"
        }
    ]
    
    results = []
    
    async with httpx.AsyncClient() as client:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📤 Test {i}: {test_case['description']}")
            print(f"   Query: {test_case['query']}")
            print(f"   Expected agents: {test_case['expected_agents']}")
            
            try:
                response = await client.post(
                    "http://localhost:8000/execute",
                    json={
                        "session_id": f"multi-agent-fix-test-{i}",
                        "query": test_case["query"]
                    },
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    agents_used = result.get("agents_used", [])
                    actions_taken = result.get("actions_taken", [])
                    iterations = result.get("iterations", 0)
                    
                    print(f"   ✅ Response received")
                    print(f"   Agents used: {agents_used}")
                    print(f"   Iterations: {iterations}")
                    print(f"   Actions taken: {len(actions_taken)}")
                    
                    # Check if multiple agents were used
                    unique_agents = len(set(agents_used))
                    if unique_agents >= 2:
                        print(f"   ✅ MULTI-AGENT SUCCESS: Used {unique_agents} different agents")
                        multi_agent_success = True
                    else:
                        print(f"   ❌ SINGLE-AGENT FAILURE: Only used {unique_agents} agent(s)")
                        multi_agent_success = False
                    
                    # Check if expected agents were used
                    expected_agents_used = all(agent in agents_used for agent in test_case["expected_agents"])
                    if expected_agents_used:
                        print(f"   ✅ CORRECT AGENTS: All expected agents were used")
                        agent_selection_success = True
                    else:
                        missing_agents = [agent for agent in test_case["expected_agents"] if agent not in agents_used]
                        print(f"   ❌ MISSING AGENTS: {missing_agents} were not used")
                        agent_selection_success = False
                    
                    # Show the reasoning steps to see the fix in action
                    reasoning = result.get("reasoning", [])
                    if reasoning:
                        print(f"   Reasoning steps: {len(reasoning)}")
                        for j, step in enumerate(reasoning, 1):
                            print(f"     {j}. {step[:100]}...")
                    
                    # Show actions taken
                    if actions_taken:
                        print(f"   Actions breakdown:")
                        for j, action in enumerate(actions_taken, 1):
                            action_type = action.get("action", "unknown")
                            reasoning = action.get("reasoning", "")[:80]
                            print(f"     {j}. {action_type}: {reasoning}...")
                    
                    results.append({
                        "test": i,
                        "description": test_case["description"],
                        "expected_agents": test_case["expected_agents"],
                        "actual_agents": agents_used,
                        "unique_agents": unique_agents,
                        "multi_agent_success": multi_agent_success,
                        "agent_selection_success": agent_selection_success,
                        "iterations": iterations,
                        "actions_count": len(actions_taken)
                    })
                    
                else:
                    print(f"   ❌ Error: {response.status_code} - {response.text}")
                    results.append({
                        "test": i,
                        "description": test_case["description"],
                        "expected_agents": test_case["expected_agents"],
                        "actual_agents": "error",
                        "unique_agents": 0,
                        "multi_agent_success": False,
                        "agent_selection_success": False,
                        "iterations": 0,
                        "actions_count": 0
                    })
                
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                results.append({
                    "test": i,
                    "description": test_case["description"],
                    "expected_agents": test_case["expected_agents"],
                    "actual_agents": "exception",
                    "unique_agents": 0,
                    "multi_agent_success": False,
                    "agent_selection_success": False,
                    "iterations": 0,
                    "actions_count": 0
                })
            
            # Small delay between tests
            if i < len(test_cases):
                print("   ⏳ Waiting 5 seconds before next test...")
                await asyncio.sleep(5)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 MULTI-AGENT FIX TEST SUMMARY")
    print("=" * 70)
    
    multi_agent_tests = sum(1 for r in results if r["multi_agent_success"])
    correct_agent_tests = sum(1 for r in results if r["agent_selection_success"])
    total_tests = len(results)
    avg_iterations = sum(r["iterations"] for r in results) / len(results) if results else 0
    avg_actions = sum(r["actions_count"] for r in results) / len(results) if results else 0
    avg_unique_agents = sum(r["unique_agents"] for r in results) / len(results) if results else 0
    
    print(f"✅ Multi-agent workflows: {multi_agent_tests}/{total_tests}")
    print(f"✅ Correct agent selection: {correct_agent_tests}/{total_tests}")
    print(f"📊 Average unique agents used: {avg_unique_agents:.1f}")
    print(f"📊 Average iterations: {avg_iterations:.1f}")
    print(f"📊 Average actions: {avg_actions:.1f}")
    
    print("\n📋 DETAILED RESULTS:")
    for result in results:
        multi_status = "✅" if result["multi_agent_success"] else "❌"
        agent_status = "✅" if result["agent_selection_success"] else "❌"
        print(f"   Test {result['test']}: {result['description']}")
        print(f"     Multi-agent: {multi_status} | Agent selection: {agent_status}")
        print(f"     Expected: {result['expected_agents']} | Actual: {result['actual_agents']}")
        print(f"     Unique agents: {result['unique_agents']} | Iterations: {result['iterations']}")
    
    # Recommendations
    print("\n💡 FIX EFFECTIVENESS:")
    if multi_agent_tests == total_tests:
        print("   ✅ PERFECT! The multi-agent fix is working correctly!")
        print("   🎉 All multi-part queries are now using multiple agents as expected.")
    elif multi_agent_tests > total_tests * 0.8:
        print("   ✅ GOOD! The multi-agent fix is mostly working.")
        print("   🔧 Minor improvements may be needed for edge cases.")
    elif multi_agent_tests > total_tests * 0.5:
        print("   ⚠️  PARTIAL: The multi-agent fix is partially working.")
        print("   🔧 Additional improvements needed.")
    else:
        print("   ❌ POOR: The multi-agent fix is not working effectively.")
        print("   🔧 Significant improvements needed.")
    
    if avg_unique_agents >= 2.0:
        print("   ✅ Excellent agent diversity - queries are properly decomposed.")
    elif avg_unique_agents >= 1.5:
        print("   ⚠️  Moderate agent diversity - some queries may need better decomposition.")
    else:
        print("   ❌ Poor agent diversity - queries are not being properly decomposed.")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_multi_agent_fix())
