#!/usr/bin/env python3
"""Test script to verify multi-agent workflows are working correctly."""

import asyncio
import httpx
import json

async def test_multi_agent_workflow():
    """Test that complex queries are properly broken down into multiple agent calls."""
    
    print("🧪 Testing Multi-Agent Workflow...")
    print("=" * 70)
    
    # Test cases that should require multiple agents
    test_cases = [
        {
            "query": "Write me a function to calculate Fibonacci number and can this be applied somewhere in the nature? Research the key areas for fibonacci applications in nature",
            "expected_agents": ["code", "research"],
            "description": "Programming + Research query"
        },
        {
            "query": "Create a marketing campaign for a new AI product and analyze the target market demographics",
            "expected_agents": ["creative", "research"],
            "description": "Creative + Research query"
        },
        {
            "query": "Debug this Python code and find the best practices for error handling in Python",
            "expected_agents": ["code", "research"],
            "description": "Code + Best practices query"
        },
        {
            "query": "Write a creative story about a robot and research the latest developments in robotics",
            "expected_agents": ["creative", "research"],
            "description": "Creative + Research query"
        }
    ]
    
    results = []
    
    async with httpx.AsyncClient() as client:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📤 Test {i}: {test_case['description']}")
            print(f"   Query: {test_case['query'][:80]}...")
            print(f"   Expected agents: {test_case['expected_agents']}")
            
            try:
                response = await client.post(
                    "http://localhost:8000/execute",
                    json={
                        "session_id": f"multi-agent-test-{i}",
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
                    if len(agents_used) > 1:
                        print(f"   ✅ MULTI-AGENT: Used {len(agents_used)} different agents")
                        multi_agent_success = True
                    else:
                        print(f"   ⚠️  SINGLE-AGENT: Only used {len(agents_used)} agent(s)")
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
                    
                    # Show the reasoning steps
                    reasoning = result.get("reasoning", [])
                    if reasoning:
                        print(f"   Reasoning steps: {len(reasoning)}")
                        for j, step in enumerate(reasoning[:3], 1):  # Show first 3 steps
                            print(f"     {j}. {step[:80]}...")
                    
                    # Show actions taken
                    if actions_taken:
                        print(f"   Actions breakdown:")
                        for j, action in enumerate(actions_taken, 1):
                            action_type = action.get("action", "unknown")
                            print(f"     {j}. {action_type}")
                    
                    results.append({
                        "test": i,
                        "description": test_case["description"],
                        "expected_agents": test_case["expected_agents"],
                        "actual_agents": agents_used,
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
    print("📊 MULTI-AGENT WORKFLOW TEST SUMMARY")
    print("=" * 70)
    
    multi_agent_tests = sum(1 for r in results if r["multi_agent_success"])
    correct_agent_tests = sum(1 for r in results if r["agent_selection_success"])
    total_tests = len(results)
    avg_iterations = sum(r["iterations"] for r in results) / len(results) if results else 0
    avg_actions = sum(r["actions_count"] for r in results) / len(results) if results else 0
    
    print(f"✅ Multi-agent workflows: {multi_agent_tests}/{total_tests}")
    print(f"✅ Correct agent selection: {correct_agent_tests}/{total_tests}")
    print(f"📊 Average iterations: {avg_iterations:.1f}")
    print(f"📊 Average actions: {avg_actions:.1f}")
    
    print("\n📋 DETAILED RESULTS:")
    for result in results:
        multi_status = "✅" if result["multi_agent_success"] else "❌"
        agent_status = "✅" if result["agent_selection_success"] else "❌"
        print(f"   Test {result['test']}: {result['description']}")
        print(f"     Multi-agent: {multi_status} | Agent selection: {agent_status}")
        print(f"     Expected: {result['expected_agents']} | Actual: {result['actual_agents']}")
        print(f"     Iterations: {result['iterations']} | Actions: {result['actions_count']}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if multi_agent_tests == total_tests and correct_agent_tests == total_tests:
        print("   ✅ Perfect! All multi-agent workflows are working correctly!")
    elif multi_agent_tests > total_tests * 0.7:
        print("   ✅ Multi-agent workflows are working well!")
    elif multi_agent_tests > total_tests * 0.5:
        print("   ⚠️  Multi-agent workflows need improvement")
    else:
        print("   ❌ Multi-agent workflows need significant improvement")
        print("   🔧 Consider enhancing the system prompt or reasoning logic")
    
    if avg_iterations < 3:
        print("   ⚠️  Low iteration count - queries might not be fully decomposed")
    elif avg_iterations > 8:
        print("   ⚠️  High iteration count - might be inefficient")
    else:
        print("   ✅ Iteration count looks good")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_multi_agent_workflow())
