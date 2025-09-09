#!/usr/bin/env python3
"""Test script for the new planning and execution system."""

import asyncio
import httpx
import json
from datetime import datetime


async def test_planning_system():
    """Test the new planning system with complex queries."""
    
    base_url = "http://localhost:8000"
    
    # Test queries that should trigger the planning system
    test_queries = [
        {
            "name": "Fibonacci Research and Code",
            "query": "Research the origins and history of Fibonacci numbers and write a highly optimized Python implementation with performance analysis",
            "expected_agents": ["research", "code"]
        },
        {
            "name": "AI Ethics Analysis",
            "query": "Research the current state of AI ethics, analyze the pros and cons of different approaches, and create a comprehensive report with recommendations",
            "expected_agents": ["research", "analysis"]
        },
        {
            "name": "Machine Learning Tutorial",
            "query": "Research the fundamentals of machine learning, write Python code examples for different algorithms, and create a creative story explaining ML concepts to beginners",
            "expected_agents": ["research", "code", "creative"]
        },
        {
            "name": "Quantum Computing Overview",
            "query": "Research quantum computing basics, analyze current developments, write Python simulations of quantum algorithms, and create a visual diagram",
            "expected_agents": ["research", "code", "analysis"]
        }
    ]
    
    print("🧠 Testing Enhanced Planning System")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Check if supervisor is running
        try:
            health_response = await client.get(f"{base_url}/health")
            if health_response.status_code != 200:
                print("❌ Supervisor is not running. Please start it first.")
                return
            print("✅ Supervisor is running")
        except Exception as e:
            print(f"❌ Cannot connect to supervisor: {e}")
            return
        
        # Check available agents
        try:
            agents_response = await client.get(f"{base_url}/agents")
            if agents_response.status_code == 200:
                agents = agents_response.json()
                print(f"✅ Found {len(agents)} registered agents: {[agent['agent_id'] for agent in agents]}")
            else:
                print("❌ Could not retrieve agent list")
                return
        except Exception as e:
            print(f"❌ Error getting agents: {e}")
            return
        
        print("\n🚀 Running Planning System Tests")
        print("=" * 60)
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n📋 Test {i}: {test_case['name']}")
            print(f"Query: {test_case['query']}")
            print("-" * 40)
            
            try:
                # Execute the query
                start_time = datetime.now()
                response = await client.post(
                    f"{base_url}/execute",
                    json={
                        "session_id": f"test-planning-{i}",
                        "query": test_case["query"]
                    }
                )
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print(f"✅ Query executed successfully in {duration:.2f}s")
                    print(f"📊 Agents used: {result.get('agents_used', [])}")
                    print(f"🔄 Iterations: {result.get('iterations', 0)}")
                    print(f"📝 Response length: {len(result.get('response', ''))} characters")
                    
                    # Check if planning system was used
                    if result.get('iterations', 0) == 1 and len(result.get('agents_used', [])) > 1:
                        print("🎯 Planning system likely used (single iteration, multiple agents)")
                    elif result.get('iterations', 0) > 1:
                        print("🔄 Traditional ReAct system used (multiple iterations)")
                    else:
                        print("❓ System behavior unclear")
                    
                    # Show first 200 characters of response
                    response_preview = result.get('response', '')[:200]
                    print(f"📄 Response preview: {response_preview}...")
                    
                    # Check if expected agents were used
                    used_agents = set(result.get('agents_used', []))
                    expected_agents = set(test_case['expected_agents'])
                    if expected_agents.issubset(used_agents):
                        print("✅ Expected agents were used")
                    else:
                        missing = expected_agents - used_agents
                        print(f"⚠️  Expected agents not used: {missing}")
                    
                else:
                    print(f"❌ Query failed with status {response.status_code}")
                    print(f"Error: {response.text}")
                    
            except Exception as e:
                print(f"❌ Error executing query: {e}")
            
            print("-" * 40)
        
        print("\n🎉 Planning System Test Complete!")
        print("=" * 60)


async def test_parallel_execution():
    """Test parallel execution capabilities."""
    
    print("\n⚡ Testing Parallel Execution")
    print("=" * 40)
    
    base_url = "http://localhost:8000"
    
    # Query that should have parallel execution
    parallel_query = "Research the history of Python programming language and simultaneously analyze the performance characteristics of different Python data structures"
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            start_time = datetime.now()
            response = await client.post(
                f"{base_url}/execute",
                json={
                    "session_id": "test-parallel",
                    "query": parallel_query
                }
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Parallel query executed in {duration:.2f}s")
                print(f"📊 Agents used: {result.get('agents_used', [])}")
                print(f"🔄 Iterations: {result.get('iterations', 0)}")
                
                # If it used multiple agents in a single iteration, it likely used parallel execution
                if result.get('iterations', 0) == 1 and len(result.get('agents_used', [])) > 1:
                    print("🚀 Parallel execution detected!")
                else:
                    print("🔄 Sequential execution used")
                    
            else:
                print(f"❌ Parallel test failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error in parallel test: {e}")


if __name__ == "__main__":
    print("🧠 Enhanced Planning System Test Suite")
    print("=" * 60)
    print("This test suite validates the new planning and execution system")
    print("that creates execution graphs and runs tasks in parallel.")
    print()
    
    asyncio.run(test_planning_system())
    asyncio.run(test_parallel_execution())
