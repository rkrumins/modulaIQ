#!/usr/bin/env python3
"""Test script to verify the complete fix for 422 errors."""

import requests
import json
import time


def test_agent_directly():
    """Test an agent directly."""
    print("🧪 Testing Agent Directly")
    print("-" * 30)
    
    try:
        response = requests.post(
            "http://localhost:8002/execute",
            json={
                "session_id": "test-session",
                "input_data": {"query": "Hello"},
                "context": {}
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Agent working correctly")
            print(f"   Response: {result.get('response', '')[:50]}...")
            return True
        else:
            print(f"❌ Agent error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False


def test_supervisor():
    """Test the supervisor."""
    print("\n🧪 Testing Supervisor")
    print("-" * 30)
    
    try:
        response = requests.post(
            "http://localhost:8000/execute",
            json={
                "session_id": "test-session",
                "query": "Hello, can you help me?"
            },
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Supervisor working correctly")
            print(f"   Response: {result.get('response', '')[:100]}...")
            print(f"   Agents used: {result.get('agents_used', [])}")
            print(f"   Iterations: {result.get('iterations', 0)}")
            
            # Check if agents were actually called
            if result.get('agents_used'):
                print("✅ Agents were successfully called by supervisor")
                return True
            else:
                print("⚠️  No agents were used - checking for errors...")
                actions = result.get('actions_taken', [])
                for action in actions:
                    if 'Error calling agent' in action.get('observation', ''):
                        print(f"   ❌ {action.get('observation', '')}")
                return False
                
        elif response.status_code == 422:
            print("❌ Supervisor still returning 422 error")
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail.get('detail', 'Unknown error')}")
            except:
                print(f"   Raw response: {response.text}")
            return False
            
        else:
            print(f"❌ Supervisor error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Supervisor test failed: {e}")
        return False


def check_system_status():
    """Check system status."""
    print("🔍 Checking System Status")
    print("-" * 30)
    
    # Check supervisor health
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Supervisor healthy - Registered agents: {data.get('registered_agents', 0)}")
        else:
            print(f"❌ Supervisor health check failed: {response.status_code}")
            return False
    except:
        print("❌ Supervisor not reachable")
        return False
    
    # Check agents
    agent_ports = [8002, 8003, 8004, 8005]
    agent_names = ["General", "Code", "Research", "Creative"]
    
    for port, name in zip(agent_ports, agent_names):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} Agent (port {port}) healthy")
            else:
                print(f"❌ {name} Agent (port {port}) unhealthy: {response.status_code}")
        except:
            print(f"❌ {name} Agent (port {port}) not reachable")
    
    return True


def main():
    """Main test function."""
    print("🔧 Complete Fix Test")
    print("=" * 50)
    
    # Check system status
    if not check_system_status():
        print("\n❌ System not ready. Please ensure:")
        print("1. Supervisor is running: python3 run_supervisor.py")
        print("2. Agents are running: python3 run_agents.py")
        return False
    
    # Test agent directly
    agent_ok = test_agent_directly()
    
    # Test supervisor
    supervisor_ok = test_supervisor()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    if agent_ok and supervisor_ok:
        print("🎉 All tests passed!")
        print("✅ The 422 error fix is working correctly")
        print("✅ Agents are being called successfully by the supervisor")
    elif agent_ok and not supervisor_ok:
        print("⚠️  Agents work but supervisor has issues")
        print("💡 The supervisor may need to be restarted to pick up code changes")
        print("💡 Run: python3 restart_supervisor.py")
    else:
        print("❌ Tests failed")
        print("💡 Check the error messages above for details")
    
    return agent_ok and supervisor_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
