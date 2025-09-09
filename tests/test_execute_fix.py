#!/usr/bin/env python3
"""Test script to verify the execute endpoint fix."""

import json
import subprocess
import time
import requests


def test_execute_endpoint():
    """Test the execute endpoint."""
    print("🧪 Testing Execute Endpoint Fix")
    print("="*50)
    
    # Test data
    test_data = {
        "session_id": "test-session-fix",
        "query": "Hello, can you help me with a simple question?"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/execute",
            json=test_data,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Execute endpoint is working!")
            print(f"Response: {result.get('response', '')[:100]}...")
            print(f"Agents used: {result.get('agents_used', [])}")
            print(f"Iterations: {result.get('iterations', 0)}")
            print(f"Actions taken: {len(result.get('actions_taken', []))} actions")
            
            # Check if actions_taken is properly formatted
            actions = result.get('actions_taken', [])
            if actions and isinstance(actions[0], dict):
                print("✅ Actions taken format is correct (list of dictionaries)")
            else:
                print("❌ Actions taken format is incorrect")
                
            return True
            
        elif response.status_code == 422:
            print("❌ Still getting 422 validation error")
            try:
                error_detail = response.json()
                print(f"Error: {error_detail.get('detail', 'Unknown error')}")
            except:
                print(f"Raw response: {response.text}")
            return False
            
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to supervisor. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def check_supervisor_status():
    """Check if supervisor is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Supervisor is running - Registered agents: {data.get('registered_agents', 0)}")
            return True
        else:
            print(f"❌ Supervisor health check failed: {response.status_code}")
            return False
    except:
        print("❌ Supervisor is not running")
        return False


def main():
    """Main test function."""
    print("🔧 Execute Endpoint Fix Test")
    print("="*50)
    
    # Check supervisor status
    if not check_supervisor_status():
        print("\n💡 To fix this issue:")
        print("1. Stop the current supervisor (Ctrl+C)")
        print("2. Restart it with: python3 run_supervisor.py")
        print("3. The fix has been applied to the code")
        return
    
    # Test execute endpoint
    success = test_execute_endpoint()
    
    print("\n" + "="*50)
    print("📊 Test Summary")
    print("="*50)
    
    if success:
        print("🎉 Execute endpoint is working correctly!")
        print("✅ The Pydantic validation error has been fixed")
    else:
        print("❌ Execute endpoint still has issues")
        print("\n💡 If you're still seeing 422 errors:")
        print("1. The supervisor needs to be restarted to pick up the code changes")
        print("2. Stop the supervisor (Ctrl+C) and restart it")
        print("3. The fix has been applied to the code")


if __name__ == "__main__":
    main()
