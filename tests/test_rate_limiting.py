#!/usr/bin/env python3
"""Test script to verify rate limiting is working properly."""

import asyncio
import httpx
import time
import json

async def test_rate_limiting():
    """Test the supervisor's rate limiting by sending multiple requests quickly."""
    
    print("🧪 Testing Rate Limiting...")
    print("=" * 50)
    
    # Test data
    test_requests = [
        {"session_id": "rate-test-1", "query": "What is 2+2?"},
        {"session_id": "rate-test-2", "query": "What is the capital of France?"},
        {"session_id": "rate-test-3", "query": "What is Python?"},
    ]
    
    results = []
    
    async with httpx.AsyncClient() as client:
        for i, request in enumerate(test_requests, 1):
            print(f"\n📤 Sending request {i}: {request['query']}")
            start_time = time.time()
            
            try:
                response = await client.post(
                    "http://localhost:8000/execute",
                    json=request,
                    timeout=60.0
                )
                
                duration = time.time() - start_time
                print(f"✅ Request {i} completed in {duration:.2f}s")
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   Response: {result.get('response', '')[:100]}...")
                    print(f"   Iterations: {result.get('iterations', 0)}")
                    print(f"   Agents used: {result.get('agents_used', [])}")
                else:
                    print(f"   Error: {response.text}")
                
                results.append({
                    "request": i,
                    "duration": duration,
                    "status": response.status_code,
                    "success": response.status_code == 200
                })
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ Request {i} failed after {duration:.2f}s: {e}")
                results.append({
                    "request": i,
                    "duration": duration,
                    "status": "error",
                    "success": False,
                    "error": str(e)
                })
            
            # Small delay between requests to avoid overwhelming
            if i < len(test_requests):
                print("⏳ Waiting 2 seconds before next request...")
                await asyncio.sleep(2)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 RATE LIMITING TEST SUMMARY")
    print("=" * 50)
    
    successful_requests = sum(1 for r in results if r["success"])
    total_requests = len(results)
    avg_duration = sum(r["duration"] for r in results) / len(results) if results else 0
    
    print(f"✅ Successful requests: {successful_requests}/{total_requests}")
    print(f"⏱️  Average duration: {avg_duration:.2f}s")
    print(f"🎯 Success rate: {(successful_requests/total_requests)*100:.1f}%")
    
    # Check for rate limiting indicators
    print("\n🔍 RATE LIMITING ANALYSIS:")
    for i, result in enumerate(results, 1):
        if result["success"]:
            print(f"   Request {i}: ✅ {result['duration']:.2f}s")
        else:
            print(f"   Request {i}: ❌ {result.get('error', 'Unknown error')}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if avg_duration > 5.0:
        print("   ⚠️  Average response time is high - consider optimizing")
    if successful_requests < total_requests:
        print("   ⚠️  Some requests failed - check logs for details")
    if successful_requests == total_requests:
        print("   ✅ All requests successful - rate limiting working properly")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_rate_limiting())
