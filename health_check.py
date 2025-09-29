#!/usr/bin/env python3
"""
FastAPI Backend Health Check Script
Run this to verify your backend is working properly
"""

import requests
import time
import sys

def check_backend_health():
    """Check if the FastAPI backend is healthy"""
    
    base_url = "http://127.0.0.1:8000"
    
    print("🔍 Checking FastAPI Backend Health...")
    print(f"📍 Base URL: {base_url}")
    
    # Test 1: Basic ping endpoint
    try:
        print("\n1️⃣ Testing ping endpoint...")
        response = requests.get(f"{base_url}/ping", timeout=5)
        if response.status_code == 200:
            print("✅ Ping endpoint: OK")
        else:
            print(f"❌ Ping endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ping endpoint error: {e}")
        return False
    
    # Test 2: Health check endpoint
    try:
        print("\n2️⃣ Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health endpoint: {data.get('status', 'unknown')}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False
    
    # Test 3: Root endpoint
    try:
        print("\n3️⃣ Testing root endpoint...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint: OK")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False
    
    # Test 4: WebSocket test endpoint
    try:
        print("\n4️⃣ Testing WebSocket availability...")
        response = requests.get(f"{base_url}/ws-test", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ WebSocket availability: {data.get('websocket_available', False)}")
        else:
            print(f"❌ WebSocket test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ WebSocket test error: {e}")
        return False
    
    return True

def stress_test_endpoints(duration=30, requests_per_second=2):
    """Light stress test to check for hanging"""
    
    print(f"\n🚀 Running light stress test for {duration} seconds...")
    print(f"📊 Making {requests_per_second} requests per second")
    
    base_url = "http://127.0.0.1:8000"
    success_count = 0
    error_count = 0
    timeout_count = 0
    
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        try:
            # Test ping endpoint with short timeout
            response = requests.get(f"{base_url}/ping", timeout=2)
            if response.status_code == 200:
                success_count += 1
                print(".", end="", flush=True)
            else:
                error_count += 1
                print("E", end="", flush=True)
        except requests.exceptions.Timeout:
            timeout_count += 1
            print("T", end="", flush=True)
        except Exception:
            error_count += 1
            print("X", end="", flush=True)
        
        # Wait before next request
        time.sleep(1.0 / requests_per_second)
    
    print("\n")
    print(f"📈 Stress test results:")
    print(f"   ✅ Successful requests: {success_count}")
    print(f"   ❌ Error requests: {error_count}")
    print(f"   ⏰ Timeout requests: {timeout_count}")
    
    if timeout_count > success_count * 0.1:  # More than 10% timeouts
        print("⚠️  WARNING: High timeout rate detected - backend may be hanging")
        return False
    else:
        print("✅ Backend appears stable under light load")
        return True

if __name__ == "__main__":
    print("🔧 FastAPI Backend Health Checker")
    print("=" * 50)
    
    # Basic health check
    if not check_backend_health():
        print("\n❌ Backend health check FAILED")
        sys.exit(1)
    
    print("\n✅ All basic health checks PASSED")
    
    # Ask user if they want to run stress test
    try:
        run_stress = input("\n🤔 Run light stress test? (y/N): ").lower().strip()
        if run_stress in ['y', 'yes']:
            if stress_test_endpoints():
                print("\n🎉 All tests PASSED - Backend is healthy!")
            else:
                print("\n⚠️  Stress test detected issues")
                sys.exit(1)
        else:
            print("\n🎉 Basic health checks PASSED!")
    except KeyboardInterrupt:
        print("\n\n👋 Health check cancelled by user")
    
    print("\n💡 Backend appears to be working correctly!")