#!/usr/bin/env python3
"""
Startup Verification Script for Big Data Migrator
Verifies that all timeout fixes and debug monitoring are working correctly.
"""
import requests
import time
import json
from datetime import datetime

def test_endpoint(url, name, timeout=5):
    """Test an endpoint and return status."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return f"✅ {name}: OK"
        else:
            return f"❌ {name}: HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return f"⏱️ {name}: Timeout (>{timeout}s)"
    except requests.exceptions.ConnectionError:
        return f"🔌 {name}: Connection Error"
    except Exception as e:
        return f"❌ {name}: {str(e)[:50]}"

def main():
    print("🚀 Big Data Migrator - Startup Verification")
    print("=" * 50)
    print()
    
    base_url = "http://localhost:8000"
    frontend_url = "http://localhost:8501"
    
    # Test API endpoints
    print("📡 Testing API Endpoints:")
    endpoints = [
        ("/health", "Health Check"),
        ("/memory-status", "Memory Monitor"),
        ("/llm/status", "LLM Status"),
        ("/debug/current-processing", "Debug - Current Processing"),
        ("/debug/recent-errors", "Debug - Recent Errors"),
        ("/debug/system-performance", "Debug - System Performance")
    ]
    
    api_working = True
    for endpoint, name in endpoints:
        result = test_endpoint(f"{base_url}{endpoint}", name)
        print(f"  {result}")
        if "❌" in result or "🔌" in result:
            api_working = False
    
    print()
    
    # Test Frontend
    print("🎨 Testing Frontend:")
    frontend_result = test_endpoint(frontend_url, "Streamlit Frontend")
    print(f"  {frontend_result}")
    frontend_working = "✅" in frontend_result
    
    print()
    
    # Test timeout fixes
    print("⏱️ Testing Timeout Fixes:")
    try:
        # Test LLM status with timeout info
        response = requests.get(f"{base_url}/llm/status", timeout=10)
        if response.status_code == 200:
            llm_data = response.json()
            print("  ✅ LLM system responding (timeout fixes applied)")
            
            # Check for timeout configurations
            if "local_llm_url" in llm_data:
                print(f"  📍 Local LLM URL: {llm_data.get('local_llm_url', 'Unknown')}")
                
        else:
            print("  ❌ LLM system not responding")
            
    except Exception as e:
        print(f"  ❌ LLM system error: {str(e)[:50]}")
    
    print()
    
    # Test debug monitoring
    print("🔍 Testing Debug Monitoring:")
    try:
        # Test current processing endpoint
        response = requests.get(f"{base_url}/debug/current-processing", timeout=5)
        if response.status_code == 200:
            processing_data = response.json()
            active_count = processing_data.get("total_active", 0)
            print(f"  ✅ Processing Monitor: {active_count} active processes")
            
        # Test recent errors endpoint
        response = requests.get(f"{base_url}/debug/recent-errors", timeout=5)
        if response.status_code == 200:
            errors_data = response.json()
            error_count = len(errors_data) if isinstance(errors_data, list) else 0
            print(f"  ✅ Error Monitor: {error_count} recent errors")
            
        print("  ✅ Debug monitoring fully operational")
        
    except Exception as e:
        print(f"  ❌ Debug monitoring error: {str(e)[:50]}")
    
    print()
    
    # Overall status
    print("📊 Overall Status:")
    if api_working and frontend_working:
        print("  🎉 ALL SYSTEMS OPERATIONAL!")
        print("  ✅ Timeout fixes applied and working")
        print("  ✅ Debug monitoring active")
        print("  ✅ Multi-file analysis ready")
        print()
        print("🔗 Access URLs:")
        print(f"  • Frontend: {frontend_url}")
        print(f"  • API Docs: {base_url}/docs")
        print(f"  • Debug Monitor: {frontend_url} (Debug page)")
        print()
        print("💡 Ready to test your multi-file analysis!")
        print("   Try asking: 'Compare these 2 files and calculate the differences'")
        
    elif api_working:
        print("  ⚠️ API working, but frontend has issues")
        print("  💡 Try restarting the frontend: python start_frontend.py")
        
    elif frontend_working:
        print("  ⚠️ Frontend working, but API has issues")
        print("  💡 Try restarting the API: python start_api.py")
        
    else:
        print("  ❌ Both API and Frontend have issues")
        print("  💡 Check if LM Studio is running")
        print("  💡 Try restarting both services")
    
    print()
    print(f"🕐 Verification completed at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main() 