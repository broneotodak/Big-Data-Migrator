#!/usr/bin/env python3
"""
Quick server status check script
"""
import requests
import time

def check_server_status():
    """Check if both API and frontend servers are running."""
    print("🔍 Checking server status...")
    
    # Check API server
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API Server (FastAPI): Running at http://localhost:8000")
            print(f"   Response: {response.json()}")
        else:
            print(f"⚠️  API Server: Unexpected status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ API Server: Not running (connection refused)")
    except requests.exceptions.Timeout:
        print("⏱️  API Server: Timeout (may still be starting)")
    except Exception as e:
        print(f"❌ API Server: Error - {str(e)}")
    
    # Check Streamlit (frontend)
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend (Streamlit): Running at http://localhost:8501")
        else:
            print(f"⚠️  Frontend: Unexpected status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Frontend: Not running (connection refused)")
    except requests.exceptions.Timeout:
        print("⏱️  Frontend: Timeout (may still be starting)")
    except Exception as e:
        print(f"❌ Frontend: Error - {str(e)}")
    
    # Check LM Studio
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ LM Studio: Running with models available")
            if models.get("data"):
                print(f"   Active models: {len(models['data'])}")
        else:
            print(f"⚠️  LM Studio: Unexpected status {response.status_code}")
    except Exception as e:
        print(f"❌ LM Studio: {str(e)}")
    
    print("\n🚀 If all servers are running, open: http://localhost:8501")

if __name__ == "__main__":
    check_server_status() 