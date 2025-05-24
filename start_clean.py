#!/usr/bin/env python3
"""
Clean startup script for Big Data Migrator
"""
import os
import sys
import time
import subprocess
import requests
from config_validator import validate_and_fix_config

def check_lm_studio():
    """Check if LM Studio is running."""
    try:
        response = requests.get("http://127.0.0.1:1234/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_api_server():
    """Check if API server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_system():
    """Start the complete system."""
    print("🚀 Big Data Migrator - Clean Startup")
    print("=" * 50)
    
    # Step 1: Validate configuration
    print("\n1. Validating configuration...")
    if not validate_and_fix_config():
        print("❌ Configuration validation failed. Please fix issues and try again.")
        return False
    
    # Step 2: Check LM Studio
    print("\n2. Checking LM Studio...")
    if check_lm_studio():
        print("✅ LM Studio is running and accessible")
    else:
        print("❌ LM Studio is not running. Please start LM Studio first.")
        print("   Make sure it's running on http://127.0.0.1:1234")
        return False
    
    # Step 3: Start API Server
    print("\n3. Starting API Server...")
    if check_api_server():
        print("✅ API Server is already running")
    else:
        print("   Starting API server...")
        try:
            # Start API server in background
            subprocess.Popen([
                sys.executable, "main.py"
            ], cwd=os.getcwd())
            
            # Wait for API to start
            for i in range(10):
                time.sleep(2)
                if check_api_server():
                    print("✅ API Server started successfully")
                    break
                print(f"   Waiting for API server... ({i+1}/10)")
            else:
                print("❌ API Server failed to start")
                return False
        except Exception as e:
            print(f"❌ Error starting API server: {e}")
            return False
    
    # Step 4: Start Frontend
    print("\n4. Starting Frontend...")
    try:
        # Start frontend
        subprocess.Popen([
            sys.executable, "start_frontend.py"
        ], cwd=os.getcwd())
        
        print("✅ Frontend startup initiated")
        
        # Give it time to start
        time.sleep(3)
        
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return False
    
    # Step 5: Final status check
    print("\n5. Final System Status:")
    print(f"   🟢 LM Studio: {'✅ Running' if check_lm_studio() else '❌ Not running'}")
    print(f"   🟢 API Server: {'✅ Running' if check_api_server() else '❌ Not running'}")
    print(f"   🟢 Frontend: Started (check http://localhost:8501)")
    
    print("\n🎉 System startup complete!")
    print("📍 Access your Big Data Migrator at: http://localhost:8501")
    print("📍 API documentation at: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    success = start_system()
    if not success:
        print("\n❌ Startup failed. Check the errors above.")
        sys.exit(1) 