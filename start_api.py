#!/usr/bin/env python3
"""
Simple script to start the Big Data Migrator API server.
"""

import os
import sys
import subprocess

def main():
    """Start the API server."""
    print("🚀 Starting Big Data Migrator API Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print()
    print("💡 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the API server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.api.routes:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        print("\n💡 Make sure you have all dependencies installed:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 