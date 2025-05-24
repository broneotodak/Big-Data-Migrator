"""
Launch script for the Streamlit frontend
"""
import os
import subprocess
import argparse
from dotenv import load_dotenv


def main():
    """Main entry point for launching the Streamlit frontend."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch Big Data Migrator Streamlit frontend")
    parser.add_argument("--port", type=int, default=None, help="Override the Streamlit port")
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv(".env")  # Load from root directory, not config/.env
    
    # Get the Streamlit port
    port = args.port or os.getenv("STREAMLIT_PORT", "8501")
    
    # Display startup message
    print(f"Starting Big Data Migrator Streamlit frontend on port {port}...")
    print("Make sure the API server is running (python main.py)")
    
    # Build the command
    cmd = [
        "streamlit", "run", 
        os.path.join("app", "frontend", "app.py"),
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false"
    ]
    
    # Execute the command
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStreamlit server stopped")
    except Exception as e:
        print(f"Error starting Streamlit: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
