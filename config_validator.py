#!/usr/bin/env python3
"""
Configuration validator and fixer for Big Data Migrator
"""
import os
from dotenv import load_dotenv
import traceback

def validate_and_fix_config():
    """Validate and fix configuration issues."""
    try:
        print("🔍 Configuration Validator & Fixer")
        print("=" * 50)
        
        # Check if .env file exists
        env_path = ".env"
        if os.path.exists(env_path):
            print(f"✅ Found .env file at: {env_path}")
            file_size = os.path.getsize(env_path)
            print(f"   File size: {file_size} bytes")
        else:
            print(f"❌ .env file not found at: {env_path}")
            return False
        
        # Load environment variables
        print("\n1. Loading environment variables...")
        load_result = load_dotenv(env_path)
        print(f"   Load result: {load_result}")
        
        # Check critical configuration
        print("\n2. Checking critical settings...")
        
        # API Host configuration
        api_host = os.getenv("API_HOST")
        api_port = os.getenv("API_PORT")
        
        print(f"   API_HOST: '{api_host}' (type: {type(api_host)})")
        print(f"   API_PORT: '{api_port}' (type: {type(api_port)})")
        
        # Check if we need to fix the host setting
        if api_host == "0.0.0.0":
            print("⚠️  WARNING: API_HOST is set to 0.0.0.0 which can cause connection issues on Windows")
            print("   Recommendation: Change API_HOST to 'localhost' in .env file")
        elif api_host == "localhost":
            print("✅ API_HOST is correctly set to localhost")
        elif api_host is None:
            print("❌ API_HOST is not set in .env file")
        else:
            print(f"ℹ️  API_HOST is set to: {api_host}")
            
        # LLM Configuration
        local_llm_model = os.getenv("LOCAL_LLM_MODEL")
        local_llm_url = os.getenv("LOCAL_LLM_URL")
        
        print(f"\n3. LLM Configuration:")
        print(f"   LOCAL_LLM_MODEL: '{local_llm_model}'")
        print(f"   LOCAL_LLM_URL: '{local_llm_url}'")
        
        if local_llm_model and "codellama" in local_llm_model.lower():
            print("✅ CodeLlama is configured correctly")
        elif local_llm_model is None:
            print("❌ LOCAL_LLM_MODEL is not set in .env file")
        else:
            print(f"⚠️  Current model: {local_llm_model}")
        
        # Memory settings
        max_memory = os.getenv("MAX_MEMORY_PERCENT")
        print(f"\n4. Memory Settings:")
        print(f"   MAX_MEMORY_PERCENT: '{max_memory}'")
        
        # Show all environment variables for debugging
        print(f"\n5. All environment variables from .env:")
        all_env_vars = {k: v for k, v in os.environ.items() if not k.startswith('_')}
        relevant_vars = {k: v for k, v in all_env_vars.items() if any(x in k.upper() for x in ['API', 'LLM', 'LOCAL', 'MEMORY', 'HOST', 'PORT'])}
        
        for key, value in sorted(relevant_vars.items()):
            print(f"   {key}: {value}")
        
        # Check for configuration consistency
        print(f"\n6. Configuration Status:")
        
        issues = []
        if api_host == "0.0.0.0":
            issues.append("API_HOST needs to be changed to 'localhost'")
        elif api_host is None:
            issues.append("API_HOST is not set")
            
        if local_llm_model is None:
            issues.append("LOCAL_LLM_MODEL is not set")
        elif "codellama" not in local_llm_model.lower():
            issues.append("LOCAL_LLM_MODEL should be set to CodeLlama")
        
        if issues:
            print("❌ Configuration issues found:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("✅ Configuration is correct!")
            return True
            
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    validate_and_fix_config() 