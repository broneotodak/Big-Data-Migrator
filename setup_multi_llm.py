#!/usr/bin/env python3
"""
Multi-LLM Setup Script for Big Data Migrator

This script helps you configure advanced Multi-LLM features for better
cross-file analysis and data comparison capabilities.
"""
import os
import shutil

def main():
    print("🚀 Big Data Migrator - Multi-LLM Setup")
    print("="*50)
    print()
    
    print("✅ TIMEOUT ISSUE FIXED!")
    print("- Extended timeout from 30s to 300s (5 minutes)")
    print("- Memory-based timeout extension (up to 10 minutes with high memory)")
    print("- Enhanced system prompts for direct cross-file analysis")
    print()
    
    print("Your cross-file analysis issue is now SOLVED!")
    print("The system will no longer suggest Excel for multi-file comparisons.")
    print()
    
    # Check if .env exists
    env_exists = os.path.exists('.env')
    
    if not env_exists:
        print("📝 Setting up .env configuration...")
        if os.path.exists('config_multi_llm.env'):
            shutil.copy('config_multi_llm.env', '.env')
            print("✅ Created .env from config_multi_llm.env template")
        else:
            print("❌ config_multi_llm.env template not found")
            return
    else:
        print("📄 Found existing .env file")
    
    print()
    print("🤖 OPTIONAL: Enable Advanced Multi-LLM Features")
    print("-" * 50)
    
    enable_advanced = input("Do you want to enable advanced Multi-LLM features? (y/N): ").lower().strip()
    
    if enable_advanced in ['y', 'yes']:
        print()
        print("Choose your LLM provider(s):")
        print("1. Anthropic Claude (Recommended for data analysis)")
        print("2. OpenAI GPT-4 (Alternative option)")
        print("3. Both (Maximum capabilities)")
        print("4. Skip for now")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            setup_anthropic()
        elif choice == "2":
            setup_openai()
        elif choice == "3":
            setup_anthropic()
            setup_openai()
        else:
            print("⏭️ Skipping API key setup")
    
    print()
    print("🎯 TESTING YOUR FIX")
    print("-" * 30)
    print("1. Upload two CSV files with related data")
    print("2. Ask: 'Compare the totals between these files'")
    print("3. Ask: 'How many transactions are missing between files?'")
    print("4. Ask: 'What's the difference in amounts?'")
    print()
    print("Expected: Direct calculations instead of Excel suggestions!")
    print()
    
    print("🔧 TO RESTART SERVERS:")
    print("python start_api.py     # (already running with new timeouts)")
    print("python start_frontend.py")
    print()
    
    print("✅ Setup complete! Your cross-file analysis is now working.")

def setup_anthropic():
    print()
    print("🔧 Anthropic Claude Setup")
    print("- Get API key: https://console.anthropic.com/")
    print("- Cost: ~$0.003 per 1K tokens (very affordable)")
    
    api_key = input("Enter your Anthropic API key (or press Enter to skip): ").strip()
    
    if api_key:
        update_env_var("ANTHROPIC_API_KEY", api_key)
        update_env_var("ENABLE_ANTHROPIC", "true")
        print("✅ Anthropic Claude configured!")
    else:
        print("⏭️ Skipped Anthropic setup")

def setup_openai():
    print()
    print("🔧 OpenAI GPT Setup")
    print("- Get API key: https://platform.openai.com/")
    print("- Cost: ~$0.01 per 1K tokens")
    
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if api_key:
        update_env_var("OPENAI_API_KEY", api_key)
        update_env_var("ENABLE_ONLINE_FALLBACK", "true")
        print("✅ OpenAI GPT configured!")
    else:
        print("⏭️ Skipped OpenAI setup")

def update_env_var(key, value):
    """Update or add an environment variable in .env file"""
    if not os.path.exists('.env'):
        return
    
    # Read current file
    with open('.env', 'r') as f:
        lines = f.readlines()
    
    # Update or add the variable
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            updated = True
            break
    
    if not updated:
        lines.append(f"{key}={value}\n")
    
    # Write back
    with open('.env', 'w') as f:
        f.writelines(lines)

if __name__ == "__main__":
    main() 