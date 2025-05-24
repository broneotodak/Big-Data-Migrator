#!/usr/bin/env python3
"""Quick test for LLM connection"""
import requests

try:
    # Test new diagnostic endpoint
    print("Testing LLM connection...")
    response = requests.get("http://localhost:8000/llm/status", timeout=5)
    
    if response.status_code == 200:
        data = response.json()
        print("SUCCESS: Got LLM status!")
        print(f"LM Studio Connection: {data.get('lm_studio_connection', 'Unknown')}")
        print(f"Conversation System: {data.get('conversation_system', 'Unknown')}")
        print(f"LLM URL: {data.get('local_llm_url', 'Unknown')}")
        print(f"Model: {data.get('local_llm_model', 'Unknown')}")
    else:
        print(f"ERROR: Status {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"ERROR: {e}")
    
print("Test complete!") 