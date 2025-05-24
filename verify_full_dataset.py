#!/usr/bin/env python3
"""
Final verification that the system processes full datasets correctly.
"""
import requests
import json

def verify_full_dataset_processing():
    """Verify the system processes full datasets as requested by the user."""
    
    base_url = "http://localhost:8000"
    
    print("=== VERIFYING FULL DATASET PROCESSING ===")
    print("User requirement: Process ALL data in uploaded files, not just samples")
    
    # Create conversation
    conversation_data = {
        "title": "Full Dataset Verification",
        "data_files": ["temp/MMSDO_P_202412_EP810177.csv"]
    }
    
    print("\n1. Creating conversation with CSV file...")
    response = requests.post(f"{base_url}/llm/conversations", json=conversation_data)
    conversation_id = response.json()["conversation_id"]
    print(f"✅ Conversation created: {conversation_id}")
    
    # Test full dataset processing
    print("\n2. Testing full dataset calculation...")
    message_data = {
        "message": "Calculate the total sum of all Transaction Amount (RM) values in the complete dataset. Show your calculation and specify whether you're using the full dataset or samples."
    }
    
    response = requests.post(
        f"{base_url}/llm/conversations/{conversation_id}/messages", 
        json=message_data
    )
    
    result = response.json()
    llm_response = result["response"]
    
    print("LLM Response:")
    print(llm_response)
    
    # Analyze response
    print("\n3. Analysis:")
    
    # Check for full dataset indicators
    full_dataset_phrases = [
        "full dataset", "complete dataset", "entire dataset", 
        "all data", "total dataset", "whole file"
    ]
    
    sample_phrases = [
        "sample", "first 5", "visible", "limited to", "partial"
    ]
    
    found_full = [phrase for phrase in full_dataset_phrases if phrase.lower() in llm_response.lower()]
    found_sample = [phrase for phrase in sample_phrases if phrase.lower() in llm_response.lower()]
    
    print(f"Full dataset indicators: {found_full}")
    print(f"Sample indicators: {found_sample}")
    
    if found_full and not found_sample:
        print("✅ SUCCESS: LLM indicates it's using the full dataset")
    elif found_sample:
        print("❌ ISSUE: LLM still indicates it's using samples")
    else:
        print("⚠️  UNCLEAR: Cannot determine data source from response")
    
    # Check for reasonable sum value
    # The full dataset should have significantly more data than 5 sample rows
    import re
    numbers = re.findall(r'[\d,]+\.?\d*', llm_response)
    if numbers:
        largest_number = max([float(num.replace(',', '')) for num in numbers if '.' in num or len(num) > 3])
        print(f"Largest number in response: {largest_number}")
        
        if largest_number > 1000:  # Much larger than sample sum of ~577
            print("✅ Sum appears to be from full dataset (> 1000)")
        else:
            print("❌ Sum appears to be from sample data only")
    
    print(f"\n=== VERIFICATION COMPLETE ===")
    print("The Big Data Migrator should now process complete datasets as required.")

if __name__ == "__main__":
    verify_full_dataset_processing() 