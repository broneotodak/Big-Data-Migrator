"""
Test specific queries that should trigger smart processing
"""
import requests
import json
import os
import time

def test_specific_queries():
    """Test the exact queries the user is using"""
    
    base_url = "http://localhost:8000"
    
    print("🔍 Testing User's Specific Queries")
    print("=" * 60)
    
    # File paths  
    file1_path = os.path.abspath("temp/MMSDO_P_202412_EP810177.csv")
    file2_path = os.path.abspath("temp/Payments by order - 2024-12-01 - 2024-12-31.csv")
    
    # Create conversation with files
    print(f"1. Creating conversation with files...")
    response = requests.post(f"{base_url}/llm/conversations", 
                           json={
                               "title": "Specific Query Test",
                               "data_files": [file1_path, file2_path]
                           })
    
    conversation_data = response.json()
    conversation_id = conversation_data["conversation_id"]
    print(f"✅ Created conversation: {conversation_id}")
    
    # Wait for data context to build
    time.sleep(3)
    
    # Get debug info first
    print(f"\n2. Checking smart processing status...")
    debug_response = requests.get(f"{base_url}/debug/conversation-debug/{conversation_id}")
    if debug_response.status_code == 200:
        debug_info = debug_response.json()
        print(f"   Smart processing enabled: {debug_info.get('smart_processing_enabled', False)}")
        print(f"   Active data files count: {debug_info.get('active_data_files_count', 0)}")
        print(f"   Active data context exists: {debug_info.get('active_data_context_exists', False)}")
        
        if debug_info.get('active_data_files_count', 0) == 0:
            print(f"   ❌ NO DATA FILES LOADED - This is the root cause!")
            return
        else:
            print(f"   ✅ Data files are loaded")
    
    # Test the user's exact queries
    test_queries = [
        "what can you explain about both files logical relations?",
        "can you find out how many transactions missing (the count) and how much is missing in RM?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i+2}. Testing query: '{query}'")
        
        # Test single LLM first
        single_response = requests.post(
            f"{base_url}/llm/conversations/{conversation_id}/messages",
            json={"message": query}
        )
        
        if single_response.status_code == 200:
            result = single_response.json()
            smart_processing = result.get('smart_processing', False)
            print(f"   Single LLM - Smart processing: {'✅' if smart_processing else '❌'}")
            
            if smart_processing:
                processed_results = result.get('processed_results', {})
                print(f"   Calculation method: {processed_results.get('calculation_method', 'unknown')}")
                print(f"   Primary answer preview: {processed_results.get('primary_answer', 'N/A')[:100]}...")
            else:
                # Check if response mentions specific data
                response_text = result.get('response', '')
                has_specific_data = any(indicator in response_text for indicator in [
                    'RM ', '34 transactions', '124 transactions', 'total:', 'amount:'
                ])
                
                if has_specific_data:
                    print(f"   ✅ Response contains specific data (good)")
                else:
                    print(f"   ❌ Response is generic (bad)")
                    
                # Check for requests for more data
                asks_for_data = any(phrase in response_text.lower() for phrase in [
                    'please share', 'need more information', 'without further information'
                ])
                
                if asks_for_data:
                    print(f"   ❌ LLM is asking for data instead of using loaded files!")
                
                print(f"   Response preview: {response_text[:200]}...")
        
        # Test multi-LLM
        multi_response = requests.post(
            f"{base_url}/llm/conversations/{conversation_id}/messages/multi",
            json={"message": query}
        )
        
        if multi_response.status_code == 200:
            multi_result = multi_response.json()
            consensus = multi_result.get('consensus_response', '')
            
            if consensus:
                has_specific_data = any(indicator in consensus for indicator in [
                    'RM ', '34 transactions', '124 transactions', 'total:', 'amount:'
                ])
                asks_for_data = any(phrase in consensus.lower() for phrase in [
                    'please share', 'need more information', 'without further information'
                ])
                
                print(f"   Multi-LLM consensus:")
                print(f"     Has specific data: {'✅' if has_specific_data else '❌'}")
                print(f"     Asks for data: {'❌ Bad' if asks_for_data else '✅ Good'}")
                print(f"     Preview: {consensus[:150]}...")

if __name__ == "__main__":
    test_specific_queries() 