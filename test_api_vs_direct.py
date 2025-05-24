"""
Compare API conversation creation vs direct conversation creation
"""
import requests
import json
import os
import time
import sys
sys.path.append('.')

from app.llm.conversation_system import LLMConversationSystem
from app.memory.memory_monitor import MemoryMonitor

def test_api_vs_direct():
    """Compare API and direct conversation creation"""
    
    base_url = "http://localhost:8000"
    
    print("🔍 Comparing API vs Direct Conversation Creation")
    print("=" * 60)
    
    # File paths
    file1_path = os.path.abspath("temp/MMSDO_P_202412_EP810177.csv")
    file2_path = os.path.abspath("temp/Payments by order - 2024-12-01 - 2024-12-31.csv")
    
    print(f"Using files:")
    print(f"  📄 {os.path.basename(file1_path)}")
    print(f"  📄 {os.path.basename(file2_path)}")
    
    # Test 1: Direct conversation system
    print(f"\n1. Testing DIRECT conversation system...")
    try:
        memory_monitor = MemoryMonitor()
        direct_llm_system = LLMConversationSystem(
            memory_monitor=memory_monitor,
            enable_smart_processing=True
        )
        
        direct_conversation_id = direct_llm_system.create_conversation(
            title="Direct Test",
            data_files=[file1_path, file2_path]
        )
        
        print(f"✅ Direct conversation created: {direct_conversation_id}")
        print(f"   Active data files: {len(direct_llm_system._active_data_files)}")
        print(f"   Smart processing enabled: {direct_llm_system.enable_smart_processing}")
        
        if direct_llm_system._active_data_files:
            for path, df in direct_llm_system._active_data_files.items():
                print(f"     - {os.path.basename(path)}: {len(df)} rows")
                
    except Exception as e:
        print(f"❌ Direct test failed: {str(e)}")
        return
    
    # Test 2: API conversation system
    print(f"\n2. Testing API conversation system...")
    try:
        api_response = requests.post(f"{base_url}/llm/conversations", 
                                   json={
                                       "title": "API Test",
                                       "data_files": [file1_path, file2_path]
                                   })
        
        if api_response.status_code == 200:
            api_data = api_response.json()
            api_conversation_id = api_data["conversation_id"]
            print(f"✅ API conversation created: {api_conversation_id}")
            
            # Wait a moment for data processing
            time.sleep(2)
            
            # Check debug info
            debug_response = requests.get(f"{base_url}/debug/conversation-debug/{api_conversation_id}")
            if debug_response.status_code == 200:
                debug_info = debug_response.json()
                print(f"   Active data files: {debug_info.get('active_data_files_count', 0)}")
                print(f"   Smart processing enabled: {debug_info.get('smart_processing_enabled', False)}")
                print(f"   Data files in conversation: {debug_info.get('data_files_count', 0)}")
                
                if debug_info.get('active_data_files_details'):
                    for path, details in debug_info['active_data_files_details'].items():
                        print(f"     - {os.path.basename(path)}: {details['rows']} rows")
                else:
                    print(f"   ❌ No active data files details!")
                    
                    # Check what files are stored in the conversation
                    if debug_info.get('data_files_in_conversation'):
                        print(f"   Files stored in conversation:")
                        for i, file_path in enumerate(debug_info['data_files_in_conversation']):
                            print(f"     {i+1}. {file_path}")
                            print(f"        Exists: {os.path.exists(file_path)}")
            else:
                print(f"❌ Could not get debug info")
                
        else:
            print(f"❌ API conversation creation failed: {api_response.status_code}")
            print(f"Error: {api_response.text}")
            return
            
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return
    
    # Test 3: Compare file path handling
    print(f"\n3. Analyzing file path differences...")
    
    # Check if API gets the same file paths
    try:
        # Get the conversation details from API
        conversation_response = requests.get(f"{base_url}/llm/conversations/{api_conversation_id}")
        if conversation_response.status_code == 200:
            conversation_data = conversation_response.json()
            # Note: API response might not include data_files due to response model
            print(f"   API conversation title: {conversation_data.get('title', 'N/A')}")
            print(f"   API conversation messages: {len(conversation_data.get('messages', []))}")
        
        # Test if API can access the files directly through a specific endpoint
        print(f"\n4. Testing API data access...")
        
        # Try to send a message that should trigger smart processing
        test_message = "How many total transactions are in both files?"
        message_response = requests.post(
            f"{base_url}/llm/conversations/{api_conversation_id}/messages",
            json={"message": test_message}
        )
        
        if message_response.status_code == 200:
            message_result = message_response.json()
            smart_processing = message_result.get('smart_processing', False)
            print(f"   Smart processing triggered: {'✅' if smart_processing else '❌'}")
            
            if smart_processing:
                processed = message_result.get('processed_results', {})
                print(f"   Calculation method: {processed.get('calculation_method', 'unknown')}")
                print(f"   Primary answer: {processed.get('primary_answer', 'N/A')[:100]}...")
            else:
                response_text = message_result.get('response', '')
                asks_for_data = 'please share' in response_text.lower() or 'need more' in response_text.lower()
                print(f"   Response asks for data: {'❌ Bad' if asks_for_data else '✅ Good'}")
                print(f"   Response preview: {response_text[:150]}...")
                
        else:
            print(f"❌ Message test failed: {message_response.status_code}")
            
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")

if __name__ == "__main__":
    test_api_vs_direct() 