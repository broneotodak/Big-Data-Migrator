"""
Test script to debug smart processing using the debug endpoint
"""
import requests
import json
import os

def test_smart_processing_debug():
    """Test smart processing with debug information"""
    
    base_url = "http://localhost:8000"
    
    print("🔍 Smart Processing Debug Test")
    print("=" * 50)
    
    # Create conversation with files
    file1_path = os.path.abspath("temp/MMSDO_P_202412_EP810177.csv")
    file2_path = os.path.abspath("temp/Payments by order - 2024-12-01 - 2024-12-31.csv")
    
    print(f"1. Creating conversation with files...")
    response = requests.post(f"{base_url}/llm/conversations", 
                           json={
                               "title": "Smart Debug Test",
                               "data_files": [file1_path, file2_path]
                           })
    
    if response.status_code != 200:
        print(f"❌ Failed to create conversation: {response.status_code}")
        return
    
    conversation_data = response.json()
    conversation_id = conversation_data["conversation_id"]
    print(f"✅ Created conversation: {conversation_id}")
    
    # Get debug information
    print(f"\n2. Getting debug information...")
    debug_response = requests.get(f"{base_url}/debug/conversation-debug/{conversation_id}")
    
    if debug_response.status_code == 200:
        debug_info = debug_response.json()
        print(f"✅ Debug info retrieved")
        
        print(f"\n📊 Conversation Status:")
        print(f"   Title: {debug_info.get('title', 'N/A')}")
        print(f"   Data files in conversation: {debug_info.get('data_files_count', 0)}")
        if debug_info.get('data_files_in_conversation'):
            for i, file_path in enumerate(debug_info['data_files_in_conversation'][:2]):
                print(f"     {i+1}. {os.path.basename(file_path)}")
        
        print(f"\n🧠 Smart Processing Status:")
        print(f"   Enabled: {debug_info.get('smart_processing_enabled', False)}")
        print(f"   Processor exists: {debug_info.get('smart_processor_exists', False)}")
        print(f"   Active data files exist: {debug_info.get('active_data_files_exists', False)}")
        print(f"   Active data files count: {debug_info.get('active_data_files_count', 0)}")
        print(f"   Active data context exists: {debug_info.get('active_data_context_exists', False)}")
        
        print(f"\n🔄 Conversation Context:")
        print(f"   Active conversation ID: {debug_info.get('active_conversation_id', 'None')}")
        print(f"   Is active conversation: {debug_info.get('is_active_conversation', False)}")
        
        # Check data files details
        if debug_info.get('active_data_files_details'):
            print(f"\n📁 Active Data Files Details:")
            for file_path, details in debug_info['active_data_files_details'].items():
                print(f"   📄 {os.path.basename(file_path)}:")
                print(f"      Rows: {details['rows']}, Columns: {details['columns']}")
                print(f"      Column names: {details['column_names'][:3]}..." if len(details['column_names']) > 3 else f"      Column names: {details['column_names']}")
        
        # Identify the issue
        print(f"\n🔧 Smart Processing Analysis:")
        if not debug_info.get('smart_processing_enabled'):
            print(f"   ❌ Smart processing is disabled")
        elif not debug_info.get('smart_processor_exists'):
            print(f"   ❌ Smart processor not initialized")
        elif not debug_info.get('active_data_files_exists'):
            print(f"   ❌ No active data files dictionary")
        elif debug_info.get('active_data_files_count', 0) == 0:
            print(f"   ❌ Active data files dictionary is empty")
        else:
            print(f"   ✅ All smart processing conditions should be met!")
            print(f"   💡 Issue might be in the query processing or file loading")
    else:
        print(f"❌ Failed to get debug info: {debug_response.status_code}")
        return
    
    # Test a comparison query
    print(f"\n3. Testing comparison query...")
    test_query = "Can you compare the transaction amounts between these files?"
    
    response = requests.post(
        f"{base_url}/llm/conversations/{conversation_id}/messages",
        json={"message": test_query}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Query successful")
        print(f"   Smart processing used: {result.get('smart_processing', 'Not specified')}")
        
        if result.get('smart_processing'):
            print(f"🎉 SMART PROCESSING WORKING!")
            processed = result.get('processed_results', {})
            print(f"   Method: {processed.get('calculation_method', 'unknown')}")
            print(f"   Primary answer: {processed.get('primary_answer', 'N/A')[:100]}...")
        else:
            print(f"⚠️  Smart processing not triggered")
            
            # Show response content for analysis
            response_text = result.get('response', '')
            if 'excel' in response_text.lower() or 'pandas' in response_text.lower():
                print(f"   ❌ Response suggests external tools instead of direct calculation")
            else:
                print(f"   📝 Response seems to contain direct analysis")
    else:
        print(f"❌ Query failed: {response.status_code}")

if __name__ == "__main__":
    test_smart_processing_debug() 