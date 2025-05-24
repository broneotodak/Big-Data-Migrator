"""
Final test to verify multi-LLM consensus and smart processing issues
"""
import requests
import json
import os
import time

def test_multi_llm_consensus():
    """Test multi-LLM consensus specifically to debug the None response issue"""
    
    base_url = "http://localhost:8000"
    
    print("🔍 Testing Multi-LLM Consensus Issue")
    print("=" * 60)
    
    # File paths  
    file1_path = os.path.abspath("temp/MMSDO_P_202412_EP810177.csv")
    file2_path = os.path.abspath("temp/Payments by order - 2024-12-01 - 2024-12-31.csv")
    
    print(f"Files to test:")
    print(f"  📄 {os.path.basename(file1_path)} {'✅' if os.path.exists(file1_path) else '❌'}")
    print(f"  📄 {os.path.basename(file2_path)} {'✅' if os.path.exists(file2_path) else '❌'}")
    
    # Check API health
    print(f"\n1. Checking API health...")
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200:
            print(f"✅ API server is running")
        else:
            print(f"❌ API server issue: {health_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API server: {str(e)}")
        return
    
    # Create conversation with files
    print(f"\n2. Creating conversation with both files...")
    response = requests.post(f"{base_url}/llm/conversations", 
                           json={
                               "title": "Multi-LLM Consensus Test",
                               "data_files": [file1_path, file2_path]
                           })
    
    if response.status_code != 200:
        print(f"❌ Failed to create conversation: {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    conversation_data = response.json()
    conversation_id = conversation_data["conversation_id"]
    print(f"✅ Created conversation: {conversation_id}")
    
    # Wait for data context to build
    time.sleep(2)
    
    # Test single LLM first (should work)
    print(f"\n3. Testing single LLM processing...")
    test_query = "Can you understand both files?"
    
    single_response = requests.post(
        f"{base_url}/llm/conversations/{conversation_id}/messages",
        json={"message": test_query}
    )
    
    if single_response.status_code == 200:
        single_result = single_response.json()
        print(f"✅ Single LLM successful")
        print(f"   Smart processing: {single_result.get('smart_processing', 'Not specified')}")
        print(f"   Response length: {len(single_result.get('response', ''))}")
        
        # Show first part of response
        response_preview = single_result.get('response', '')[:200]
        print(f"   Response preview: {response_preview}...")
    else:
        print(f"❌ Single LLM failed: {single_response.status_code}")
        print(f"Error: {single_response.text}")
    
    # Test multi-LLM processing (the issue)
    print(f"\n4. Testing multi-LLM processing...")
    
    multi_response = requests.post(
        f"{base_url}/llm/conversations/{conversation_id}/messages/multi",
        json={"message": test_query}
    )
    
    if multi_response.status_code == 200:
        multi_result = multi_response.json()
        print(f"✅ Multi-LLM endpoint successful")
        
        print(f"\n📊 Multi-LLM Results:")
        print(f"   Mode: {multi_result.get('mode', 'unknown')}")
        print(f"   Providers used: {multi_result.get('providers_used', 0)}")
        print(f"   Successful responses: {multi_result.get('successful_responses', 0)}")
        
        # Check best response
        best_response = multi_result.get('best_response', '')
        print(f"   Best response length: {len(best_response)}")
        if best_response:
            print(f"   Best response preview: {best_response[:200]}...")
        else:
            print(f"   ❌ Best response is empty!")
        
        # Check consensus response  
        consensus_response = multi_result.get('consensus_response', '')
        print(f"   Consensus response: {'✅ Exists' if consensus_response else '❌ None/Empty'}")
        if consensus_response:
            print(f"   Consensus preview: {consensus_response[:200]}...")
        else:
            print(f"   ⚠️  This is likely the issue - consensus is None/empty")
        
        # Check individual responses
        all_responses = multi_result.get('all_responses', [])
        print(f"\n🔍 Individual Provider Analysis:")
        for i, resp in enumerate(all_responses, 1):
            provider = resp.get('provider', 'unknown')
            success = resp.get('success', False)
            response_length = len(resp.get('response', ''))
            error = resp.get('error', 'None')
            
            print(f"   {i}. {provider}: {'✅' if success else '❌'} Success")
            print(f"      Response length: {response_length}")
            if not success and error != 'None':
                print(f"      Error: {error}")
            elif success and response_length > 0:
                preview = resp.get('response', '')[:100]
                print(f"      Preview: {preview}...")
        
        # Analyze the issue
        successful_responses = [r for r in all_responses if r.get('success', False)]
        failed_responses = [r for r in all_responses if not r.get('success', False)]
        
        print(f"\n🔧 Issue Analysis:")
        print(f"   Total responses: {len(all_responses)}")
        print(f"   Successful: {len(successful_responses)}")
        print(f"   Failed: {len(failed_responses)}")
        
        if len(successful_responses) > 0 and not consensus_response:
            print(f"   💡 ISSUE IDENTIFIED: We have successful responses but no consensus!")
            print(f"   This suggests a problem in the consensus generation logic.")
        elif len(successful_responses) == 0:
            print(f"   💡 ISSUE: No successful responses from any provider")
            for resp in failed_responses:
                provider = resp.get('provider', 'unknown')
                error = resp.get('error', 'Unknown error')
                print(f"      {provider} failed: {error}")
        
    else:
        print(f"❌ Multi-LLM failed: {multi_response.status_code}")
        print(f"Error: {multi_response.text}")
    
    # Check recent errors
    print(f"\n5. Checking recent errors...")
    try:
        errors_response = requests.get(f"{base_url}/debug/recent-errors")
        if errors_response.status_code == 200:
            recent_errors = errors_response.json()
            if recent_errors:
                print(f"⚠️  Found {len(recent_errors)} recent errors:")
                for error in recent_errors[-3:]:  # Show last 3 errors
                    print(f"   - {error.get('type', 'unknown')}: {error.get('message', 'No message')}")
            else:
                print(f"✅ No recent errors")
        else:
            print(f"⚠️  Could not fetch recent errors")
    except Exception as e:
        print(f"⚠️  Error checking recent errors: {str(e)}")

if __name__ == "__main__":
    test_multi_llm_consensus() 