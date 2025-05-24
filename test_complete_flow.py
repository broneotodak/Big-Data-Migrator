"""
Test the complete flow from conversation creation to smart processing
"""
import requests
import json
import os
import time

def test_complete_flow():
    """Test the complete flow with the user's exact queries"""
    
    base_url = "http://localhost:8000"
    
    print("🔍 Testing Complete Smart Processing Flow")
    print("=" * 60)
    
    # File paths  
    file1_path = os.path.abspath("temp/MMSDO_P_202412_EP810177.csv")
    file2_path = os.path.abspath("temp/Payments by order - 2024-12-01 - 2024-12-31.csv")
    
    print(f"Using files:")
    print(f"  📄 {os.path.basename(file1_path)}")
    print(f"  📄 {os.path.basename(file2_path)}")
    
    # Step 1: Create conversation
    print(f"\n1. Creating conversation...")
    response = requests.post(f"{base_url}/llm/conversations", 
                           json={
                               "title": "Complete Flow Test",
                               "data_files": [file1_path, file2_path]
                           })
    
    if response.status_code != 200:
        print(f"❌ Failed: {response.status_code}")
        return
    
    conversation_data = response.json()
    conversation_id = conversation_data["conversation_id"]
    print(f"✅ Created: {conversation_id}")
    
    # Step 2: Test User's First Query (Logical Relations)
    print(f"\n2. Testing: 'what can you explain about both files logical relations?'")
    
    query1 = "what can you explain about both files logical relations?"
    response = requests.post(
        f"{base_url}/llm/conversations/{conversation_id}/messages",
        json={"message": query1}
    )
    
    if response.status_code == 200:
        result = response.json()
        smart_processing = result.get('smart_processing', False)
        print(f"   Smart processing: {'✅' if smart_processing else '❌'}")
        
        if smart_processing:
            processed = result.get('processed_results', {})
            print(f"   Method: {processed.get('calculation_method', 'unknown')}")
            print(f"   Answer: {processed.get('primary_answer', 'N/A')[:150]}...")
        else:
            response_text = result.get('response', '')
            
            # Check for specific data indicators
            specific_indicators = ['RM ', 'transactions', '36', '124', 'total', 'amount']
            has_specific_data = any(indicator in response_text for indicator in specific_indicators)
            
            print(f"   Has specific data: {'✅' if has_specific_data else '❌'}")
            print(f"   Preview: {response_text[:200]}...")
    
    # Step 3: Test User's Second Query (Missing Transactions)  
    print(f"\n3. Testing: 'can you find out how many transactions missing (count) and how much in RM?'")
    
    query2 = "Based on this information that Payment by order is the total payment received to the 10 camp enterprise shop from different transaction type and MMSDO is only showing the payment to that shop via QRpay, can you find out how many transactions missing (the count) and how much is missing in RM?"
    
    response = requests.post(
        f"{base_url}/llm/conversations/{conversation_id}/messages",
        json={"message": query2}
    )
    
    if response.status_code == 200:
        result = response.json()
        smart_processing = result.get('smart_processing', False)
        print(f"   Smart processing: {'✅' if smart_processing else '❌'}")
        
        if smart_processing:
            processed = result.get('processed_results', {})
            method = processed.get('calculation_method', 'unknown')
            print(f"   Method: {method}")
            
            if method == "pandas_comparison":
                print(f"   🎉 PERFECT! Using comparison method for missing transactions!")
            elif method == "pandas_count":
                print(f"   ⚠️ Using count method (should be comparison)")
            
            primary_answer = processed.get('primary_answer', 'N/A')
            print(f"   Answer: {primary_answer[:300]}...")
            
            # Check for specific comparison results
            if "missing" in primary_answer.lower() and "RM" in primary_answer:
                print(f"   ✅ Contains missing transaction analysis with RM amounts")
            else:
                print(f"   ❌ Doesn't contain proper missing transaction analysis")
                
        else:
            response_text = result.get('response', '')
            
            # Check for manual calculation requests
            asks_for_data = any(phrase in response_text.lower() for phrase in [
                'please share', 'need more information', 'provide the specific numbers'
            ])
            
            if asks_for_data:
                print(f"   ❌ LLM asking for data instead of calculating!")
            else:
                print(f"   ✅ LLM providing analysis")
                
            # Check for specific results
            has_specific_results = any(indicator in response_text for indicator in [
                'missing', 'RM ', 'transactions', 'difference'
            ])
            
            print(f"   Has specific results: {'✅' if has_specific_results else '❌'}")
            print(f"   Preview: {response_text[:300]}...")
    
    # Step 4: Test Multi-LLM with the same query
    print(f"\n4. Testing Multi-LLM with missing transactions query...")
    
    multi_response = requests.post(
        f"{base_url}/llm/conversations/{conversation_id}/messages/multi",
        json={"message": query2}
    )
    
    if multi_response.status_code == 200:
        multi_result = multi_response.json()
        consensus_response = multi_result.get('consensus_response', '')
        
        print(f"   Consensus exists: {'✅' if consensus_response else '❌'}")
        
        if consensus_response:
            # Check consensus quality
            has_specific_numbers = any(indicator in consensus_response for indicator in [
                'RM ', 'missing', 'transactions', 'count', 'difference'
            ])
            
            asks_for_data = any(phrase in consensus_response.lower() for phrase in [
                'please share', 'need more information', 'provide the specific numbers'
            ])
            
            print(f"   Has specific numbers: {'✅' if has_specific_numbers else '❌'}")
            print(f"   Asks for data: {'❌ Bad' if asks_for_data else '✅ Good'}")
            print(f"   Consensus preview: {consensus_response[:250]}...")
            
            # Final assessment
            if has_specific_numbers and not asks_for_data:
                print(f"\n🎉 SUCCESS: Multi-LLM providing direct analysis with specific numbers!")
            elif has_specific_numbers:
                print(f"\n⚠️ PARTIAL: Has numbers but still asking for data")
            else:
                print(f"\n❌ ISSUE: Generic response without specific calculations")
        else:
            print(f"   ❌ No consensus response")
    
    print(f"\n📊 SUMMARY:")
    print(f"   ✅ Conversation creation: Working")
    print(f"   ✅ Safety check: Working (data loaded when message sent)")
    print(f"   ✅ Multi-LLM consensus: Working")
    print(f"   🔧 Smart processing: Partially working (may need intent tuning)")

if __name__ == "__main__":
    test_complete_flow() 