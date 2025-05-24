"""
Test script to verify Smart Query Processor functionality
"""
import requests
import json
import time

def test_smart_processor():
    """Test the Smart Query Processor with sample data files"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Smart Query Processor")
    print("=" * 50)
    
    # Create a new conversation
    print("1. Creating new conversation...")
    response = requests.post(f"{base_url}/llm/conversations", 
                           json={"title": "Smart Processor Test"})
    
    if response.status_code != 200:
        print(f"❌ Failed to create conversation: {response.status_code}")
        return
    
    conversation_data = response.json()
    conversation_id = conversation_data["conversation_id"]
    print(f"✅ Created conversation: {conversation_id}")
    
    # Test query that should trigger smart processing
    test_query = "Can you check how much is missing by comparing those 2 files? I want to know the count and sum of RM missing"
    
    print(f"\n2. Sending test query: '{test_query}'")
    print("⏳ Processing (this should use Smart Query Processor)...")
    
    start_time = time.time()
    
    # Send the message
    response = requests.post(
        f"{base_url}/llm/conversations/{conversation_id}/messages/multi",
        json={"message": test_query}
    )
    
    processing_time = time.time() - start_time
    
    if response.status_code != 200:
        print(f"❌ Failed to send message: {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    result = response.json()
    
    print(f"\n3. Results (processed in {processing_time:.2f}s):")
    print("-" * 30)
    
    # Check if smart processing was used
    if "smart_processing" in result and result.get("smart_processing"):
        print("✅ SMART PROCESSING USED!")
        print(f"   Calculation method: {result.get('processed_results', {}).get('calculation_method', 'unknown')}")
        print(f"   Primary answer: {result.get('processed_results', {}).get('primary_answer', 'N/A')}")
    else:
        print("⚠️  Normal LLM processing used")
    
    # Check the response content
    response_text = result.get("best_response", result.get("response", "No response"))
    print(f"\n4. LLM Response:")
    print("-" * 20)
    print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
    
    # Check for Excel suggestions (should NOT appear)
    excel_indicators = ["excel", "spreadsheet", "vlookup", "countif", "sumif"]
    excel_found = any(indicator.lower() in response_text.lower() for indicator in excel_indicators)
    
    if excel_found:
        print("\n❌ ISSUE: Response still contains Excel suggestions!")
        for indicator in excel_indicators:
            if indicator.lower() in response_text.lower():
                print(f"   Found: '{indicator}'")
    else:
        print("\n✅ SUCCESS: No Excel suggestions found!")
    
    # Check for direct calculations
    calculation_indicators = ["RM", "transactions", "total", "difference", "missing"]
    calculations_found = sum(1 for indicator in calculation_indicators 
                           if indicator.lower() in response_text.lower())
    
    print(f"\n5. Analysis Quality:")
    print(f"   Calculation indicators found: {calculations_found}/{len(calculation_indicators)}")
    
    if calculations_found >= 3:
        print("✅ Response contains specific calculations")
    else:
        print("⚠️  Response may lack specific calculations")
    
    return {
        "smart_processing_used": result.get("smart_processing", False),
        "processing_time": processing_time,
        "excel_suggestions": excel_found,
        "calculation_quality": calculations_found >= 3,
        "response_length": len(response_text)
    }

def test_system_status():
    """Test system status endpoints"""
    base_url = "http://localhost:8000"
    
    print("\n🔍 System Status Check")
    print("=" * 30)
    
    endpoints = [
        "/health",
        "/llm/status", 
        "/memory-status",
        "/debug/system-performance"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint}: OK")
            else:
                print(f"⚠️  {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: {str(e)}")

if __name__ == "__main__":
    try:
        # Test system status first
        test_system_status()
        
        # Test smart processor
        result = test_smart_processor()
        
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        if result:
            print(f"Smart Processing: {'✅ ENABLED' if result['smart_processing_used'] else '⚠️  DISABLED'}")
            print(f"Processing Time: {result['processing_time']:.2f}s")
            print(f"Excel Suggestions: {'❌ FOUND' if result['excel_suggestions'] else '✅ NONE'}")
            print(f"Calculation Quality: {'✅ GOOD' if result['calculation_quality'] else '⚠️  POOR'}")
            
            # Overall assessment
            if (result['smart_processing_used'] and 
                not result['excel_suggestions'] and 
                result['calculation_quality']):
                print("\n🎉 OVERALL: SUCCESS - Smart Query Processor working correctly!")
            else:
                print("\n⚠️  OVERALL: NEEDS IMPROVEMENT")
        else:
            print("❌ Test failed to complete")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 