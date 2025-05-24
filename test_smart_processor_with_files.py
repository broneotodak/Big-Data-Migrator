"""
Enhanced test script to verify Smart Query Processor with actual file uploads
"""
import requests
import json
import time
import os

def test_smart_processor_with_files():
    """Test the Smart Query Processor with actual file uploads"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Smart Query Processor with File Uploads")
    print("=" * 60)
    
    # File paths
    file1_path = "temp/MMSDO_P_202412_EP810177.csv"
    file2_path = "temp/Payments by order - 2024-12-01 - 2024-12-31.csv"
    
    # Check if files exist
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print(f"❌ Files not found:")
        print(f"   {file1_path}: {'✅' if os.path.exists(file1_path) else '❌'}")
        print(f"   {file2_path}: {'✅' if os.path.exists(file2_path) else '❌'}")
        return
    
    print(f"✅ Found data files:")
    print(f"   📄 {file1_path}")
    print(f"   📄 {file2_path}")
    
    # Create a new conversation
    print(f"\n1. Creating new conversation...")
    response = requests.post(f"{base_url}/llm/conversations", 
                           json={
                               "title": "Smart Processor Test with Files",
                               "data_files": [file1_path, file2_path]  # Include files in conversation
                           })
    
    if response.status_code != 200:
        print(f"❌ Failed to create conversation: {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    conversation_data = response.json()
    conversation_id = conversation_data["conversation_id"]
    print(f"✅ Created conversation: {conversation_id}")
    
    # Upload files to the conversation (simulate file upload)
    print(f"\n2. Building data context with files...")
    try:
        # The conversation system should automatically build context for the provided files
        # Let's verify the conversation has data context
        time.sleep(2)  # Allow time for context building
        print("✅ Data context should be built automatically")
    except Exception as e:
        print(f"⚠️  Context building issue: {str(e)}")
    
    # Test query that should trigger smart processing
    test_query = "Can you check how much is missing by comparing those 2 files? I want to know the count and sum of RM missing"
    
    print(f"\n3. Sending test query: '{test_query}'")
    print("⏳ Processing (this should use Smart Query Processor with uploaded files)...")
    
    start_time = time.time()
    
    # Send the message - use single LLM endpoint to test smart processing integration
    response = requests.post(
        f"{base_url}/llm/conversations/{conversation_id}/messages",
        json={"message": test_query}
    )
    
    processing_time = time.time() - start_time
    
    if response.status_code != 200:
        print(f"❌ Failed to send message: {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    result = response.json()
    
    print(f"\n4. Results (processed in {processing_time:.2f}s):")
    print("-" * 40)
    
    # Check if smart processing was used
    if "smart_processing" in result and result.get("smart_processing"):
        print("🎉 SMART PROCESSING USED!")
        if "processed_results" in result:
            processed = result["processed_results"]
            print(f"   ✅ Calculation method: {processed.get('calculation_method', 'unknown')}")
            print(f"   📊 Primary answer: {processed.get('primary_answer', 'N/A')[:100]}...")
        else:
            print("   ⚠️  No processed results details")
    else:
        print("⚠️  Normal LLM processing used")
        # Check if it's because files weren't loaded
        if "No data files available" in result.get("response", ""):
            print("   💡 Likely cause: Data files not loaded properly")
    
    # Check the response content
    response_text = result.get("response", "No response")
    print(f"\n5. LLM Response:")
    print("-" * 25)
    print(response_text[:800] + "..." if len(response_text) > 800 else response_text)
    
    # Check for Excel suggestions (should NOT appear if smart processing works)
    excel_indicators = ["excel", "spreadsheet", "vlookup", "countif", "sumif", "command-line", "diff", "awk"]
    excel_found = any(indicator.lower() in response_text.lower() for indicator in excel_indicators)
    
    if excel_found:
        print("\n❌ ISSUE: Response contains tool suggestions!")
        for indicator in excel_indicators:
            if indicator.lower() in response_text.lower():
                print(f"   Found: '{indicator}'")
    else:
        print("\n✅ SUCCESS: No external tool suggestions found!")
    
    # Check for direct calculations and specific numbers
    calculation_indicators = ["RM", "transactions", "total", "difference", "missing"]
    calculations_found = sum(1 for indicator in calculation_indicators 
                           if indicator.lower() in response_text.lower())
    
    # Look for specific numbers that indicate actual processing
    has_specific_numbers = any(pattern in response_text for pattern in [
        "RM 4,737", "RM 25,684", "34 transactions", "124 transactions"
    ])
    
    print(f"\n6. Analysis Quality:")
    print(f"   📊 Calculation indicators: {calculations_found}/{len(calculation_indicators)}")
    print(f"   🔢 Specific numbers found: {'✅' if has_specific_numbers else '❌'}")
    
    # Overall assessment
    overall_success = (
        result.get("smart_processing", False) and 
        not excel_found and 
        calculations_found >= 3 and
        has_specific_numbers
    )
    
    return {
        "smart_processing_used": result.get("smart_processing", False),
        "processing_time": processing_time,
        "external_tool_suggestions": excel_found,
        "calculation_quality": calculations_found >= 3,
        "specific_numbers": has_specific_numbers,
        "response_length": len(response_text),
        "overall_success": overall_success
    }

def test_individual_file_processing():
    """Test processing individual files to verify Smart Query Processor can load them"""
    print("\n🔍 Testing Individual File Processing")
    print("=" * 40)
    
    try:
        import pandas as pd
        from app.processors.smart_query_processor import SmartQueryProcessor
        
        # Initialize the processor
        processor = SmartQueryProcessor()
        
        # Load files manually
        file1_path = "temp/MMSDO_P_202412_EP810177.csv"
        file2_path = "temp/Payments by order - 2024-12-01 - 2024-12-31.csv"
        
        data_files = {}
        
        # Load file 1
        if os.path.exists(file1_path):
            df1 = pd.read_csv(file1_path)
            data_files[file1_path] = df1
            print(f"✅ Loaded {file1_path}: {len(df1)} rows, {len(df1.columns)} columns")
        else:
            print(f"❌ File not found: {file1_path}")
            
        # Load file 2
        if os.path.exists(file2_path):
            df2 = pd.read_csv(file2_path)
            data_files[file2_path] = df2
            print(f"✅ Loaded {file2_path}: {len(df2)} rows, {len(df2.columns)} columns")
        else:
            print(f"❌ File not found: {file2_path}")
        
        if len(data_files) >= 2:
            # Test the smart processor directly
            test_query = "How much is missing by comparing those 2 files?"
            
            print(f"\n🧮 Testing Smart Processor directly...")
            results = processor.process_query(test_query, data_files)
            
            print(f"✅ Direct processing successful!")
            print(f"   Method: {results.calculation_method}")
            print(f"   Primary Answer: {results.primary_answer}")
            print(f"   Summary: {results.formatted_summary[:200]}...")
            
            return True
        else:
            print("❌ Insufficient files loaded for testing")
            return False
            
    except Exception as e:
        print(f"❌ Error in direct processing test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        print("🚀 Starting Enhanced Smart Query Processor Test")
        print("=" * 70)
        
        # Test direct file processing first
        direct_test_success = test_individual_file_processing()
        
        # Test full system integration
        result = test_smart_processor_with_files()
        
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 70)
        
        print(f"Direct Processor Test: {'✅ PASSED' if direct_test_success else '❌ FAILED'}")
        
        if result:
            print(f"Smart Processing Used: {'✅ YES' if result['smart_processing_used'] else '⚠️  NO'}")
            print(f"Processing Time: {result['processing_time']:.2f}s")
            print(f"External Tool Suggestions: {'❌ FOUND' if result['external_tool_suggestions'] else '✅ NONE'}")
            print(f"Calculation Quality: {'✅ GOOD' if result['calculation_quality'] else '⚠️  POOR'}")
            print(f"Specific Numbers: {'✅ YES' if result['specific_numbers'] else '❌ NO'}")
            print(f"Overall Success: {'🎉 YES' if result['overall_success'] else '⚠️  NO'}")
            
            if result['overall_success']:
                print("\n🎉 OVERALL: SUCCESS - Smart Query Processor is working perfectly!")
                print("   The system performs direct calculations without suggesting external tools.")
            elif result['smart_processing_used']:
                print("\n⚠️  OVERALL: PARTIAL SUCCESS - Smart processing used but needs refinement")
            else:
                print("\n⚠️  OVERALL: INTEGRATION ISSUE - Smart processor not being triggered")
                print("   Recommendation: Check file upload and data context building process")
        else:
            print("❌ Integration test failed to complete")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 