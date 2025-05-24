"""
Direct test of conversation system file loading and smart processing
"""
import os
import sys
sys.path.append('.')

from app.llm.conversation_system import LLMConversationSystem
from app.memory.memory_monitor import MemoryMonitor

def test_direct_conversation_system():
    """Test the conversation system directly without API"""
    
    print("🔍 Direct Conversation System Test")
    print("=" * 50)
    
    # Initialize components
    memory_monitor = MemoryMonitor()
    
    # Initialize conversation system
    llm_system = LLMConversationSystem(
        memory_monitor=memory_monitor,
        enable_smart_processing=True
    )
    
    print(f"✅ LLM system initialized")
    print(f"   Smart processing enabled: {llm_system.enable_smart_processing}")
    print(f"   Smart processor available: {llm_system.smart_processor is not None}")
    
    # Test file paths
    file1_path = os.path.abspath("temp/MMSDO_P_202412_EP810177.csv")
    file2_path = os.path.abspath("temp/Payments by order - 2024-12-01 - 2024-12-31.csv")
    
    print(f"\n📁 File paths:")
    print(f"   File 1: {file1_path} {'✅' if os.path.exists(file1_path) else '❌'}")
    print(f"   File 2: {file2_path} {'✅' if os.path.exists(file2_path) else '❌'}")
    
    # Create conversation with data files
    print(f"\n1. Creating conversation with data files...")
    try:
        conversation_id = llm_system.create_conversation(
            title="Direct Test Conversation",
            data_files=[file1_path, file2_path]
        )
        print(f"✅ Conversation created: {conversation_id}")
    except Exception as e:
        print(f"❌ Error creating conversation: {str(e)}")
        return
    
    # Check if active data files were loaded
    print(f"\n2. Checking active data files...")
    print(f"   Active data files dict exists: {llm_system._active_data_files is not None}")
    if llm_system._active_data_files is not None:
        print(f"   Active data files count: {len(llm_system._active_data_files)}")
        if llm_system._active_data_files:
            print(f"   Active data files:")
            for file_path, df in llm_system._active_data_files.items():
                print(f"     - {os.path.basename(file_path)}: {len(df)} rows, {len(df.columns)} columns")
        else:
            print(f"   ❌ No active data files loaded!")
    
    # Check active data context
    print(f"\n3. Checking active data context...")
    print(f"   Active data context exists: {llm_system._active_data_context is not None}")
    if llm_system._active_data_context:
        files_in_context = llm_system._active_data_context.get("files", {})
        print(f"   Files in context: {len(files_in_context)}")
    
    # If files are loaded, test smart processing
    if llm_system._active_data_files and len(llm_system._active_data_files) > 0:
        print(f"\n4. Testing smart processing directly...")
        try:
            test_query = "Compare transaction amounts between these files"
            
            # Test the smart processor directly
            results = llm_system.smart_processor.process_query(test_query, llm_system._active_data_files)
            
            print(f"✅ Smart processor test successful!")
            print(f"   Method: {results.calculation_method}")
            print(f"   Primary answer: {results.primary_answer[:100]}...")
            
            # Test full add_message
            print(f"\n5. Testing full add_message flow...")
            response = llm_system.add_message(test_query, conversation_id)
            
            print(f"✅ Add message successful!")
            print(f"   Smart processing used: {response.get('smart_processing', False)}")
            print(f"   Response length: {len(response.get('response', ''))}")
            
            if response.get('smart_processing'):
                print(f"🎉 SMART PROCESSING WORKING!")
                processed = response.get('processed_results', {})
                print(f"   Calculation method: {processed.get('calculation_method', 'unknown')}")
            else:
                print(f"⚠️  Smart processing not triggered in add_message")
            
        except Exception as e:
            print(f"❌ Error testing smart processing: {str(e)}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"\n❌ Cannot test smart processing - no active data files!")
        
        # Try building data context manually
        print(f"\n🔧 Attempting manual data context building...")
        try:
            llm_system.build_data_context([file1_path, file2_path], conversation_id)
            print(f"✅ Manual data context building complete")
            print(f"   Active data files after manual build: {len(llm_system._active_data_files) if llm_system._active_data_files else 0}")
        except Exception as e:
            print(f"❌ Manual data context building failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_direct_conversation_system() 