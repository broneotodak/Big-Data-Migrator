"""
Debug the build_data_context method to see why files aren't loading
"""
import os
import sys
sys.path.append('.')

import pandas as pd
from app.llm.conversation_system import LLMConversationSystem
from app.memory.memory_monitor import MemoryMonitor

def test_data_context_building():
    """Test data context building directly"""
    
    print("🔍 Testing Data Context Building")
    print("=" * 50)
    
    # File paths
    file1_path = os.path.abspath("temp/MMSDO_P_202412_EP810177.csv")
    file2_path = os.path.abspath("temp/Payments by order - 2024-12-01 - 2024-12-31.csv")
    
    print(f"File paths:")
    print(f"  📄 {file1_path}")
    print(f"  📄 {file2_path}")
    print(f"  Exists: {os.path.exists(file1_path)} | {os.path.exists(file2_path)}")
    
    # Test direct pandas loading first
    print(f"\n1. Testing direct pandas loading...")
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        print(f"✅ Direct pandas loading successful")
        print(f"   File 1: {len(df1)} rows, {len(df1.columns)} columns")
        print(f"   File 2: {len(df2)} rows, {len(df2.columns)} columns")
    except Exception as e:
        print(f"❌ Direct pandas loading failed: {str(e)}")
        return
    
    # Test conversation system initialization
    print(f"\n2. Testing conversation system initialization...")
    try:
        memory_monitor = MemoryMonitor()
        llm_system = LLMConversationSystem(
            memory_monitor=memory_monitor,
            enable_smart_processing=True
        )
        print(f"✅ LLM system initialized")
        print(f"   Smart processing enabled: {llm_system.enable_smart_processing}")
        print(f"   Smart processor exists: {llm_system.smart_processor is not None}")
        print(f"   Initial active data files: {len(llm_system._active_data_files) if llm_system._active_data_files else 0}")
    except Exception as e:
        print(f"❌ LLM system initialization failed: {str(e)}")
        return
    
    # Test conversation creation
    print(f"\n3. Testing conversation creation...")
    try:
        conversation_id = llm_system.create_conversation(
            title="Data Context Debug Test",
            data_files=[file1_path, file2_path]
        )
        print(f"✅ Conversation created: {conversation_id}")
        
        # Check if active conversation is set
        print(f"   Active conversation ID: {llm_system.conversation_manager.active_conversation_id}")
        print(f"   Matches created ID: {llm_system.conversation_manager.active_conversation_id == conversation_id}")
        
    except Exception as e:
        print(f"❌ Conversation creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Check data loading results
    print(f"\n4. Checking data loading results...")
    print(f"   Active data files count: {len(llm_system._active_data_files) if llm_system._active_data_files else 0}")
    print(f"   Active data context exists: {llm_system._active_data_context is not None}")
    
    if llm_system._active_data_files:
        print(f"   ✅ Data files loaded successfully:")
        for file_path, df in llm_system._active_data_files.items():
            print(f"     - {os.path.basename(file_path)}: {len(df)} rows, {len(df.columns)} columns")
    else:
        print(f"   ❌ No data files in _active_data_files!")
        
        # Check if conversation has the files
        conversation = llm_system.conversation_manager.get_conversation(conversation_id)
        if conversation:
            print(f"   Conversation data files: {len(conversation.data_files)}")
            for i, file_path in enumerate(conversation.data_files):
                print(f"     {i+1}. {file_path}")
                print(f"        Exists: {os.path.exists(file_path)}")
        
        # Try manual build_data_context
        print(f"\n5. Testing manual build_data_context...")
        try:
            context = llm_system.build_data_context([file1_path, file2_path], conversation_id)
            print(f"✅ Manual build_data_context completed")
            print(f"   Context keys: {list(context.keys()) if context else 'None'}")
            print(f"   Active data files after manual build: {len(llm_system._active_data_files) if llm_system._active_data_files else 0}")
            
            if llm_system._active_data_files:
                print(f"   ✅ Files loaded in manual build:")
                for file_path, df in llm_system._active_data_files.items():
                    print(f"     - {os.path.basename(file_path)}: {len(df)} rows")
            else:
                print(f"   ❌ Manual build also failed to load files!")
                
        except Exception as e:
            print(f"❌ Manual build_data_context failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_data_context_building() 