"""
Simple CLI tool to demonstrate the LLM conversation system
"""
import os
import sys
import json
import argparse
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from app.llm.conversation_system import LLMConversationSystem
from app.llm.online_llm_fallback import OnlineLLMConfig
from app.memory.memory_monitor import MemoryMonitor
from app.memory.resource_optimizer import ResourceOptimizer

def main():
    """Main function for LLM conversation CLI."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Big Data Migrator LLM Conversation CLI")
    parser.add_argument("--data", "-d", nargs="+", help="Paths to data files to analyze")
    parser.add_argument("--title", "-t", default="Data Conversation", help="Conversation title")
    parser.add_argument("--model", "-m", default=None, help="Local LLM model name")
    parser.add_argument("--online", "-o", action="store_true", help="Enable online LLM fallback")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv(os.path.join("config", ".env"))
    
    # Initialize memory management
    memory_monitor = MemoryMonitor()
    resource_optimizer = ResourceOptimizer(memory_monitor)
    
    # Configure online LLM fallback if enabled
    online_llm_config = None
    if args.online or os.getenv("ENABLE_ONLINE_FALLBACK", "false").lower() == "true":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not set. Set in environment or .env file.")
            sys.exit(1)
        online_llm_config = OnlineLLMConfig(
            api_key=api_key,
            model=os.getenv("ONLINE_LLM_MODEL", "gpt-4o")
        )
    
    # Initialize LLM conversation system
    llm_system = LLMConversationSystem(
        local_llm_url=os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1"),
        local_llm_model=args.model or os.getenv("LOCAL_LLM_MODEL", "CodeLlama-34B-Instruct"),
        memory_monitor=memory_monitor,
        resource_optimizer=resource_optimizer,
        online_llm_config=online_llm_config,
        enable_online_fallback=bool(online_llm_config)
    )
    
    # Check LLM connection
    print(f"Connecting to {args.model or os.getenv('LOCAL_LLM_MODEL', 'CodeLlama-34B-Instruct')} at {os.getenv('LOCAL_LLM_URL', 'http://localhost:1234/v1')}...")
    llm_system.llm_client.check_connection()
    
    # Create a conversation
    print(f"Creating conversation: {args.title}")
    conversation_id = llm_system.create_conversation(
        title=args.title,
        data_files=args.data
    )
    print(f"Conversation created with ID: {conversation_id}")
    
    # If data files provided, build context
    if args.data:
        print(f"Building data context for {len(args.data)} files...")
        for file in args.data:
            print(f"  - {file}")
        
        data_context = llm_system.build_data_context(args.data, conversation_id)
        print("Data context built successfully")
    
    # Interactive conversation loop
    print("\nEntering conversation mode. Type 'exit' to quit, 'guidance' for suggestions, or 'help' for commands.")
    while True:
        try:
            # Get user input
            user_message = input("\n> ")
            
            # Process commands
            if user_message.lower() == 'exit':
                break
                
            elif user_message.lower() == 'help':
                print("\nAvailable commands:")
                print("  help      - Show this help message")
                print("  exit      - Exit the conversation")
                print("  guidance  - Get data exploration suggestions")
                print("  optimize  - Run schema optimization (requires online LLM)")
                print("  clear     - Clear the screen")
                print("  Any other input will be sent as a message to the LLM")
                continue
                
            elif user_message.lower() == 'guidance':
                print("\nGenerating guidance...")
                guidance = llm_system.generate_guidance(conversation_id)
                
                # Print suggestions
                if guidance.get("suggestions"):
                    print("\n--- Suggestions ---")
                    for i, suggestion in enumerate(guidance["suggestions"], 1):
                        print(f"{i}. {suggestion['content']}")
                
                # Print questions
                if guidance.get("questions"):
                    print("\n--- Questions ---")
                    for i, question in enumerate(guidance["questions"], 1):
                        print(f"{i}. {question['content']}")
                
                # Print improvements
                if guidance.get("improvements"):
                    print("\n--- Improvements ---")
                    for i, improvement in enumerate(guidance["improvements"], 1):
                        print(f"{i}. {improvement['content']}")
                continue
                
            elif user_message.lower() == 'optimize':
                if not llm_system.enable_online_fallback:
                    print("\nError: Online LLM fallback not enabled. Use --online flag or set ENABLE_ONLINE_FALLBACK=true")
                    continue
                
                print("\nStarting schema optimization with online LLM...")
                llm_system.optimize_schema_with_fallback(conversation_id)
                print("Optimization started in background. Results will be added to the conversation.")
                continue
                
            elif user_message.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            # Normal message - get response
            print("\nThinking...")
            response = llm_system.add_message(user_message, conversation_id)
            print(f"\n{response['response']}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
