"""
Conversation persistence for data discussions.

This module provides functionality to save, load, and manage
chat conversation history with data context.
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import streamlit as st


class ConversationPersistence:
    """
    Manages persistence of data conversations.
    
    This class:
    - Saves chat history with data context
    - Exports decisions and recommendations
    - Allows conversation resumption after breaks 
    - Tracks data preparation progress
    - Creates audit trails for data decisions
    """
    
    def __init__(self, storage_dir: str = "conversations"):
        """
        Initialize conversation persistence.
        
        Args:
            storage_dir: Directory for storing conversations
        """
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize state
        if "saved_conversations" not in st.session_state:
            st.session_state.saved_conversations = []
            
        # Load available conversations
        self._load_available_conversations()
    
    def _load_available_conversations(self):
        """Load list of available conversations."""
        conversations = []
        
        try:
            # Get all JSON files in the storage directory
            json_files = [f for f in os.listdir(self.storage_dir) if f.endswith('.json')]
            
            for file in json_files:
                file_path = os.path.join(self.storage_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Try to read the metadata only
                        data = json.load(f)
                        
                        conversations.append({
                            "id": data.get("id", file.replace(".json", "")),
                            "title": data.get("title", "Untitled Conversation"),
                            "date": data.get("timestamp", "Unknown"),
                            "file_count": len(data.get("data_files", [])),
                            "message_count": len(data.get("messages", [])),
                            "file_path": file_path
                        })
                except Exception as e:
                    # Skip invalid files
                    continue
            
            # Sort by date, newest first
            conversations.sort(key=lambda x: x.get("date", ""), reverse=True)
            
            st.session_state.saved_conversations = conversations
        except Exception as e:
            st.error(f"Error loading conversations: {str(e)}")
    
    def save_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """
        Save a conversation to disk.
        
        Args:
            conversation_data: Conversation data to save
            
        Returns:
            Path to the saved file
        """
        try:
            # Ensure the conversation has an ID and timestamp
            conversation_id = conversation_data.get("id", str(int(time.time())))
            if "id" not in conversation_data:
                conversation_data["id"] = conversation_id
            
            if "timestamp" not in conversation_data:
                conversation_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create a filename
            filename = f"conversation_{conversation_id}.json"
            file_path = os.path.join(self.storage_dir, filename)
            
            # Save the conversation
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            # Reload available conversations
            self._load_available_conversations()
            
            return file_path
        except Exception as e:
            st.error(f"Error saving conversation: {str(e)}")
            return ""
    
    def load_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Load a conversation from disk.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation data
        """
        try:
            # Find the conversation file
            matching_conversations = [c for c in st.session_state.saved_conversations if c["id"] == conversation_id]
            
            if not matching_conversations:
                # Try direct file path
                file_path = os.path.join(self.storage_dir, f"conversation_{conversation_id}.json")
                if not os.path.exists(file_path):
                    return {}
            else:
                file_path = matching_conversations[0]["file_path"]
            
            # Load the conversation
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            return conversation_data
        except Exception as e:
            st.error(f"Error loading conversation: {str(e)}")
            return {}
    
    def export_conversation(self, conversation_id: str, format_type: str = "json") -> str:
        """
        Export a conversation to a specific format.
        
        Args:
            conversation_id: Conversation ID
            format_type: Export format (json, csv, html, or markdown)
            
        Returns:
            Exported content
        """
        try:
            # Load the conversation
            conversation_data = self.load_conversation(conversation_id)
            
            if not conversation_data:
                return ""
            
            if format_type == "json":
                # Already in JSON format
                return json.dumps(conversation_data, indent=2)
            
            elif format_type == "csv":
                # Convert to CSV (messages only)
                messages = conversation_data.get("messages", [])
                df = pd.DataFrame(messages)
                return df.to_csv(index=False)
            
            elif format_type == "html":
                # Generate HTML report
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Conversation {conversation_data.get('title', 'Untitled')}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .message {{ margin-bottom: 15px; padding: 10px; border-radius: 5px; }}
                        .user {{ background-color: #e6f7ff; }}
                        .assistant {{ background-color: #f6f6f6; }}
                        .system {{ background-color: #ffffcc; font-style: italic; }}
                        .metadata {{ font-size: 0.8em; color: #666; }}
                        h1 {{ color: #333; }}
                        h2 {{ color: #666; }}
                    </style>
                </head>
                <body>
                    <h1>{conversation_data.get('title', 'Conversation Export')}</h1>
                    <div class="metadata">
                        <p>Date: {conversation_data.get('timestamp', 'Unknown')}</p>
                        <p>Files: {', '.join(conversation_data.get('data_files', []))}</p>
                    </div>
                    <h2>Messages</h2>
                """
                
                # Add messages
                for msg in conversation_data.get("messages", []):
                    role = msg.get("role", "")
                    content = msg.get("content", "").replace("\n", "<br>")
                    html += f'<div class="message {role}"><strong>{role.capitalize()}:</strong><br>{content}</div>\n'
                
                html += """
                </body>
                </html>
                """
                return html
            
            elif format_type == "markdown":
                # Generate Markdown report
                md = f"# {conversation_data.get('title', 'Conversation Export')}\n\n"
                md += f"Date: {conversation_data.get('timestamp', 'Unknown')}\n\n"
                md += f"Files: {', '.join(conversation_data.get('data_files', []))}\n\n"
                md += "## Messages\n\n"
                
                # Add messages
                for msg in conversation_data.get("messages", []):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    md += f"**{role.capitalize()}:**\n\n{content}\n\n---\n\n"
                
                return md
            
            else:
                return json.dumps(conversation_data, indent=2)
        except Exception as e:
            st.error(f"Error exporting conversation: {str(e)}")
            return ""
    
    def extract_decisions(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Extract key decisions from a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of decision dictionaries
        """
        try:
            # Load the conversation
            conversation_data = self.load_conversation(conversation_id)
            
            if not conversation_data:
                return []
            
            messages = conversation_data.get("messages", [])
            decisions = []
            
            # Look for messages that might contain decisions
            # This is a simple heuristic that looks for specific keywords
            for i, message in enumerate(messages):
                if message.get("role") == "assistant":
                    content = message.get("content", "").lower()
                    
                    # Check for decision indicators
                    decision_indicators = [
                        "recommend", "suggest", "advice", "decision",
                        "should", "could", "would be best", "optimal",
                        "proposed solution", "best approach"
                    ]
                    
                    if any(indicator in content for indicator in decision_indicators):
                        # Find the user question that prompted this
                        question = ""
                        if i > 0 and messages[i-1].get("role") == "user":
                            question = messages[i-1].get("content", "")
                        
                        decisions.append({
                            "type": "recommendation",
                            "question": question,
                            "decision": message.get("content"),
                            "timestamp": message.get("timestamp", "")
                        })
            
            return decisions
        except Exception as e:
            st.error(f"Error extracting decisions: {str(e)}")
            return []
    
    def track_progress(self, conversation_id: str, progress_data: Dict[str, Any]) -> bool:
        """
        Track data preparation progress for a conversation.
        
        Args:
            conversation_id: Conversation ID
            progress_data: Progress data to add
            
        Returns:
            Success flag
        """
        try:
            # Load the conversation
            conversation_data = self.load_conversation(conversation_id)
            
            if not conversation_data:
                return False
            
            # Initialize progress tracking if not exists
            if "progress" not in conversation_data:
                conversation_data["progress"] = []
            
            # Add this progress point
            progress_point = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **progress_data
            }
            conversation_data["progress"].append(progress_point)
            
            # Save updated conversation
            self.save_conversation(conversation_data)
            
            return True
        except Exception as e:
            st.error(f"Error tracking progress: {str(e)}")
            return False
    
    def create_audit_trail(self, conversation_id: str) -> Dict[str, Any]:
        """
        Create an audit trail of data decisions from a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Audit trail data
        """
        try:
            # Load the conversation
            conversation_data = self.load_conversation(conversation_id)
            
            if not conversation_data:
                return {}
            
            # Extract decisions
            decisions = self.extract_decisions(conversation_id)
            
            # Create audit trail
            audit_trail = {
                "conversation_id": conversation_id,
                "title": conversation_data.get("title", "Untitled"),
                "date": conversation_data.get("timestamp"),
                "data_files": conversation_data.get("data_files", []),
                "decisions": decisions,
                "progress": conversation_data.get("progress", []),
            }
            
            return audit_trail
        except Exception as e:
            st.error(f"Error creating audit trail: {str(e)}")
            return {}
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List available conversations.
        
        Returns:
            List of conversation metadata
        """
        return st.session_state.saved_conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Success flag
        """
        try:
            # Find the conversation file
            matching_conversations = [c for c in st.session_state.saved_conversations if c["id"] == conversation_id]
            
            if not matching_conversations:
                # Try direct file path
                file_path = os.path.join(self.storage_dir, f"conversation_{conversation_id}.json")
                if not os.path.exists(file_path):
                    return False
            else:
                file_path = matching_conversations[0]["file_path"]
            
            # Delete the file
            os.remove(file_path)
            
            # Reload available conversations
            self._load_available_conversations()
            
            return True
        except Exception as e:
            st.error(f"Error deleting conversation: {str(e)}")
            return False
