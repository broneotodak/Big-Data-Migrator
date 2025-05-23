"""
Chat interface for data discussions powered by the LLM conversation system.

This module provides a sophisticated Streamlit-based chat interface for:
- Real-time conversations with LLMs about uploaded data
- Visual data exploration within chat
- Schema suggestions and relationship mapping
- Interactive data visualization
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import uuid
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

# Import custom components
from app.llm.conversation_system import LLMConversationSystem
from app.llm.online_llm_fallback import OnlineLLMConfig
from app.memory.memory_monitor import MemoryMonitor
from app.memory.resource_optimizer import ResourceOptimizer
from app.frontend.visualization import create_schema_diagram, create_relationship_diagram
from app.frontend.suggestion_engine import SmartSuggestionEngine


class ChatInterface:
    """
    Sophisticated chat interface for data discussions with LLMs.
    
    This class provides:
    - Real-time conversation with LLM about uploaded data
    - Data-aware chat context with file information
    - Interactive data exploration within chat
    - Visual data previews integrated into conversation
    - Export conversation history and decisions
    """
    
    def __init__(self, api_url: str = None):
        """
        Initialize the chat interface.
        
        Args:
            api_url: URL for the API server
        """
        self.api_url = api_url or f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}"
        
        # Initialize conversation state if not already exists
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = {}
        if "data_context" not in st.session_state:
            st.session_state.data_context = {}
        if "data_preview" not in st.session_state:
            st.session_state.data_preview = {}
        if "suggestion_history" not in st.session_state:
            st.session_state.suggestion_history = []
        
        # Initialize suggestion engine
        self.suggestion_engine = SmartSuggestionEngine()
    
    def render(self):
        """Render the main chat interface with all components."""
        # Layout setup
        col1, col2 = st.columns([7, 3])
        
        with col1:
            # Chat area
            st.header("Data Discussion")
            
            # Render conversation history
            self._render_chat_history()
            
            # Message input area
            self._render_message_input()
        
        with col2:
            # Data context sidebar
            with st.expander("Data Context", expanded=True):
                self._render_data_context()
            
            # Smart suggestions area
            with st.expander("Suggestions", expanded=True):
                self._render_suggestions()
            
            # Data preview area
            with st.expander("Data Preview", expanded=True):
                self._render_data_preview()
            
            # Export and action buttons
            self._render_action_buttons()
    
    def _render_chat_history(self):
        """Render the conversation history."""
        message_container = st.container()
        
        with message_container:
            for message in st.session_state.messages:
                role = message.get("role", "")
                content = message.get("content", "")
                with st.chat_message(role):
                    if "visualization" in message:
                        # This is a special message with visualization
                        st.write(content)
                        viz_type = message["visualization"]["type"]
                        
                        if viz_type == "dataframe":
                            df = pd.DataFrame(message["visualization"]["data"])
                            st.dataframe(df)
                        elif viz_type == "chart":
                            chart_data = message["visualization"]["data"]
                            chart_type = message["visualization"]["chart_type"]
                            if chart_type == "bar":
                                fig = px.bar(chart_data, x=chart_data["x"], y=chart_data["y"])
                                st.plotly_chart(fig, use_container_width=True)
                            elif chart_type == "line":
                                fig = px.line(chart_data, x=chart_data["x"], y=chart_data["y"])
                                st.plotly_chart(fig, use_container_width=True)
                            elif chart_type == "schema":
                                # Schema diagram is rendered using a custom function
                                with st.spinner("Generating schema diagram..."):
                                    st.image(create_schema_diagram(chart_data))
                        else:
                            st.write("Unsupported visualization type")
                    else:
                        # Regular text message
                        st.write(content)
    
    def _render_message_input(self):
        """Render the message input area."""
        if st.session_state.conversation_id:
            # Create a form for the chat input
            with st.form(key="chat_input_form", clear_on_submit=True):
                user_input = st.text_area("Your message:", key="user_input", height=100)
                col1, col2 = st.columns([1, 5])
                
                with col1:
                    submit_button = st.form_submit_button(label="Send")
                
                with col2:
                    if submit_button and user_input:
                        # Send message to API
                        self._send_message(user_input)
        else:
            st.info("Please upload data files first to start a conversation")
    
    def _render_data_context(self):
        """Render the data context information."""
        if st.session_state.uploaded_files:
            st.write("📁 Uploaded Files:")
            for file_name, file_info in st.session_state.uploaded_files.items():
                st.write(f"- {file_name}")
            
            if "files" in st.session_state.data_context:
                file_contexts = st.session_state.data_context.get("files", {})
                for file_path, context in file_contexts.items():
                    file_name = os.path.basename(file_path)
                    with st.expander(f"ℹ️ {file_name} Info"):
                        if "row_count" in context:
                            st.write(f"Rows: {context['row_count']:,}")
                        if "column_count" in context:
                            st.write(f"Columns: {context['column_count']}")
                        if "data_quality" in context:
                            quality = context["data_quality"]
                            st.write(f"Quality Score: {quality.get('overall_score', 'N/A')}")
        else:
            st.info("No data files uploaded")
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload data files to analyze",
                accept_multiple_files=True,
                type=["csv", "xlsx", "xls", "json"]
            )
            
            if uploaded_files:
                self._handle_file_upload(uploaded_files)
    
    def _render_suggestions(self):
        """Render smart suggestions."""
        if st.session_state.conversation_id:
            if st.button("Generate Suggestions"):
                with st.spinner("Generating suggestions..."):
                    suggestions = self._get_suggestions()
                    
                    # Update suggestion history
                    for suggestion in suggestions:
                        if suggestion not in st.session_state.suggestion_history:
                            st.session_state.suggestion_history.append(suggestion)
            
            # Display suggestions
            if st.session_state.suggestion_history:
                for i, suggestion in enumerate(st.session_state.suggestion_history):
                    suggestion_type = suggestion.get("suggestion_type", "")
                    content = suggestion.get("content", "")
                    
                    if suggestion_type == "question":
                        icon = "❓"
                    elif suggestion_type == "exploration":
                        icon = "🔍"
                    elif suggestion_type == "improvement":
                        icon = "⚡"
                    else:
                        icon = "💡"
                        
                    if st.button(f"{icon} {content}", key=f"suggestion_{i}"):
                        self._send_message(content)
            else:
                st.write("No suggestions available yet.")
        else:
            st.info("Upload data to get suggestions")
    
    def _render_data_preview(self):
        """Render data previews."""
        if st.session_state.data_preview:
            for file_name, df in st.session_state.data_preview.items():
                with st.expander(f"Preview: {file_name}"):
                    st.dataframe(df, height=200)
        else:
            st.info("No data previews available")
    
    def _render_action_buttons(self):
        """Render action buttons."""
        if st.session_state.conversation_id:
            st.write("Actions:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Conversation"):
                    self._export_conversation()
                
                if st.button("Schema Analysis"):
                    self._analyze_schema()
            
            with col2:
                if st.button("Relationship Map"):
                    self._create_relationship_map()
                
                if st.button("Quality Analysis"):
                    self._analyze_quality()
    
    def _handle_file_upload(self, uploaded_files):
        """
        Handle file uploads and create conversation.
        
        Args:
            uploaded_files: List of uploaded files
        """
        with st.spinner("Processing uploaded files..."):
            file_paths = []
            
            # Create temp directory if it doesn't exist
            os.makedirs("temp", exist_ok=True)
            
            # Save uploaded files to temp directory
            for uploaded_file in uploaded_files:
                file_path = os.path.join("temp", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(os.path.abspath(file_path))
                
                # Store file info
                st.session_state.uploaded_files[uploaded_file.name] = {
                    "path": file_path,
                    "type": uploaded_file.type,
                    "size": uploaded_file.size
                }
                
                # Create data preview
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(file_path)
                    elif uploaded_file.name.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(file_path)
                    elif uploaded_file.name.endswith(".json"):
                        df = pd.read_json(file_path)
                    else:
                        df = pd.DataFrame(["Unsupported file format"])
                        
                    st.session_state.data_preview[uploaded_file.name] = df
                except Exception as e:
                    st.error(f"Error previewing {uploaded_file.name}: {str(e)}")
            
            # Create conversation with uploaded files
            try:
                conversation = self._create_conversation(file_paths)
                st.session_state.conversation_id = conversation["conversation_id"]
                
                # Add system message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I've analyzed your data files and I'm ready to help. What would you like to know about your data?"
                })
                
                # Get data context
                self._get_data_context()
                
                st.success(f"Created conversation about {len(file_paths)} data files")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error creating conversation: {str(e)}")
    
    def _create_conversation(self, file_paths):
        """
        Create a conversation with the API.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Conversation response
        """
        response = requests.post(
            f"{self.api_url}/llm/conversations",
            json={"title": "Data Analysis", "data_files": file_paths}
        )
        return response.json()
    
    def _get_data_context(self):
        """Get data context from the conversation."""
        # This would normally come from the API, but we'll simulate it for now
        # In a real implementation, you would get this from the conversation_system's data context
        st.session_state.data_context = {
            "files": {}
        }
        
        for file_name, file_info in st.session_state.uploaded_files.items():
            file_path = file_info["path"]
            df = st.session_state.data_preview.get(file_name)
            
            if df is not None:
                st.session_state.data_context["files"][file_path] = {
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data_quality": {
                        "overall_score": 0.85,
                        "completeness": 0.92,
                        "consistency": 0.88,
                        "anomalies": []
                    }
                }
    
    def _send_message(self, message):
        """
        Send a message to the conversation API.
        
        Args:
            message: Message content
        """
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": message})
        
        # Send message to API
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{self.api_url}/llm/conversations/{st.session_state.conversation_id}/messages",
                    json={"message": message}
                )
                response_data = response.json()
                
                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data["response"]
                })
                
                # Check if we need to update data context
                if "guidance" in response_data:
                    # Update suggestions
                    for suggestion in response_data["guidance"].get("suggestions", []):
                        if suggestion not in st.session_state.suggestion_history:
                            st.session_state.suggestion_history.append(suggestion)
        except Exception as e:
            st.error(f"Error sending message: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I'm sorry, I encountered an error: {str(e)}"
            })
    
    def _get_suggestions(self):
        """
        Get suggestions from the API.
        
        Returns:
            List of suggestions
        """
        try:
            response = requests.post(
                f"{self.api_url}/llm/conversations/{st.session_state.conversation_id}/guidance"
            )
            guidance_data = response.json()
            
            # Combine all suggestion types
            suggestions = []
            suggestions.extend(guidance_data.get("suggestions", []))
            suggestions.extend(guidance_data.get("questions", []))
            suggestions.extend(guidance_data.get("improvements", []))
            
            return suggestions
        except Exception as e:
            st.error(f"Error getting suggestions: {str(e)}")
            return []
    
    def _export_conversation(self):
        """Export the conversation history."""
        if not st.session_state.messages:
            st.warning("No conversation to export")
            return
        
        # Prepare conversation for export
        conversation = {
            "id": st.session_state.conversation_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.messages,
            "data_files": list(st.session_state.uploaded_files.keys()),
            "data_context": st.session_state.data_context
        }
        
        # Convert to JSON
        conversation_json = json.dumps(conversation, indent=2)
        
        # Create download button
        st.download_button(
            label="Download Conversation JSON",
            data=conversation_json,
            file_name=f"conversation_{st.session_state.conversation_id}.json",
            mime="application/json"
        )
    
    def _analyze_schema(self):
        """Analyze data schema and add visualization to chat."""
        if not st.session_state.data_preview:
            st.warning("No data to analyze")
            return
        
        with st.spinner("Analyzing schema..."):
            # Generate schema visualization
            schemas = {}
            
            for file_name, df in st.session_state.data_preview.items():
                schema = {}
                for column in df.columns:
                    dtype = str(df[column].dtype)
                    schema[column] = dtype
                schemas[file_name] = schema
            
            # Add schema visualization to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Here is the schema analysis of your data:",
                "visualization": {
                    "type": "schema",
                    "data": schemas
                }
            })
    
    def _create_relationship_map(self):
        """Create and display relationship map between data files."""
        if len(st.session_state.data_preview) < 2:
            st.warning("Need at least two data files to create relationships")
            return
        
        with st.spinner("Generating relationship map..."):
            # In a real implementation, you would get this from the API
            # Here we'll just simulate some relationships
            relationships = []
            files = list(st.session_state.data_preview.keys())
            
            for i in range(len(files) - 1):
                for j in range(i + 1, len(files)):
                    # Try to find common column names as potential relationships
                    df1 = st.session_state.data_preview[files[i]]
                    df2 = st.session_state.data_preview[files[j]]
                    
                    common_columns = set(df1.columns).intersection(set(df2.columns))
                    
                    for col in common_columns:
                        relationships.append({
                            "source_file": files[i],
                            "target_file": files[j],
                            "source_column": col,
                            "target_column": col,
                            "relationship_type": "one-to-many",
                            "confidence": 0.7
                        })
            
            # Add relationship visualization to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I've analyzed potential relationships between your data files:",
                "visualization": {
                    "type": "chart",
                    "chart_type": "schema",
                    "data": {
                        "files": files,
                        "relationships": relationships
                    }
                }
            })
    
    def _analyze_quality(self):
        """Analyze data quality and add results to chat."""
        if not st.session_state.data_preview:
            st.warning("No data to analyze")
            return
        
        with st.spinner("Analyzing data quality..."):
            quality_results = {}
            
            for file_name, df in st.session_state.data_preview.items():
                file_results = {
                    "missing_values": {},
                    "completeness": {},
                    "duplicates": 0
                }
                
                # Calculate missing values
                for column in df.columns:
                    missing = df[column].isna().sum()
                    if missing > 0:
                        file_results["missing_values"][column] = missing
                        file_results["completeness"][column] = 1 - (missing / len(df))
                
                # Check for duplicates
                file_results["duplicates"] = df.duplicated().sum()
                
                quality_results[file_name] = file_results
            
            # Add quality results to chat
            quality_message = "Here's my analysis of your data quality:\n\n"
            
            for file_name, results in quality_results.items():
                quality_message += f"### {file_name}:\n"
                
                # Duplicates
                quality_message += f"- Found {results['duplicates']} duplicate rows\n"
                
                # Missing values
                missing = results.get("missing_values", {})
                if missing:
                    quality_message += "- Missing values in columns:\n"
                    for col, count in missing.items():
                        quality_message += f"  - {col}: {count} missing values\n"
                else:
                    quality_message += "- No missing values found\n"
                
                quality_message += "\n"
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": quality_message
            })
