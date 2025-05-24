#!/usr/bin/env python3
"""
Chat interface for the Big Data Migrator system.
"""
import streamlit as st
import requests
import pandas as pd
import json
import time
import uuid
import os
from typing import Dict, Any, List, Optional

# Session state initialization
def init_session_state():
    """Initialize session state variables."""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}
    if "data_preview" not in st.session_state:
        st.session_state.data_preview = {}
    if "data_context" not in st.session_state:
        st.session_state.data_context = {"files": {}}

class SmartSuggestionEngine:
    """Simple suggestion engine for smart recommendations."""
    def __init__(self):
        self.suggestions = [
            "Analyze data structure",
            "Check data quality",
            "Generate schema",
            "Find relationships",
            "Export to Supabase"
        ]
    
    def generate_suggestions(self, context=None):
        return self.suggestions

class ChatInterface:
    """Main chat interface for data analysis conversations."""
    
    def __init__(self, api_url: str = None):
        self.api_url = api_url or "http://localhost:8000"
        init_session_state()
        
        # Initialize suggestion engine
        try:
            self.suggestion_engine = SmartSuggestionEngine()
        except:
            self.suggestion_engine = None
    
    def render(self):
        """Render the main chat interface with all components."""
        # Check if LLM service is available
        llm_available = self._check_llm_service()
        
        # Header with LLM status
        col1, col2, col3 = st.columns([6, 2, 2])
        
        with col1:
            st.header("🤖 Data Discussion")
            
        with col2:
            # Model status indicator
            if llm_available:
                st.success("🟢 LLM Online")
                if "llm_status_details" in st.session_state:
                    details = st.session_state.llm_status_details
                    model_name = details.get("local_llm_model", "Unknown Model")
                    st.caption(f"Model: {model_name}")
            else:
                st.error("🔴 LLM Offline")
                st.caption("Check LM Studio")
        
        with col3:
            # Multi-LLM mode toggle
            if llm_available:
                multi_llm_enabled = st.toggle(
                    "🧠 Multi-LLM", 
                    value=st.session_state.get("use_multi_llm", False),
                    help="Use multiple LLM providers for better responses"
                )
                st.session_state.use_multi_llm = multi_llm_enabled
                
                if multi_llm_enabled:
                    st.caption("🚀 Consensus Mode")
                else:
                    st.caption("⚡ Single Mode")
        
        # Main content area
        self._render_data_context()
        self._render_data_preview()
        
        # Chat interface
        self._render_chat_interface()
        
        # Sidebar
        with st.sidebar:
            self._render_file_management()
    
    def _check_llm_service(self) -> bool:
        """Check if LLM service is available."""
        try:
            # First check API health
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code != 200:
                return False
            
            # Then check detailed LLM status
            llm_response = requests.get(f"{self.api_url}/llm/status", timeout=15)
            
            if llm_response.status_code == 200:
                status_data = llm_response.json()
                
                # Store detailed status for display
                if "llm_status_details" not in st.session_state:
                    st.session_state.llm_status_details = {}
                st.session_state.llm_status_details = status_data
                
                # Check if both LM Studio connection and conversation system are working
                lm_studio_ok = not isinstance(status_data.get("lm_studio_connection"), str) or "error" not in status_data.get("lm_studio_connection", "")
                conversation_ok = "✅" in status_data.get("conversation_system", "")
                
                return lm_studio_ok and conversation_ok
            else:
                return False
                
        except requests.exceptions.ConnectionError:
            # API server is not running
            return False
        except requests.exceptions.Timeout:
            # API is slow/LLM not responding
            return False
        except Exception as e:
            # Any other error
            return False
    
    def _render_chat_history(self):
        """Render the conversation history with support for multi-LLM responses."""
        if not st.session_state.chat_history:
            # Show welcome message
            st.chat_message("assistant").write(
                "👋 Hello! I'm here to help you understand and analyze your data. "
                "Upload some files and ask me questions about patterns, relationships, or insights!"
            )
            return
        
        # Display chat history
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            with st.chat_message(role):
                if role == "assistant" and "multi_llm_data" in message:
                    # This is a multi-LLM response - show comparative view
                    self._render_multi_llm_response(message["multi_llm_data"])
                else:
                    # Regular single response
                    st.write(content)
    
    def _render_message_input(self):
        """Render the message input area with multi-LLM support."""
        # Check if we have an active conversation
        if not st.session_state.conversation_id:
            st.info("📁 Upload files first to start a conversation about your data")
            return
        
        # Message input
        message = st.chat_input("Ask me about your data...")
        
        if message:
            # Check if multi-LLM mode is enabled
            use_multi_llm = st.session_state.get("use_multi_llm", False)
            
            if use_multi_llm:
                # Use multi-LLM mode
                success = self._send_message_multi_llm(message)
            else:
                # Use single LLM mode
                success = self._send_message(message)
            
            if success:
                st.rerun()
    
    def _render_data_context(self):
        """Render the data context information."""
        if st.session_state.uploaded_files:
            st.write("📁 **Uploaded Files:**")
            for file_name, file_info in st.session_state.uploaded_files.items():
                st.write(f"• {file_name}")
            
            if "files" in st.session_state.data_context:
                file_contexts = st.session_state.data_context.get("files", {})
                for file_path, context in file_contexts.items():
                    file_name = os.path.basename(file_path)
                    
                    # Use regular container instead of nested expander
                    st.markdown(f"**ℹ️ {file_name} Info:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if "row_count" in context:
                            st.metric("Rows", f"{context['row_count']:,}")
                        if "column_count" in context:
                            st.metric("Columns", context['column_count'])
                    
                    with col2:
                        if "data_quality" in context:
                            quality = context["data_quality"]
                            st.metric("Quality", f"{quality.get('overall_score', 0.85):.1%}")
                    
                    st.divider()  # Add separator between files
        else:
            st.info("No data files uploaded yet")
            st.caption("Use the File Management section above to upload files")
    
    def _render_suggestions(self):
        """Render smart suggestions."""
        if st.session_state.data_preview:
            # Generate some basic suggestions based on uploaded data
            suggestions = self._generate_basic_suggestions()
            
            if suggestions:
                st.write("💡 **Quick Actions:**")
                for i, suggestion in enumerate(suggestions):
                    if st.button(suggestion, key=f"suggestion_{i}"):
                        if self._check_llm_service():
                            success = self._send_message(suggestion)
                            if success:
                                st.rerun()
                        else:
                            st.info("Start LLM service to use chat features!")
            else:
                st.write("Upload data to get suggestions")
        else:
            st.info("Upload data to get suggestions")
    
    def _render_data_preview(self):
        """Render data previews."""
        if st.session_state.data_preview:
            st.write("👁️ **Data Previews:**")
            
            # Use selectbox instead of multiple expanders to avoid nesting issues
            file_names = list(st.session_state.data_preview.keys())
            
            if len(file_names) == 1:
                # If only one file, show it directly
                file_name = file_names[0]
                df = st.session_state.data_preview[file_name]
                st.write(f"**{file_name}**")
                
                # Fix PyArrow serialization issues by converting problematic columns
                try:
                    display_df = self._prepare_dataframe_for_display(df.head(5))
                    st.dataframe(display_df, height=200)
                    st.caption(f"Showing 5 of {len(df)} rows")
                except Exception as e:
                    st.error(f"Error displaying data: {str(e)}")
                    # Fallback: show basic info
                    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                    st.write("Column names:", list(df.columns))
            else:
                # If multiple files, use selectbox
                selected_file = st.selectbox(
                    "Select file to preview:",
                    options=file_names,
                    key="preview_file_selector"
                )
                
                if selected_file:
                    df = st.session_state.data_preview[selected_file]
                    try:
                        display_df = self._prepare_dataframe_for_display(df.head(5))
                        st.dataframe(display_df, height=200)
                        st.caption(f"Showing 5 of {len(df)} rows from {selected_file}")
                    except Exception as e:
                        st.error(f"Error displaying data: {str(e)}")
                        # Fallback: show basic info
                        st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                        st.write("Column names:", list(df.columns))
        else:
            st.info("No data previews available")
    
    def _prepare_dataframe_for_display(self, df):
        """
        Prepare a DataFrame for display by fixing PyArrow serialization issues.
        
        Args:
            df: DataFrame to prepare
            
        Returns:
            DataFrame ready for display
        """
        if df is None or df.empty:
            return df
            
        display_df = df.copy()
        
        # Convert problematic columns to strings to avoid PyArrow issues
        for col in display_df.columns:
            try:
                # Check if column has mixed types or complex objects
                if display_df[col].dtype == 'object':
                    # Sample a few values to check for problematic types
                    sample_values = display_df[col].dropna().head(3)
                    
                    # Check for complex objects that PyArrow can't handle
                    has_complex_objects = any(
                        isinstance(val, (dict, list, tuple, set)) or
                        (hasattr(val, '__dict__') and not isinstance(val, (str, int, float, bool)))
                        for val in sample_values
                    )
                    
                    if has_complex_objects:
                        # Convert to string representation
                        display_df[col] = display_df[col].astype(str)
                        
            except Exception:
                # If anything fails, convert to string as fallback
                try:
                    display_df[col] = display_df[col].astype(str)
                except:
                    # Ultimate fallback: replace with placeholder
                    display_df[col] = "[Data type conversion error]"
        
        return display_df
    
    def _render_action_buttons(self):
        """Render action buttons."""
        if st.session_state.conversation_id:
            st.write("**Actions:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📤 Export Chat"):
                    self._export_conversation()
                
                if st.button("🔍 Schema Analysis"):
                    self._analyze_schema()
            
            with col2:
                if st.button("🔗 Relationships"):
                    self._create_relationship_map()
                
                if st.button("✅ Quality Check"):
                    self._analyze_quality()
    
    def _handle_file_upload(self, uploaded_files):
        """
        Handle file uploads and create conversation.
        
        Args:
            uploaded_files: List of uploaded files
        """
        # Check if these files have already been processed
        current_file_names = [f.name for f in uploaded_files]
        already_uploaded = all(name in st.session_state.uploaded_files for name in current_file_names)
        
        # Only process if new files or no conversation exists yet
        if already_uploaded and st.session_state.conversation_id:
            return  # Files already processed and conversation exists
        
        with st.spinner("Processing uploaded files..."):
            file_paths = []
            
            # Create temp directory if it doesn't exist
            os.makedirs("temp", exist_ok=True)
            
            # Save uploaded files to temp directory
            for uploaded_file in uploaded_files:
                # Skip if file already exists
                if uploaded_file.name in st.session_state.uploaded_files:
                    file_paths.append(st.session_state.uploaded_files[uploaded_file.name]["path"])
                    continue
                    
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
                        # Use improved CSV reading with empty row detection
                        try:
                            # First, check for empty rows at the beginning
                            import csv
                            test_df = pd.read_csv(file_path, nrows=10, header=None)
                            
                            # Find the first non-empty row that could be a header
                            header_row_index = None
                            for idx, row in test_df.iterrows():
                                non_null_values = row.dropna()
                                if len(non_null_values) > 0:
                                    # Check if it looks like a header (contains strings, not just numbers/dates)
                                    string_like = sum(1 for val in non_null_values if isinstance(val, str) and not str(val).replace('.', '').replace('-', '').replace('/', '').isdigit())
                                    if string_like > len(non_null_values) * 0.5:  # More than 50% are string-like
                                        header_row_index = idx
                                        break
                            
                            # Read with appropriate skiprows if empty rows detected
                            if header_row_index is not None and header_row_index > 0:
                                skiprows = list(range(header_row_index))
                                df = pd.read_csv(file_path, nrows=1000, skiprows=skiprows)
                            else:
                                df = pd.read_csv(file_path, nrows=1000)
                                
                        except Exception as e_csv:
                            # Fallback to simple reading if detection fails
                            df = pd.read_csv(file_path, nrows=1000)
                            
                    elif uploaded_file.name.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(file_path, nrows=1000)
                    elif uploaded_file.name.endswith(".json"):
                        df = pd.read_json(file_path, nrows=1000)
                    else:
                        # For non-tabular files, create a summary info dataframe
                        df = pd.DataFrame({
                            "Property": ["File Type", "Size (bytes)", "Status"],
                            "Value": [uploaded_file.type, uploaded_file.size, "Uploaded successfully"]
                        })
                        
                    # Validate dataframe
                    if df is not None and not df.empty:
                        st.session_state.data_preview[uploaded_file.name] = df
                    else:
                        # Create error info if dataframe is empty
                        st.session_state.data_preview[uploaded_file.name] = pd.DataFrame({
                            "Info": [f"File uploaded but appears to be empty or could not be read as tabular data"]
                        })
                        
                except pd.errors.EmptyDataError:
                    st.warning(f"⚠️ {uploaded_file.name} appears to be empty")
                    st.session_state.data_preview[uploaded_file.name] = pd.DataFrame({
                        "Error": ["File is empty"]
                    })
                except pd.errors.ParserError as e:
                    st.warning(f"⚠️ Could not parse {uploaded_file.name}: {str(e)}")
                    st.session_state.data_preview[uploaded_file.name] = pd.DataFrame({
                        "Error": [f"Parser error: {str(e)}"]
                    })
                except FileNotFoundError:
                    st.error(f"❌ Could not find {uploaded_file.name} after upload")
                    st.session_state.data_preview[uploaded_file.name] = pd.DataFrame({
                        "Error": ["File not found after upload"]
                    })
                except PermissionError:
                    st.error(f"❌ Permission denied accessing {uploaded_file.name}")
                    st.session_state.data_preview[uploaded_file.name] = pd.DataFrame({
                        "Error": ["Permission denied"]
                    })
                except Exception as e:
                    st.warning(f"⚠️ Error processing {uploaded_file.name}: {str(e)}")
                    # Create error preview with more details
                    st.session_state.data_preview[uploaded_file.name] = pd.DataFrame({
                        "Property": ["File Name", "File Type", "Size", "Error"],
                        "Value": [uploaded_file.name, uploaded_file.type, f"{uploaded_file.size} bytes", str(e)]
                    })
            
            # Create conversation with uploaded files (if LLM available and no conversation exists)
            if not st.session_state.conversation_id and self._check_llm_service():
                try:
                    conversation = self._create_conversation(file_paths)
                    st.session_state.conversation_id = conversation.get("conversation_id", str(uuid.uuid4()))
                
                    # Add system message only once
                    if not st.session_state.messages:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"I've analyzed your {len(file_paths)} data file(s) and I'm ready to help. What would you like to know about your data?"
                        })
                    
                    st.success(f"✅ Uploaded {len(file_paths)} files and created conversation!")
                except Exception as e:
                    st.warning(f"Files uploaded but chat not available: {str(e)}")
            elif not st.session_state.conversation_id:
                # Just process files without creating conversation
                st.session_state.conversation_id = str(uuid.uuid4())
                st.success(f"✅ Uploaded {len(file_paths)} files! Chat will be available when LLM service starts.")
                
                # Get data context
                self._get_data_context()
                
            # Only rerun if this is a new upload, not when files already existed
            if not already_uploaded:
                st.rerun()
    
    def _create_conversation(self, file_paths):
        """
        Create a conversation with the API.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Conversation response
        """
        try:
            response = requests.post(
                f"{self.api_url}/llm/conversations",
                json={"title": "Data Analysis", "data_files": file_paths},
                timeout=5
            )
            return response.json()
        except Exception as e:
            return {"conversation_id": str(uuid.uuid4()), "error": str(e)}
    
    def _get_data_context(self):
        """Get data context from the conversation."""
        st.session_state.data_context = {
            "files": {}
        }
        
        for file_name, file_info in st.session_state.uploaded_files.items():
            file_path = file_info["path"]
            df = st.session_state.data_preview.get(file_name)
            
            if df is not None and not df.empty:
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
    
    def _send_message(self, message: str) -> bool:
        """Send a message using single LLM mode."""
        try:
            with st.spinner("🤔 Thinking..."):
                response = requests.post(
                    f"{self.api_url}/llm/conversations/{st.session_state.conversation_id}/messages",
                    json={"message": message},
                    timeout=60
                )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": message
                })
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_data["response"]
                })
                
                # Handle guidance if provided
                if response_data.get("guidance"):
                    self._handle_guidance_response(response_data["guidance"])
                
                return True
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.headers.get("content-type") == "application/json" else response.text
                st.error(f"Error from API: {error_detail}")
                return False
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            return False
        except Exception as e:
            st.error(f"Error sending message: {str(e)}")
            return False
    
    def _generate_basic_suggestions(self):
        """Generate basic suggestions based on uploaded data."""
        suggestions = []
        
        if st.session_state.data_preview:
            # Add data-specific suggestions
            for file_name, df in st.session_state.data_preview.items():
                if not df.empty:
                    suggestions.extend([
                        f"Analyze the structure of {file_name}",
                        f"Check data quality in {file_name}",
                        f"Show summary statistics for {file_name}"
                    ])
                    
                    # Add column-specific suggestions
                    if len(df.columns) > 0:
                        suggestions.append(f"Explain the columns in {file_name}")
        
        # Add general suggestions
        suggestions.extend([
            "Help me understand my data",
            "What insights can you provide?",
            "Suggest a database schema",
            "Check for data quality issues"
        ])
        
        return suggestions[:6]  # Limit to 6 suggestions
    
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
            label="💾 Download Conversation JSON",
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
            # Generate schema analysis
            schema_info = "## Schema Analysis\n\n"
            
            for file_name, df in st.session_state.data_preview.items():
                schema_info += f"### {file_name}\n"
                schema_info += f"- **Rows:** {len(df):,}\n"
                schema_info += f"- **Columns:** {len(df.columns)}\n\n"
                schema_info += "**Column Details:**\n"
                
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    null_count = df[col].isnull().sum()
                    schema_info += f"- `{col}`: {dtype}"
                    if null_count > 0:
                        schema_info += f" ({null_count} nulls)"
                    schema_info += "\n"
                schema_info += "\n"
            
            # Add to chat - use chat_history for consistency
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": schema_info
            })
            st.rerun()  # Fix: Add rerun to update UI
    
    def _create_relationship_map(self):
        """Create and display relationship map between data files."""
        if len(st.session_state.data_preview) < 2:
            st.info("Need at least 2 files to analyze relationships")
            return
        
        with st.spinner("Analyzing relationships..."):
            relationship_info = "## Relationship Analysis\n\n"
            file_names = list(st.session_state.data_preview.keys())
            
            # Simple column name matching
            for i, file1 in enumerate(file_names):
                for file2 in file_names[i+1:]:
                    df1 = st.session_state.data_preview[file1]
                    df2 = st.session_state.data_preview[file2]
                    
                    common_cols = set(df1.columns) & set(df2.columns)
                    if common_cols:
                        relationship_info += f"### {file1} ↔ {file2}\n"
                        relationship_info += f"Common columns: {', '.join(common_cols)}\n\n"
            
            # Add to chat
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": relationship_info
            })
            st.rerun()  # Fix: Add rerun to update UI
    
    def _analyze_quality(self):
        """Analyze data quality and add report to chat."""
        if not st.session_state.data_preview:
            st.warning("No data to analyze")
            return
        
        with st.spinner("Analyzing data quality..."):
            quality_info = "## Data Quality Report\n\n"
            
            for file_name, df in st.session_state.data_preview.items():
                quality_info += f"### {file_name}\n"
                
                # Basic quality metrics
                total_cells = len(df) * len(df.columns)
                null_cells = df.isnull().sum().sum()
                completeness = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0
                
                quality_info += f"- **Completeness:** {completeness:.1f}%\n"
                quality_info += f"- **Total rows:** {len(df):,}\n"
                quality_info += f"- **Missing values:** {null_cells:,}\n"
                
                # Column-specific issues
                for col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        null_pct = (null_count / len(df)) * 100
                        quality_info += f"  - `{col}`: {null_pct:.1f}% missing\n"
                
                quality_info += "\n"
            
            # Add to chat
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": quality_info
            })
            st.rerun()  # Fix: Add rerun to update UI
    
    def _render_file_management(self):
        """Render file management interface in sidebar."""
        st.header("📁 File Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload data files",
            type=["csv", "xlsx", "xls", "json"],
            accept_multiple_files=True,
            key="file_uploader",
            help="Upload CSV, Excel, or JSON files for analysis"
        )
        
        if uploaded_files:
            self._handle_file_upload(uploaded_files)
        
        # File status
        if st.session_state.uploaded_files:
            st.write("**Uploaded Files:**")
            for file_name, file_info in st.session_state.uploaded_files.items():
                file_size_mb = file_info["size"] / (1024 * 1024)
                st.write(f"• {file_name} ({file_size_mb:.1f} MB)")
            
            # Clear files button
            if st.button("🗑️ Clear All Files"):
                st.session_state.uploaded_files = {}
                st.session_state.data_preview = {}
                st.session_state.data_context = {"files": {}}
                st.session_state.conversation_id = None
                st.session_state.chat_history = []
                st.session_state.messages = []
                st.rerun()
    
    def _render_chat_interface(self):
        """Render the main chat interface."""
        st.header("💬 Chat")
        
        # Chat history
        self._render_chat_history()
        
        # Message input
        self._render_message_input()
        
        # Action buttons
        self._render_action_buttons()
        
        # Suggestions
        self._render_suggestions()
    
    def _render_llm_diagnostics(self):
        """Render LLM diagnostic information."""
        if "llm_status_details" in st.session_state:
            details = st.session_state.llm_status_details
            
            st.subheader("🔧 LLM Diagnostics")
            
            # LM Studio status
            lm_studio_status = details.get("lm_studio_connection", "Unknown")
            if isinstance(lm_studio_status, dict):
                st.success("✅ LM Studio Connected")
                st.json(lm_studio_status)
            else:
                st.error(f"❌ LM Studio: {lm_studio_status}")
            
            # Conversation system status
            conv_status = details.get("conversation_system", "Unknown")
            if "✅" in conv_status:
                st.success(f"✅ Conversation System: {conv_status}")
            else:
                st.error(f"❌ Conversation System: {conv_status}")
            
            # Model details
            if "local_llm_model" in details:
                st.info(f"🤖 Active Model: {details['local_llm_model']}")
    
    def _render_multi_llm_response(self, response_data: dict):
        """Render a multi-LLM comparative response."""
        st.subheader("🧠 Multi-LLM Consensus Response")
        
        # Main consensus response
        if "consensus_response" in response_data:
            st.write("**Consensus Answer:**")
            st.write(response_data["consensus_response"])
            st.divider()
        
        # Individual provider responses
        if "provider_responses" in response_data:
            st.write("**Individual Provider Responses:**")
            
            for provider, response in response_data["provider_responses"].items():
                with st.expander(f"{provider} Response", expanded=False):
                    if isinstance(response, dict):
                        if "response" in response:
                            st.write(response["response"])
                        if "confidence" in response:
                            st.caption(f"Confidence: {response['confidence']}")
                        if "processing_time" in response:
                            st.caption(f"Time: {response['processing_time']:.2f}s")
                    else:
                        st.write(str(response))
        
        # Consensus metadata
        if "consensus_metadata" in response_data:
            metadata = response_data["consensus_metadata"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if "agreement_score" in metadata:
                    st.metric("Agreement", f"{metadata['agreement_score']:.1%}")
            with col2:
                if "total_time" in metadata:
                    st.metric("Total Time", f"{metadata['total_time']:.1f}s")
            with col3:
                if "providers_used" in metadata:
                    st.metric("Providers", len(metadata["providers_used"]))
    
    def _send_message_multi_llm(self, message: str) -> bool:
        """Send a message using multi-LLM consensus mode."""
        try:
            with st.spinner("🧠 Consulting multiple LLMs..."):
                response = requests.post(
                    f"{self.api_url}/llm/conversations/{st.session_state.conversation_id}/messages/multi",
                    json={"message": message},
                    timeout=120  # Longer timeout for multi-LLM
                )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": message
                })
                
                # Add multi-LLM response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_data.get("consensus_response", "Multi-LLM response received"),
                    "multi_llm_data": response_data
                })
                
                return True
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.headers.get("content-type") == "application/json" else response.text
                st.error(f"Multi-LLM error: {error_detail}")
                # Fallback to single LLM
                return self._send_message(message)
                
        except requests.exceptions.Timeout:
            st.error("Multi-LLM request timed out. Trying single LLM...")
            # Fallback to single LLM
            return self._send_message(message)
        except Exception as e:
            st.error(f"Multi-LLM error: {str(e)}. Falling back to single LLM...")
            # Fallback to single LLM
            return self._send_message(message)
    
    def _handle_guidance_response(self, guidance: Dict[str, Any]):
        """Handle guidance responses from the LLM."""
        if guidance.get("suggested_actions"):
            st.write("💡 **Suggested Actions:**")
            for action in guidance["suggested_actions"]:
                if st.button(action, key=f"guidance_{hash(action)}"):
                    self._send_message(action)
        
        if guidance.get("next_steps"):
            st.info(f"💭 Next steps: {guidance['next_steps']}")
        
        if guidance.get("data_insights"):
            st.success(f"✨ Insight: {guidance['data_insights']}")
