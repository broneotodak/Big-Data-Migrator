"""
Streamlit frontend for Big Data Migrator
"""
import os
import streamlit as st
import requests
import time
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import uuid

# Import custom components
from chat_interface import ChatInterface
from data_exploration_chat import DataExplorationChat
from conversation_persistence import ConversationPersistence
from suggestion_engine import SmartSuggestionEngine
from visualization import create_schema_diagram, create_relationship_diagram
from debug_monitor import show_debug_monitor, show_processing_details, show_memory_and_performance

# Load environment variables
load_dotenv(".env")  # Load from root directory, not config/.env

# API URL configuration
API_URL = f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}"

# App configuration
st.set_page_config(
    page_title="Big Data Migrator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    """Main Streamlit application"""
    st.title("Big Data Migrator")
    st.subheader("Process, analyze, and migrate large datasets efficiently")
    
    with st.sidebar:
        st.header("Navigation")
        
        # Initialize page selection if not exists
        if "page_selection" not in st.session_state:
            st.session_state.page_selection = "Home"
        
        page = st.radio(
            "Select a page:",
            ["Home", "Upload & Process", "Memory Monitor", "Data Chat", "Data Explorer", "Database Migration", "Debug", "About"],
            index=["Home", "Upload & Process", "Memory Monitor", "Data Chat", "Data Explorer", "Database Migration", "Debug", "About"].index(st.session_state.page_selection) if st.session_state.page_selection in ["Home", "Upload & Process", "Memory Monitor", "Data Chat", "Data Explorer", "Database Migration", "Debug", "About"] else 0,
            key="sidebar_nav"
        )
        
        # Update session state when radio selection changes
        if page != st.session_state.page_selection:
            st.session_state.page_selection = page
        
        st.header("System Status")
        
        # API Status Check
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Check API", use_container_width=True):
                try:
                    response = requests.get(f"{API_URL}/health", timeout=3)
                    if response.status_code == 200:
                        st.success("✅ API Online")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                except Exception as e:
                    st.error("❌ API Offline")
        
        with col2:
            if st.button("🚀 Start API", use_container_width=True):
                st.info("💡 Run: `python main.py`")
        
        # Memory status display with controlled auto-refresh
        st.subheader("💾 Memory Status")
        
        # Initialize refresh state
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = 0
        
        # Add refresh button and auto-refresh option
        col1, col2 = st.columns(2)
        with col1:
            manual_refresh = st.button("🔄 Refresh", use_container_width=True)
        with col2:
            auto_refresh = st.checkbox("🔁 Auto", value=False)  # Default to false to prevent issues
        
        # Check if we should refresh (every 5 seconds if auto-refresh is on)
        current_time = time.time()
        should_refresh = (
            manual_refresh or 
            (auto_refresh and (current_time - st.session_state.last_refresh) > 5)
        )
        
        if should_refresh:
            st.session_state.last_refresh = current_time
            if auto_refresh:
                st.rerun()  # Only rerun if auto-refresh triggered the update
        
        try:
            response = requests.get(f"{API_URL}/memory-status", timeout=2)
            if response.status_code == 200:
                memory_stats = response.json()
                
                # Display metrics in a clean format
                usage_percent = memory_stats.get('memory_usage_percent', 0)
                max_file_size = memory_stats.get('max_safe_file_size_mb', 0)
                available_gb = memory_stats.get('available_memory_gb', 0)
                
                st.metric(
                    label="Memory Usage", 
                    value=f"{usage_percent:.1f}%",
                    delta=f"Available: {available_gb:.1f} GB"
                )
                
                # Progress bar with color coding
                if usage_percent < 60:
                    st.progress(usage_percent/100, text="🟢 Good")
                elif usage_percent < 80:
                    st.progress(usage_percent/100, text="🟡 Warning")
                else:
                    st.progress(usage_percent/100, text="🔴 High Usage")
                
                st.info(f"📁 Max safe file: {max_file_size:.1f} MB")
                
                # Show timestamp of last update
                display_time = time.strftime("%H:%M:%S", time.localtime(st.session_state.last_refresh))
                st.caption(f"Last updated: {display_time}")
                
            else:
                st.warning("⚠️ Memory API unavailable")
        except Exception as e:
            st.warning("⚠️ Memory status offline")
            # Show simple fallback system info
            try:
                import psutil
                memory = psutil.virtual_memory()
                st.metric("💾 System Memory", f"{memory.percent:.1f}%")
                st.progress(memory.percent / 100)
                st.caption("📊 System memory (install API for detailed monitoring)")
            except ImportError:
                # Simple static fallback without external dependencies
                st.info("💡 **Start API for memory monitoring**")
                st.code("python start_api.py", language="bash")
                st.caption("📊 Memory details available when API is running")
    
    # Add debug monitor to sidebar
    show_debug_monitor()
    
    # Page content
    current_page = st.session_state.page_selection
    
    if current_page == "Home":
        render_home_page()
    elif current_page == "Upload & Process":
        render_upload_page()
    elif current_page == "Memory Monitor":
        render_memory_page()
    elif current_page == "Data Chat":
        render_chat_page()
    elif current_page == "Data Explorer":
        render_explorer_page()
    elif current_page == "Database Migration":
        render_migration_page()
    elif current_page == "Debug":
        render_debug_page()
    elif current_page == "About":
        render_about_page()


def render_home_page():
    """Render the home page."""
    st.write("## Welcome to Big Data Migrator")
    st.write("""
    This application helps you process, analyze, and migrate large datasets efficiently.
    Use the sidebar to navigate to different features.
    
    ### Key Features
    - **Upload & Process**: Upload and process data files
    - **Memory Monitor**: Monitor memory usage and system health
    - **Data Chat**: Chat with your data using advanced LLM capabilities
    - **Data Explorer**: Interactive data exploration with visualizations
    - **Database Migration**: Generate migration plans and execute them
    """)
    
    # Quick access cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 Chat with Your Data", use_container_width=True):
            st.session_state.page_selection = "Data Chat"
            st.rerun()
    
    with col2:
        if st.button("📊 Explore Your Data", use_container_width=True):
            st.session_state.page_selection = "Data Explorer"
            st.rerun()
    
    with col3:
        if st.button("📤 Upload Files", use_container_width=True):
            st.session_state.page_selection = "Upload & Process"
            st.rerun()


def render_upload_page():
    """Render the upload and process page."""
    st.header("Upload & Process")
    st.write("Upload your data files for processing and analysis.")
    
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        # Process button
        if st.button("Process Files"):
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        files = {"file": file}
                        response = requests.post(f"{API_URL}/upload-file", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Processed {file.name} successfully")
                            st.json(result)
                        else:
                            st.error(f"Error processing {file.name}: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


def render_memory_page():
    """Render the memory monitoring page."""
    st.header("Memory Monitor")
    st.write("Monitor memory usage and system health.")
    
    try:
        response = requests.get(f"{API_URL}/memory-status")
        if response.status_code == 200:
            memory_data = response.json()
            
            # Use correct field names from API response
            available_gb = memory_data.get("available_memory_gb", 0)
            total_gb = memory_data.get("total_memory_gb", 0)
            used_gb = memory_data.get("used_memory_gb", 0)
            usage_percent = memory_data.get("memory_usage_percent", 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Available Memory", f"{available_gb:.2f} GB")
            
            with col2:
                st.metric("Total Memory", f"{total_gb:.2f} GB")
            
            with col3:
                st.metric("Memory Usage", f"{usage_percent:.1f}%")
            
            # Memory gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=usage_percent,
                title={"text": "Memory Usage"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "green"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "red"},
                    ]
                }
            ))
            st.plotly_chart(fig)
        else:
            st.error(f"Error fetching memory status: {response.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def render_chat_page():
    """Render the data chat page."""
    st.header("Chat with Your Data")
    
    # Initialize chat interface
    chat_interface = ChatInterface(api_url=API_URL)
    
    # Render the chat interface
    chat_interface.render()


def render_explorer_page():
    """Render the data exploration page."""
    st.header("Interactive Data Explorer")
    
    # Initialize data exploration chat
    exploration_chat = DataExplorationChat(api_url=API_URL)
    
    # Render the exploration interface
    exploration_chat.render()


def render_migration_page():
    """Render the database migration page."""
    st.header("Database Migration")
    st.write("Generate and execute database migration plans.")
    
    # Initialize persistence for loading conversations
    persistence = ConversationPersistence()
    
    # Select conversation for migration planning
    conversations = persistence.list_conversations()
    if conversations:
        selected_conversation = st.selectbox(
            "Select a conversation for migration planning",
            options=[c["id"] for c in conversations],
            format_func=lambda x: next((c["title"] for c in conversations if c["id"] == x), x)
        )
        
        if selected_conversation:
            # Load the conversation
            conversation_data = persistence.load_conversation(selected_conversation)
            
            # Display conversation summary
            st.subheader("Conversation Summary")
            st.write(f"Title: {conversation_data.get('title', 'Unknown')}")
            st.write(f"Date: {conversation_data.get('timestamp', 'Unknown')}")
            st.write(f"Files: {', '.join(conversation_data.get('data_files', []))}")
            
            # Extract decisions
            decisions = persistence.extract_decisions(selected_conversation)
            if decisions:
                st.subheader("Key Decisions")
                for i, decision in enumerate(decisions):
                    with st.expander(f"Decision {i+1}"):
                        st.write(f"**Question:** {decision.get('question', '')}")
                        st.write(f"**Decision:** {decision.get('decision', '')}")
            
            # Generate migration plan
            if st.button("Generate Migration Plan"):
                with st.spinner("Generating migration plan..."):
                    # In a real implementation, this would call the API
                    # For now, we'll just simulate a response
                    
                    # Create a suggestion engine
                    suggestion_engine = SmartSuggestionEngine()
                    
                    # Get migration recommendations
                    if "data_preview" in st.session_state:
                        recommendations = suggestion_engine.generate_migration_recommendations(
                            st.session_state.data_preview
                        )
                        
                        # Display recommendations
                        st.subheader("Migration Strategy")
                        st.info(recommendations["explanation"])
                        
                        st.subheader("Recommended Technologies")
                        for tech in recommendations["target_technologies"]:
                            st.write(f"- {tech}")
                        
                        st.subheader("Migration Steps")
                        for i, step in enumerate(recommendations["migration_steps"]):
                            st.write(f"{i+1}. {step}")
                    else:
                        st.error("No data preview available for migration planning")
    else:
        st.info("No conversations found. Start a data chat to create migration plans.")


def render_debug_page():
    """Render the debug monitoring page."""
    st.header("🔍 Debug Monitor")
    st.write("Real-time system monitoring and debugging information.")
    
    # Create tabs for different debug views
    tab1, tab2, tab3 = st.tabs(["💾 Performance", "🔄 Processing", "🧠 LLM Status"])
    
    with tab1:
        show_memory_and_performance()
    
    with tab2:
        st.subheader("Current Processing Tasks")
        
        try:
            response = requests.get(f"{API_URL}/debug/current-processing", timeout=2)
            if response.status_code == 200:
                processing_info = response.json()
                
                if processing_info.get("active_processes"):
                    st.success(f"🔄 {len(processing_info['active_processes'])} active processes")
                    
                    for process in processing_info["active_processes"]:
                        with st.expander(f"📁 {process.get('task', 'Unknown Task')}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Duration:** {process.get('duration_seconds', 0):.1f}s")
                                st.write(f"**Progress:** {process.get('progress', 0)*100:.1f}%")
                            
                            with col2:
                                if process.get('current_step'):
                                    st.write(f"**Current Step:** {process['current_step']}")
                                
                                # Show progress bar
                                progress = process.get('progress', 0)
                                if progress > 0:
                                    st.progress(progress)
                else:
                    st.info("✅ No active processing tasks")
            else:
                st.warning("⚠️ Cannot get processing status")
                
        except requests.exceptions.RequestException:
            st.error("❌ Processing status API unavailable")
        
        # Recent Errors Section
        st.subheader("Recent Errors")
        
        try:
            response = requests.get(f"{API_URL}/debug/recent-errors", timeout=2)
            if response.status_code == 200:
                errors = response.json()
                
                if errors:
                    for error in errors[-5:]:  # Show last 5 errors
                        timestamp = error.get("timestamp", "Unknown")
                        message = error.get("message", "Unknown error")
                        error_type = error.get("type", "general")
                        
                        # Format timestamp
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            formatted_time = dt.strftime("%H:%M:%S")
                        except:
                            formatted_time = timestamp
                        
                        with st.expander(f"⚠️ {error_type.title()} Error - {formatted_time}"):
                            st.error(message)
                            
                            if error.get("details"):
                                st.json(error["details"])
                else:
                    st.success("✅ No recent errors")
            
        except requests.exceptions.RequestException:
            st.warning("⚠️ Error log unavailable")
    
    with tab3:
        st.subheader("LLM Provider Status")
        
        try:
            response = requests.get(f"{API_URL}/llm/status", timeout=2)
            if response.status_code == 200:
                llm_status = response.json()
                
                # Display LM Studio status
                if "lm_studio_connection" in llm_status:
                    conn_status = llm_status["lm_studio_connection"]
                    if "✅" in str(conn_status):
                        st.success(f"**Local LLM (LM Studio):** {conn_status}")
                    else:
                        st.error(f"**Local LLM (LM Studio):** {conn_status}")
                
                # Display conversation system status
                if "conversation_system" in llm_status:
                    conv_status = llm_status["conversation_system"]
                    if "✅" in str(conv_status):
                        st.success(f"**Conversation System:** {conv_status}")
                    else:
                        st.error(f"**Conversation System:** {conv_status}")
                
                # Display configuration
                st.subheader("Configuration")
                config_info = {
                    "Local LLM URL": llm_status.get("local_llm_url", "Unknown"),
                    "Local LLM Model": llm_status.get("local_llm_model", "Unknown"),
                    "Online Fallback": "✅ Enabled" if llm_status.get("enable_online_fallback") else "❌ Disabled"
                }
                
                for key, value in config_info.items():
                    st.write(f"**{key}:** {value}")
            
        except requests.exceptions.RequestException:
            st.error("❌ LLM status API unavailable")
        
        # Conversation Details (if available)
        if hasattr(st.session_state, 'current_conversation_id') and st.session_state.current_conversation_id:
            st.subheader("Current Conversation Details")
            show_processing_details(st.session_state.current_conversation_id)


def render_about_page():
    """Render the about page."""
    st.header("About Big Data Migrator")
    st.write("""
    Big Data Migrator is an intelligent system for processing, analyzing, and migrating large 
    data files with LLM-powered conversation capabilities.
    
    ### Features
    
    - **Large File Processing**: Handle large CSV, Excel, PDF, and other file formats
    - **Data Context Understanding**: Intelligent analysis of data files including statistics and relationships
    - **LLM-powered Conversations**: Discuss your data with both local and online LLMs
    - **Memory Optimization**: Smart resource management for processing large datasets
    - **User Guidance**: Get intelligent suggestions and recommendations for data exploration
    - **Schema Optimization**: Improve data structures with advanced relationship detection
    
    ### Technology Stack
    
    - Python backend with FastAPI
    - Streamlit frontend
    - Local LLM integration with online fallback options
    - Advanced data processing with pandas and dask
    - Memory monitoring and optimization
    """)
    
    # Version information
    st.info(f"Version: 0.1.0")


if __name__ == "__main__":
    main()