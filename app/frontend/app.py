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
from app.frontend.chat_interface import ChatInterface
from app.frontend.data_exploration_chat import DataExplorationChat
from app.frontend.conversation_persistence import ConversationPersistence
from app.frontend.suggestion_engine import SmartSuggestionEngine
from app.frontend.visualization import create_schema_diagram, create_relationship_diagram
from app.frontend.onboarding import check_and_show_onboarding

# Load environment variables
load_dotenv(os.path.join("config", ".env"))

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
    
    # Check and show onboarding for first-time users
    if check_and_show_onboarding():
        return
    
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select a page:",
            ["Home", "Upload & Process", "Memory Monitor", "Data Chat", "Data Explorer", "Database Migration", "About"],
        )
        
        st.header("System Status")
        if st.button("Check API Status"):
            try:
                response = requests.get(f"{API_URL}/health")
                if response.status_code == 200:
                    st.success("API is online and healthy")
                else:
                    st.error(f"API returned status code {response.status_code}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")        # Display memory status
        try:
            response = requests.get(f"{API_URL}/memory-status")
            if response.status_code == 200:
                memory_stats = response.json()
                st.metric(
                    label="Memory Usage", 
                    value=f"{memory_stats.get('percent_used', 0) * 100:.1f}%",
                    delta=None
                )
                st.progress(memory_stats.get('percent_used', 0))
                
                max_file_size = memory_stats.get('max_safe_file_size_mb', 0)
                st.info(f"Max safe file size: {max_file_size:.1f} MB")
        except Exception as e:
            st.warning("Memory status unavailable")
    
    # Page content
    if page == "Home":
        render_home_page()
    elif page == "Upload & Process":
        render_upload_page()
    elif page == "Memory Monitor":
        render_memory_page()
    elif page == "Data Chat":
        render_chat_page()
    elif page == "Data Explorer":
        render_explorer_page()
    elif page == "Database Migration":
        render_migration_page()
    elif page == "About":
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
        st.info("💬 [Chat with Your Data](#data-chat)")
    
    with col2:
        st.info("📊 [Explore Your Data](#data-explorer)")
    
    with col3:
        st.info("📤 [Upload Files](#upload-process)")


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
            
            # Display memory usage
            available_mb = memory_data.get("available_memory", 0) / (1024 * 1024)
            total_mb = memory_data.get("total_memory", 0) / (1024 * 1024)
            usage_percent = memory_data.get("percent_used", 0) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Available Memory", f"{available_mb:.2f} MB")
            
            with col2:
                st.metric("Total Memory", f"{total_mb:.2f} MB")
            
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
    - Local LLM integration (CodeLlama-34B) with online fallback options
    - Advanced data processing with pandas and dask
    - Memory monitoring and optimization
    """)
    
    # Version information
    st.info(f"Version: 0.1.0")


if __name__ == "__main__":
    main()