"""
User onboarding component for first-time users.

This module provides a user onboarding experience for first-time users
of the Big Data Migrator application.
"""
import streamlit as st
import time
from typing import Callable, Dict, Optional


class UserOnboarding:
    """
    User onboarding flow for first-time users.
    
    This class:
    - Displays welcome information for first-time users
    - Provides guided tours of key features
    - Explains data chat capabilities
    - Offers quickstart options for common workflows
    """
    
    def __init__(self):
        """Initialize the user onboarding component."""
        # Initialize onboarding state
        if "onboarding_completed" not in st.session_state:
            st.session_state.onboarding_completed = False
        if "current_step" not in st.session_state:
            st.session_state.current_step = 0
        if "show_welcome" not in st.session_state:
            st.session_state.show_welcome = True
    
    def check_and_show(self) -> bool:
        """
        Check if onboarding should be shown and display if needed.
        
        Returns:
            True if onboarding was shown, False otherwise
        """
        # Skip if onboarding is completed
        if st.session_state.onboarding_completed:
            return False
            
        # Show onboarding
        self._show_onboarding()
        return True
    
    def _show_onboarding(self):
        """Display the onboarding flow."""
        # Show welcome modal on first visit
        if st.session_state.show_welcome:
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.markdown("# 👋 Welcome to Big Data Migrator!")
                    st.markdown("""
                    This tool helps you process, analyze, and migrate large datasets efficiently.
                    Let's get you started with a brief tour of the key features.
                    """)
                    
                    if st.button("Start Tour", key="start_tour"):
                        st.session_state.show_welcome = False
                        st.experimental_rerun()
                    
                    if st.button("Skip Tour", key="skip_tour"):
                        st.session_state.onboarding_completed = True
                        st.session_state.show_welcome = False
                        st.experimental_rerun()
        else:
            # Show tour steps
            self._show_tour_step()
    
    def _show_tour_step(self):
        """Show the current tour step."""
        steps = [
            self._step_upload_data,
            self._step_data_chat,
            self._step_data_explorer,
            self._step_migration,
            self._step_completion
        ]
        
        # Show the current step
        current_step = st.session_state.current_step
        if current_step < len(steps):
            steps[current_step]()
        else:
            # Mark onboarding as completed
            st.session_state.onboarding_completed = True
            st.experimental_rerun()
    
    def _step_upload_data(self):
        """Tour step: Uploading data."""
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("## Step 1: Upload Your Data")
                st.markdown("""
                Start by uploading your data files in the **Upload & Process** section.
                The system can handle various file formats including:
                - CSV files
                - Excel spreadsheets
                - Database exports
                - And more!
                
                Memory optimization features will help you process even large files efficiently.
                """)
            
            with col2:
                st.image("https://via.placeholder.com/300x200?text=Upload+Demo", use_column_width=True)
                
            self._show_navigation_buttons()
    
    def _step_data_chat(self):
        """Tour step: Using Data Chat."""
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("## Step 2: Chat with Your Data")
                st.markdown("""
                The **Data Chat** feature allows you to have natural language conversations about your data.
                You can:
                - Ask questions about your dataset
                - Generate visualizations through conversation
                - Get insights and recommendations
                - Export your conversations for documentation
                
                The chat is context-aware and understands your data structure.
                """)
            
            with col2:
                st.image("https://via.placeholder.com/300x200?text=Chat+Demo", use_column_width=True)
                
            self._show_navigation_buttons()
    
    def _step_data_explorer(self):
        """Tour step: Using Data Explorer."""
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("## Step 3: Explore Your Data")
                st.markdown("""
                The **Data Explorer** provides interactive tools for exploring your datasets:
                - Visualize data distributions and relationships
                - Analyze schema structure
                - Identify data quality issues
                - Discover optimization opportunities
                
                Use the explorer to gain deeper insights into your data before migration.
                """)
            
            with col2:
                st.image("https://via.placeholder.com/300x200?text=Explorer+Demo", use_column_width=True)
                
            self._show_navigation_buttons()
    
    def _step_migration(self):
        """Tour step: Database Migration."""
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("## Step 4: Plan Your Migration")
                st.markdown("""
                The **Database Migration** section helps you plan and execute data migrations:
                - Generate migration strategies based on your data
                - Get technology recommendations
                - Review step-by-step migration plans
                - Track migration progress
                
                Your chat conversations and data exploration will inform better migration decisions.
                """)
            
            with col2:
                st.image("https://via.placeholder.com/300x200?text=Migration+Demo", use_column_width=True)
                
            self._show_navigation_buttons()
    
    def _step_completion(self):
        """Tour step: Completion."""
        with st.container():
            st.markdown("## You're All Set!")
            st.markdown("""
            You now know the basics of using the Big Data Migrator.
            
            Here are some quick actions to get started:
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📤 Upload Data")
                if st.button("Go to Upload", key="go_upload"):
                    st.session_state.onboarding_completed = True
                    st.session_state._nav_selection = "Upload & Process"
                    st.experimental_rerun()
            
            with col2:
                st.markdown("### 💬 Start Chatting")
                if st.button("Go to Data Chat", key="go_chat"):
                    st.session_state.onboarding_completed = True
                    st.session_state._nav_selection = "Data Chat"
                    st.experimental_rerun()
            
            with col3:
                st.markdown("### 📊 Explore Data")
                if st.button("Go to Explorer", key="go_explorer"):
                    st.session_state.onboarding_completed = True
                    st.session_state._nav_selection = "Data Explorer"
                    st.experimental_rerun()
            
            st.markdown("---")
            if st.button("Finish Tour", key="finish_tour"):
                st.session_state.onboarding_completed = True
                st.experimental_rerun()
    
    def _show_navigation_buttons(self):
        """Show navigation buttons for tour steps."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state.current_step > 0:
                if st.button("← Previous", key=f"prev_{st.session_state.current_step}"):
                    st.session_state.current_step -= 1
                    st.experimental_rerun()
        
        with col3:
            if st.button("Next →", key=f"next_{st.session_state.current_step}"):
                st.session_state.current_step += 1
                st.experimental_rerun()
        
        with col2:
            if st.button("Skip Tour", key=f"skip_{st.session_state.current_step}"):
                st.session_state.onboarding_completed = True
                st.experimental_rerun()


# Function to use in app.py
def check_and_show_onboarding() -> bool:
    """
    Check if onboarding should be shown and display if needed.
    
    Returns:
        True if onboarding was shown, False otherwise
    """
    onboarding = UserOnboarding()
    return onboarding.check_and_show()
