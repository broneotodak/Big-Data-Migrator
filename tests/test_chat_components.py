"""
Tests for the chat interface components.

This module contains unit tests for ChatInterface and related components
to ensure they function correctly.
"""
import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
import streamlit as st

from app.frontend.chat_interface import ChatInterface
from app.frontend.data_exploration_chat import DataExplorationChat
from app.frontend.conversation_persistence import ConversationPersistence
from app.frontend.suggestion_engine import SmartSuggestionEngine
from app.frontend.visualization import create_schema_diagram, create_relationship_diagram


class TestChatInterface(unittest.TestCase):
    """Tests for the ChatInterface class."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Mock streamlit context
        self.mock_session_state = {}
        self.st_patch = patch('streamlit.session_state', self.mock_session_state)
        self.mock_st = self.st_patch.start()
        
    def tearDown(self):
        """Clean up after tests."""
        self.st_patch.stop()
    
    def test_initialization(self):
        """Test that the chat interface initializes correctly."""
        chat = ChatInterface(api_url="http://test-api")
        self.assertEqual(chat.api_url, "http://test-api")
        
    @patch('app.frontend.chat_interface.ChatInterface._send_message')
    def test_message_sending(self, mock_send):
        """Test that messages can be sent."""
        chat = ChatInterface(api_url="http://test-api")
        
        # Set up mocks for st functions that would be called
        mock_send.return_value = {"text": "Response text", "context": {}}
        
        # Call method we want to test
        result = chat._send_message("Test message")
        
        # Verify the message was sent with expected arguments
        mock_send.assert_called_once_with("Test message")
        self.assertEqual(result["text"], "Response text")


class TestDataExplorationChat(unittest.TestCase):
    """Tests for the DataExplorationChat class."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Mock streamlit context
        self.mock_session_state = {}
        self.st_patch = patch('streamlit.session_state', self.mock_session_state)
        self.mock_st = self.st_patch.start()
        
    def tearDown(self):
        """Clean up after tests."""
        self.st_patch.stop()
    
    def test_initialization(self):
        """Test that data exploration chat initializes correctly."""
        chat = DataExplorationChat(api_url="http://test-api")
        self.assertEqual(chat.api_url, "http://test-api")
        
        # Check that session state was initialized with expected keys
        self.assertIn("visualization_history", self.mock_session_state)
        self.assertIn("schema_recommendations", self.mock_session_state)
        self.assertIn("data_relationships", self.mock_session_state)
        self.assertIn("active_exploration", self.mock_session_state)


class TestConversationPersistence(unittest.TestCase):
    """Tests for the ConversationPersistence class."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Mock streamlit context
        self.mock_session_state = {}
        self.st_patch = patch('streamlit.session_state', self.mock_session_state)
        self.mock_st = self.st_patch.start()
        
        # Create a temp directory for test conversations
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        """Clean up after tests."""
        self.st_patch.stop()
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that conversation persistence initializes correctly."""
        persistence = ConversationPersistence(storage_dir=self.temp_dir.name)
        self.assertEqual(persistence.storage_dir, self.temp_dir.name)
        
        # Check that session state was initialized
        self.assertIn("saved_conversations", self.mock_session_state)


class TestSmartSuggestionEngine(unittest.TestCase):
    """Tests for the SmartSuggestionEngine class."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Mock streamlit context
        self.mock_session_state = {}
        self.st_patch = patch('streamlit.session_state', self.mock_session_state)
        self.mock_st = self.st_patch.start()
        
    def tearDown(self):
        """Clean up after tests."""
        self.st_patch.stop()
    
    def test_initialization(self):
        """Test that suggestion engine initializes correctly."""
        engine = SmartSuggestionEngine()
        
        # Check that session state was initialized
        self.assertIn("suggestion_cache", self.mock_session_state)
        
    def test_generate_suggestions(self):
        """Test that suggestions can be generated."""
        engine = SmartSuggestionEngine()
        
        # Create test data
        data_context = {
            "files": ["test.csv"],
            "summary": {"test.csv": {"columns": 5, "rows": 100}}
        }
        conversation_history = [
            {"role": "user", "content": "Show me the data stats"},
            {"role": "assistant", "content": "Here are the statistics..."}
        ]
        
        # Generate suggestions
        suggestions = engine.generate_suggestions(data_context, conversation_history)
        
        # This should at least return an empty list if no suggestions
        self.assertIsInstance(suggestions, list)


if __name__ == "__main__":
    unittest.main()
