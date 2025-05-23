"""
Tests for the visualization components.

This module contains unit tests for visualization utilities
to ensure they function correctly.
"""
import unittest
import pandas as pd
import numpy as np
import matplotlib
import networkx as nx
from unittest.mock import MagicMock, patch

from app.frontend.visualization import create_schema_diagram, create_relationship_diagram


class TestVisualization(unittest.TestCase):
    """Tests for the visualization utilities."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Use non-interactive backend for matplotlib
        matplotlib.use('Agg')
        
    def test_create_schema_diagram(self):
        """Test that schema diagrams can be created."""
        # Create test schema data
        schema_data = {
            "tables": {
                "users": {
                    "columns": {
                        "id": {"type": "int", "primary_key": True},
                        "name": {"type": "varchar(255)"},
                        "email": {"type": "varchar(255)"}
                    }
                },
                "orders": {
                    "columns": {
                        "id": {"type": "int", "primary_key": True},
                        "user_id": {"type": "int"},
                        "amount": {"type": "decimal(10,2)"}
                    }
                }
            },
            "relationships": [
                {"source": "orders", "target": "users", "source_column": "user_id", "target_column": "id"}
            ]
        }
        
        # Generate diagram
        result = create_schema_diagram(schema_data)
        
        # Check that we got a bytes result (the image)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)
        
    def test_create_schema_diagram_files_only(self):
        """Test that schema diagrams can be created with just file names."""
        # Create test schema data with just files
        schema_data = {
            "files": ["users.csv", "orders.csv"],
            "relationships": [
                {"source": "orders.csv", "target": "users.csv"}
            ]
        }
        
        # Generate diagram
        result = create_schema_diagram(schema_data)
        
        # Check that we got a bytes result (the image)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)
        
    def test_create_relationship_diagram(self):
        """Test that relationship diagrams can be created."""
        # Create test relationship data
        relationships = [
            {"source": "orders", "target": "users", "strength": 0.8, "description": "Orders belong to users"},
            {"source": "products", "target": "orders", "strength": 0.7, "description": "Products in orders"}
        ]
        
        # Generate diagram
        result = create_relationship_diagram(relationships)
        
        # Check that we got a bytes result (the image)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
