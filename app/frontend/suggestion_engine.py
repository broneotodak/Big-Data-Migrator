"""
Smart suggestion engine for contextual data recommendations.

This module provides intelligent suggestions during data conversations,
helping users explore their data more effectively and make better decisions.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import time


class SmartSuggestionEngine:
    """
    Provides contextual data suggestions and recommendations.
    
    This class:
    - Provides contextual suggestions during conversation
    - Offers data improvement recommendations
    - Suggests optimal database schemas
    - Recommends data cleaning steps
    - Provides migration strategy options
    """
    
    def __init__(self):
        """Initialize the smart suggestion engine."""
        # Initialize suggestion cache
        if "suggestion_cache" not in st.session_state:
            st.session_state.suggestion_cache = {}
    
    def generate_suggestions(self, 
                           data_context: Dict[str, Any], 
                           conversation_history: List[Dict[str, Any]],
                           max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate contextual suggestions based on data and conversation.
        
        Args:
            data_context: Data context information
            conversation_history: Conversation history
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggestion dictionaries
        """
        suggestions = []
        
        # Check if we have suggestions in cache and they're recent
        cache_key = self._create_cache_key(data_context, conversation_history)
        if cache_key in st.session_state.suggestion_cache:
            cached_result = st.session_state.suggestion_cache[cache_key]
            # Use cache if it's less than 5 minutes old
            if time.time() - cached_result["timestamp"] < 300:
                return cached_result["suggestions"][:max_suggestions]
        
        # Generate suggestions based on data context
        if "files" in data_context:
            file_suggestions = self._generate_file_based_suggestions(data_context["files"])
            suggestions.extend(file_suggestions)
        
        # Generate suggestions based on conversation
        if conversation_history:
            conversation_suggestions = self._generate_conversation_based_suggestions(conversation_history)
            suggestions.extend(conversation_suggestions)
        
        # Generate data improvement suggestions
        if "files" in data_context:
            improvement_suggestions = self._generate_improvement_suggestions(data_context["files"])
            suggestions.extend(improvement_suggestions)
        
        # Sort by priority and limit
        suggestions.sort(key=lambda x: x.get("priority", 0), reverse=True)
        suggestions = suggestions[:max_suggestions]
        
        # Cache the result
        st.session_state.suggestion_cache[cache_key] = {
            "timestamp": time.time(),
            "suggestions": suggestions
        }
        
        return suggestions
    
    def _create_cache_key(self, data_context: Dict[str, Any], conversation_history: List[Dict[str, Any]]) -> str:
        """
        Create a cache key for suggestions.
        
        Args:
            data_context: Data context
            conversation_history: Conversation history
            
        Returns:
            Cache key string
        """
        # Use the number of files and messages as a simple cache key
        file_count = len(data_context.get("files", {}))
        message_count = len(conversation_history)
        
        # Include the latest message in the key if available
        latest_msg = ""
        if conversation_history:
            latest_msg = conversation_history[-1].get("content", "")[:50]
        
        return f"files_{file_count}_msgs_{message_count}_latest_{latest_msg}"
    
    def _generate_file_based_suggestions(self, files_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate suggestions based on file metadata.
        
        Args:
            files_context: Files context information
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        for file_path, file_context in files_context.items():
            file_name = file_path.split("/")[-1] if "/" in file_path else file_path.split("\\")[-1]
            
            # Suggestion for data overview
            suggestions.append({
                "suggestion_type": "exploration",
                "content": f"Give me a summary of the {file_name} dataset",
                "priority": 90
            })
            
            # Check for missing values
            if "data_quality" in file_context and "completeness" in file_context["data_quality"]:
                completeness = file_context["data_quality"]["completeness"]
                if completeness < 0.95:
                    suggestions.append({
                        "suggestion_type": "improvement",
                        "content": f"What's the best way to handle missing values in {file_name}?",
                        "priority": 80 if completeness < 0.9 else 70
                    })
            
            # Suggestion for column types
            if "column_types" in file_context:
                suggestions.append({
                    "suggestion_type": "exploration",
                    "content": f"What are the data types in {file_name} and are they appropriate?",
                    "priority": 60
                })
            
            # Suggestion for outliers if numeric columns exist
            numeric_columns = [col for col, dtype in file_context.get("column_types", {}).items() 
                              if "int" in dtype.lower() or "float" in dtype.lower()]
            if numeric_columns:
                suggestions.append({
                    "suggestion_type": "exploration",
                    "content": f"Are there any outliers in {file_name} that I should be aware of?",
                    "priority": 50
                })
        
        # If multiple files, suggest relationship analysis
        if len(files_context) > 1:
            suggestions.append({
                "suggestion_type": "exploration",
                "content": "What relationships exist between these datasets?",
                "priority": 95
            })
            
            suggestions.append({
                "suggestion_type": "improvement",
                "content": "What's the optimal database schema for these datasets?",
                "priority": 85
            })
        
        return suggestions
    
    def _generate_conversation_based_suggestions(self, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate suggestions based on conversation history.
        
        Args:
            conversation_history: Conversation history
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Only look at recent messages
        recent_messages = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        
        # Look for keywords to generate targeted suggestions
        keywords = {
            "missing values": [
                {
                    "suggestion_type": "improvement",
                    "content": "What imputation methods would work best for this data?",
                    "priority": 75
                }
            ],
            "outlier": [
                {
                    "suggestion_type": "improvement",
                    "content": "Should I remove outliers or transform them?",
                    "priority": 70
                }
            ],
            "correlation": [
                {
                    "suggestion_type": "exploration",
                    "content": "Show me the correlation matrix between numeric columns",
                    "priority": 65
                }
            ],
            "distribution": [
                {
                    "suggestion_type": "exploration",
                    "content": "What's the distribution of values in the key columns?",
                    "priority": 60
                }
            ],
            "join": [
                {
                    "suggestion_type": "question",
                    "content": "What columns should I use to join these tables?",
                    "priority": 80
                }
            ],
            "duplicate": [
                {
                    "suggestion_type": "improvement",
                    "content": "How should I handle duplicate records?",
                    "priority": 75
                }
            ],
            "clean": [
                {
                    "suggestion_type": "improvement",
                    "content": "What data cleaning steps do you recommend?",
                    "priority": 85
                }
            ],
            "schema": [
                {
                    "suggestion_type": "improvement",
                    "content": "Can you suggest an optimal database schema?",
                    "priority": 90
                }
            ],
            "migrate": [
                {
                    "suggestion_type": "improvement",
                    "content": "What's the best migration strategy for this data?",
                    "priority": 95
                }
            ]
        }
        
        # Check for keywords in messages
        mentioned_keywords = set()
        for message in recent_messages:
            content = message.get("content", "").lower()
            
            for keyword in keywords:
                if keyword in content and keyword not in mentioned_keywords:
                    suggestions.extend(keywords[keyword])
                    mentioned_keywords.add(keyword)
        
        return suggestions
    
    def _generate_improvement_suggestions(self, files_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate data improvement suggestions.
        
        Args:
            files_context: Files context information
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Generic improvement suggestions
        improvement_suggestions = [
            {
                "suggestion_type": "improvement",
                "content": "How can I improve the data quality of these datasets?",
                "priority": 60
            },
            {
                "suggestion_type": "improvement",
                "content": "What data validation rules should I implement?",
                "priority": 55
            },
            {
                "suggestion_type": "improvement",
                "content": "How should I handle inconsistent formatting?",
                "priority": 50
            }
        ]
        
        suggestions.extend(improvement_suggestions)
        
        return suggestions
    
    def generate_schema_suggestions(self, 
                                  data_preview: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate database schema suggestions.
        
        Args:
            data_preview: Dictionary of data previews
            
        Returns:
            Schema suggestions
        """
        suggestions = {
            "tables": {},
            "relationships": [],
            "general_recommendations": []
        }
        
        # Generate table schemas
        for file_name, df in data_preview.items():
            table_name = os.path.splitext(file_name)[0]
            columns = {}
            
            # Analyze columns
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isna().sum()
                unique_count = df[col].nunique()
                
                # Determine column type
                if "int" in dtype:
                    sql_type = "INTEGER"
                elif "float" in dtype:
                    sql_type = "DECIMAL"
                elif "datetime" in dtype:
                    sql_type = "TIMESTAMP"
                elif "bool" in dtype:
                    sql_type = "BOOLEAN"
                else:
                    # Check if it could be an enum
                    if unique_count <= 10 and unique_count / len(df) < 0.1:
                        unique_values = df[col].dropna().unique().tolist()
                        sql_type = f"ENUM({', '.join([str(v) for v in unique_values])})"
                    elif df[col].fillna('').str.len().max() < 50:
                        sql_type = "VARCHAR(50)"
                    else:
                        sql_type = "TEXT"
                
                # Check if it could be a primary key
                is_pk = col.lower() == 'id' or col.lower().endswith('_id')
                if is_pk and unique_count == len(df) - null_count and null_count == 0:
                    is_pk = True
                    sql_type += " PRIMARY KEY"
                else:
                    is_pk = False
                
                columns[col] = {
                    "type": sql_type,
                    "nullable": null_count > 0,
                    "unique_values": unique_count,
                    "is_primary_key": is_pk
                }
            
            suggestions["tables"][table_name] = {
                "columns": columns,
                "recommended_indices": []
            }
            
            # Recommend indices for columns that might be foreign keys
            for col in df.columns:
                if col.lower().endswith('_id') and col.lower() != 'id':
                    suggestions["tables"][table_name]["recommended_indices"].append(col)
        
        # Detect potential relationships
        if len(data_preview) > 1:
            suggestions["relationships"] = self._detect_relationships(data_preview)
        
        # Add general recommendations
        suggestions["general_recommendations"] = [
            "Consider normalizing your database schema to reduce data redundancy",
            "Ensure consistent naming conventions for tables and columns",
            "Implement appropriate foreign key constraints for data integrity"
        ]
        
        return suggestions
    
    def _detect_relationships(self, data_preview: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Detect potential relationships between dataframes.
        
        Args:
            data_preview: Dictionary of dataframes
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        table_names = list(data_preview.keys())
        
        for i in range(len(table_names)):
            for j in range(len(table_names)):
                if i != j:  # Don't compare with self
                    df1 = data_preview[table_names[i]]
                    df2 = data_preview[table_names[j]]
                    
                    # Check for primary key columns in df1 that might be foreign keys in df2
                    for col1 in df1.columns:
                        if col1.lower() == 'id' or col1.lower().endswith('_id'):
                            # Check if this column exists in df2, or a similar one
                            for col2 in df2.columns:
                                if col2 == col1 or (col2.lower().endswith('_id') and 
                                                   col1.replace('_id', '') in col2.lower()):
                                    # Check overlap
                                    if df2[col2].isin(df1[col1]).any():
                                        relationships.append({
                                            "source_table": table_names[i],
                                            "source_column": col1,
                                            "target_table": table_names[j],
                                            "target_column": col2,
                                            "relationship_type": "one-to-many",
                                            "confidence": 0.8
                                        })
        
        return relationships
    
    def generate_cleaning_recommendations(self, 
                                        data_preview: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Generate data cleaning recommendations.
        
        Args:
            data_preview: Dictionary of data previews
            
        Returns:
            List of cleaning recommendations
        """
        recommendations = []
        
        for file_name, df in data_preview.items():
            file_recommendations = {"file": file_name, "recommendations": []}
            
            # Check for missing values
            null_cols = df.columns[df.isna().any()].tolist()
            if null_cols:
                file_recommendations["recommendations"].append({
                    "type": "missing_values",
                    "description": f"Found {len(null_cols)} columns with missing values",
                    "affected_columns": null_cols,
                    "suggestion": "Consider imputation or filtering rows with missing values"
                })
            
            # Check for duplicates
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                file_recommendations["recommendations"].append({
                    "type": "duplicates",
                    "description": f"Found {dup_count} duplicate rows",
                    "suggestion": "Consider removing duplicates or investigating why they exist"
                })
            
            # Check for inconsistent formatting in string columns
            string_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in string_cols:
                # Check for mixed case
                if df[col].str.islower().any() and df[col].str.isupper().any() and df[col].str.istitle().any():
                    file_recommendations["recommendations"].append({
                        "type": "inconsistent_formatting",
                        "description": f"Column '{col}' has inconsistent casing",
                        "affected_columns": [col],
                        "suggestion": "Standardize casing (e.g., all lowercase or title case)"
                    })
            
            # Check for potential date columns that are stored as strings
            for col in string_cols:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        pd.to_datetime(df[col])
                        file_recommendations["recommendations"].append({
                            "type": "incorrect_data_type",
                            "description": f"Column '{col}' appears to be a date but is stored as string",
                            "affected_columns": [col],
                            "suggestion": "Convert to datetime type for proper date operations"
                        })
                    except:
                        pass
            
            # Only add this file if it has recommendations
            if file_recommendations["recommendations"]:
                recommendations.append(file_recommendations)
        
        return recommendations
    
    def generate_migration_recommendations(self, 
                                         data_preview: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate migration strategy recommendations.
        
        Args:
            data_preview: Dictionary of data previews
            
        Returns:
            Migration recommendations
        """
        # Analyze the data to determine appropriate migration strategies
        total_rows = sum(len(df) for df in data_preview.values())
        max_columns = max(len(df.columns) for df in data_preview.values())
        has_complex_types = any(len(df.select_dtypes(include=['object']).columns) > 0 for df in data_preview.values())
        table_count = len(data_preview)
        has_relationships = table_count > 1
        
        # Determine recommendation strategy
        strategy = "undetermined"
        
        if total_rows > 1000000:  # More than a million rows
            if has_relationships:
                strategy = "relational"
            else:
                strategy = "big_data"
        elif has_complex_types and table_count <= 2:
            strategy = "document"
        elif table_count > 5 and has_relationships:
            strategy = "relational"
        else:
            strategy = "simple_relational"
        
        # Generate recommendations based on strategy
        recommendations = {
            "strategy": strategy,
            "explanation": "",
            "target_technologies": [],
            "migration_steps": []
        }
        
        if strategy == "relational":
            recommendations["explanation"] = "Your data has multiple tables with clear relationships, making a relational database the best choice."
            recommendations["target_technologies"] = ["PostgreSQL", "MySQL", "SQL Server"]
            recommendations["migration_steps"] = [
                "Create normalized schema based on detected relationships",
                "Set up appropriate indexes for common query patterns",
                "Implement foreign key constraints for data integrity",
                "Migrate data table by table, starting with tables that don't have foreign key dependencies",
                "Validate referential integrity after migration"
            ]
        
        elif strategy == "big_data":
            recommendations["explanation"] = "Your data is large and might benefit from big data technologies for processing."
            recommendations["target_technologies"] = ["Apache Spark", "Amazon Redshift", "Google BigQuery"]
            recommendations["migration_steps"] = [
                "Partition data by appropriate dimensions (e.g., date)",
                "Consider columnar storage format for efficient querying",
                "Implement appropriate distribution strategy for parallel processing",
                "Set up data pipeline for incremental updates",
                "Optimize schema for analytical queries"
            ]
        
        elif strategy == "document":
            recommendations["explanation"] = "Your data has complex types and fewer relationships, making a document database a good fit."
            recommendations["target_technologies"] = ["MongoDB", "Couchbase", "Amazon DocumentDB"]
            recommendations["migration_steps"] = [
                "Design document schema that captures natural data hierarchy",
                "Denormalize data where appropriate for query efficiency",
                "Create appropriate indexes for common query patterns",
                "Implement validation rules for document structure",
                "Set up sharding strategy if data volume is expected to grow"
            ]
        
        else:  # simple_relational
            recommendations["explanation"] = "Your data is relatively simple with a small number of tables. A straightforward relational database is sufficient."
            recommendations["target_technologies"] = ["SQLite", "PostgreSQL", "MySQL"]
            recommendations["migration_steps"] = [
                "Create tables with appropriate primary keys",
                "Set up basic indexes for frequently queried columns",
                "Migrate data with simple ETL process",
                "Implement basic data validation rules",
                "Document schema for future reference"
            ]
        
        return recommendations
