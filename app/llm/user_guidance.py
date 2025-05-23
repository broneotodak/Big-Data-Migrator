"""
User guidance system for data-driven LLM conversations.

This module provides intelligent guidance during data conversations,
including question generation, exploration suggestions, and explanations.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field

from app.utils.logging_config import get_logger
from app.llm.lm_studio_client import LMStudioClient
from app.llm.data_context_builder import DataFileContext, RelationshipInfo

logger = get_logger(__name__)

@dataclass
class GuidanceSuggestion:
    """A suggestion for user guidance."""
    suggestion_type: str  # "question", "exploration", "improvement", "explanation"
    content: str
    context: Optional[Dict[str, Any]] = None
    priority: int = 0  # Higher number = higher priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suggestion_type": self.suggestion_type,
            "content": self.content,
            "context": self.context,
            "priority": self.priority
        }

class UserGuidanceSystem:
    """
    Provides intelligent guidance during data conversations.
    
    This class:
    - Generates intelligent questions about unclear data relationships
    - Provides data exploration suggestions
    - Offers schema optimization recommendations
    - Guides users through data preparation steps
    - Explains complex data patterns in simple terms
    """
    
    def __init__(self, llm_client: LMStudioClient):
        """
        Initialize the user guidance system.
        
        Args:
            llm_client: LM Studio client for generating guidance
        """
        self.llm_client = llm_client
        logger.info("UserGuidanceSystem initialized")
    
    def generate_questions(self, 
                          data_contexts: Dict[str, Any], 
                          conversation_history: List[Dict[str, str]],
                          max_questions: int = 3) -> List[GuidanceSuggestion]:
        """
        Generate intelligent questions about unclear relationships or data issues.
        
        Args:
            data_contexts: Data context information
            conversation_history: Conversation history
            max_questions: Maximum number of questions to generate
            
        Returns:
            List of question suggestions
        """
        questions = []
        
        # Extract file contexts and relationships
        file_contexts = data_contexts.get("files", {})
        relationships = data_contexts.get("relationships", [])
        
        # Check for ambiguous relationships
        if len(file_contexts) > 1:
            # Look for low-confidence relationships
            ambiguous_relations = [r for r in relationships if isinstance(r, dict) and 0.3 < r.get("confidence", 0) < 0.7]
            
            if ambiguous_relations:
                # Generate questions about ambiguous relationships
                for relation in ambiguous_relations[:max_questions]:
                    source_file = os.path.basename(relation.get("source_file", ""))
                    target_file = os.path.basename(relation.get("target_file", ""))
                    source_col = relation.get("source_column", "")
                    target_col = relation.get("target_column", "")
                    
                    question = (
                        f"I noticed a potential relationship between {source_file}.{source_col} and "
                        f"{target_file}.{target_col}, but I'm not certain. Can you confirm if these columns "
                        f"should be used to join these datasets?"
                    )
                    
                    questions.append(GuidanceSuggestion(
                        suggestion_type="question",
                        content=question,
                        context={
                            "relationship": relation
                        },
                        priority=70
                    ))
        
        # Check for data quality issues
        for file_path, file_info in file_contexts.items():
            file_name = os.path.basename(file_path)
            
            # Check for data quality issues
            data_quality = file_info.get("data_quality", {})
            issues = data_quality.get("issues", [])
            
            if "high_missing_data" in issues:
                question = f"I noticed that {file_name} has a high percentage of missing values. How would you like to handle these missing values?"
                
                questions.append(GuidanceSuggestion(
                    suggestion_type="question",
                    content=question,
                    context={
                        "file": file_name,
                        "issue": "high_missing_data"
                    },
                    priority=80
                ))
            
            if "inconsistent_types" in issues:
                question = f"There appear to be inconsistent data types in {file_name}. Would you like me to suggest a data cleaning approach?"
                
                questions.append(GuidanceSuggestion(
                    suggestion_type="question",
                    content=question,
                    context={
                        "file": file_name,
                        "issue": "inconsistent_types"
                    },
                    priority=85
                ))
        
        # Check if we have enough questions
        if len(questions) < max_questions:
            # Generate general exploratory questions
            if len(file_contexts) == 1:
                # Single file questions
                file_path = next(iter(file_contexts))
                file_name = os.path.basename(file_path)
                file_info = file_contexts[file_path]
                
                column_names = file_info.get("column_names", [])
                if column_names:
                    question = f"What specific insights are you looking to gain from {file_name}?"
                    
                    questions.append(GuidanceSuggestion(
                        suggestion_type="question",
                        content=question,
                        context={
                            "file": file_name
                        },
                        priority=50
                    ))
            
            elif len(file_contexts) > 1:
                # Multiple file questions
                if not relationships:
                    question = "These files don't appear to have obvious relationships. What connections between them are you interested in exploring?"
                    
                    questions.append(GuidanceSuggestion(
                        suggestion_type="question",
                        content=question,
                        priority=60
                    ))
                else:
                    question = "Would you like to analyze these datasets individually or perform a combined analysis?"
                    
                    questions.append(GuidanceSuggestion(
                        suggestion_type="question",
                        content=question,
                        priority=55
                    ))
        
        # Sort by priority (higher first) and limit
        questions.sort(key=lambda q: q.priority, reverse=True)
        return questions[:max_questions]
    
    def generate_exploration_suggestions(self, 
                                       data_contexts: Dict[str, Any],
                                       conversation_history: List[Dict[str, str]], 
                                       max_suggestions: int = 3) -> List[GuidanceSuggestion]:
        """
        Generate suggestions for data exploration.
        
        Args:
            data_contexts: Data context information
            conversation_history: Conversation history
            max_suggestions: Maximum number of suggestions to generate
            
        Returns:
            List of exploration suggestions
        """
        suggestions = []
        
        # Extract file contexts
        file_contexts = data_contexts.get("files", {})
        
        # Check if we have any files
        if not file_contexts:
            return suggestions
            
        # Single file suggestions
        if len(file_contexts) == 1:
            file_path = next(iter(file_contexts))
            file_name = os.path.basename(file_path)
            file_info = file_contexts[file_path]
            
            # Column analysis suggestion
            column_stats = file_info.get("statistics", {}).get("columns", {})
            numeric_columns = [col for col, stats in column_stats.items() 
                              if stats.get("type", "").startswith(("int", "float"))]
            
            if numeric_columns:
                suggestion = (
                    f"Explore descriptive statistics and distributions of key numeric columns in {file_name}: "
                    f"{', '.join(numeric_columns[:3])}"
                )
                
                suggestions.append(GuidanceSuggestion(
                    suggestion_type="exploration",
                    content=suggestion,
                    context={
                        "file": file_name,
                        "columns": numeric_columns[:3],
                        "action": "analyze_distribution"
                    },
                    priority=70
                ))
            
            # Categorical analysis
            categorical_columns = []
            for col, stats in column_stats.items():
                if stats.get("appears_categorical") or (stats.get("unique_ratio", 1.0) < 0.2):
                    categorical_columns.append(col)
            
            if categorical_columns:
                suggestion = (
                    f"Examine frequency distribution of categorical variables in {file_name}: "
                    f"{', '.join(categorical_columns[:3])}"
                )
                
                suggestions.append(GuidanceSuggestion(
                    suggestion_type="exploration",
                    content=suggestion,
                    context={
                        "file": file_name,
                        "columns": categorical_columns[:3],
                        "action": "analyze_categories"
                    },
                    priority=65
                ))
            
            # Correlation analysis
            if len(numeric_columns) > 1:
                suggestion = (
                    f"Analyze correlations between numeric variables in {file_name} "
                    f"to identify potential relationships"
                )
                
                suggestions.append(GuidanceSuggestion(
                    suggestion_type="exploration",
                    content=suggestion,
                    context={
                        "file": file_name,
                        "columns": numeric_columns,
                        "action": "analyze_correlation"
                    },
                    priority=75
                ))
        
        # Multiple files suggestions
        elif len(file_contexts) > 1:
            # Get file names
            file_names = [os.path.basename(path) for path in file_contexts.keys()]
            
            # Relationship exploration
            suggestion = (
                f"Explore relationships between {file_names[0]} and {file_names[1]} "
                f"to identify potential join opportunities"
            )
            
            suggestions.append(GuidanceSuggestion(
                suggestion_type="exploration",
                content=suggestion,
                context={
                    "files": file_names[:2],
                    "action": "explore_relationships"
                },
                priority=80
            ))
            
            # Combined analysis
            suggestion = (
                f"Perform a combined analysis of all datasets to gain comprehensive insights"
            )
            
            suggestions.append(GuidanceSuggestion(
                suggestion_type="exploration",
                content=suggestion,
                context={
                    "files": file_names,
                    "action": "combined_analysis"
                },
                priority=75
            ))
        
        # Additional exploratory suggestions
        suggestion = "Identify outliers and anomalies in the data that may affect analysis"
        
        suggestions.append(GuidanceSuggestion(
            suggestion_type="exploration",
            content=suggestion,
            context={
                "action": "identify_outliers"
            },
            priority=60
        ))
        
        # Sort by priority and limit
        suggestions.sort(key=lambda s: s.priority, reverse=True)
        return suggestions[:max_suggestions]
    
    def generate_schema_recommendations(self, 
                                      data_contexts: Dict[str, Any],
                                      max_recommendations: int = 3) -> List[GuidanceSuggestion]:
        """
        Generate schema optimization recommendations.
        
        Args:
            data_contexts: Data context information
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of schema recommendations
        """
        recommendations = []
        
        # Extract file contexts
        file_contexts = data_contexts.get("files", {})
        relationships = data_contexts.get("relationships", [])
        
        # Check data quality issues first
        for file_path, file_info in file_contexts.items():
            file_name = os.path.basename(file_path)
            data_quality = file_info.get("data_quality", {})
            column_issues = data_quality.get("column_issues", {})
            
            # Check for duplicate columns
            duplicate_cols = []
            for col, issues in column_issues.items():
                for issue in issues:
                    if issue.startswith("duplicate_of_"):
                        other_col = issue.replace("duplicate_of_", "")
                        duplicate_cols.append((col, other_col))
            
            if duplicate_cols:
                cols_str = ", ".join([f"{a} and {b}" for a, b in duplicate_cols[:2]])
                if len(duplicate_cols) > 2:
                    cols_str += f", and {len(duplicate_cols) - 2} more pairs"
                
                recommendation = (
                    f"Consider removing duplicate columns in {file_name}: {cols_str}. "
                    f"This will improve data quality and reduce storage requirements."
                )
                
                recommendations.append(GuidanceSuggestion(
                    suggestion_type="improvement",
                    content=recommendation,
                    context={
                        "file": file_name,
                        "duplicate_columns": duplicate_cols,
                        "action": "remove_duplicates"
                    },
                    priority=85
                ))
            
            # Check for columns with high missing values
            high_missing_cols = []
            for col, issues in column_issues.items():
                for issue in issues:
                    if issue.startswith("high_missing_values"):
                        high_missing_cols.append(col)
            
            if high_missing_cols and len(high_missing_cols) > 2:
                cols_str = f"{high_missing_cols[0]}, {high_missing_cols[1]}, and {len(high_missing_cols) - 2} more"
                
                recommendation = (
                    f"Consider handling missing values in {file_name} columns: {cols_str}. "
                    f"Options include imputation, removal, or using algorithms that handle missing values."
                )
                
                recommendations.append(GuidanceSuggestion(
                    suggestion_type="improvement",
                    content=recommendation,
                    context={
                        "file": file_name,
                        "columns": high_missing_cols,
                        "action": "handle_missing_values"
                    },
                    priority=80
                ))
            
            # Type consistency issues
            inconsistent_type_cols = []
            for col, issues in column_issues.items():
                if "inconsistent_types" in issues:
                    inconsistent_type_cols.append(col)
            
            if inconsistent_type_cols:
                cols_str = ", ".join(inconsistent_type_cols[:3])
                if len(inconsistent_type_cols) > 3:
                    cols_str += f", and {len(inconsistent_type_cols) - 3} more"
                
                recommendation = (
                    f"Fix data type inconsistencies in {file_name} for columns: {cols_str}. "
                    f"Ensure values are consistently formatted to improve data quality."
                )
                
                recommendations.append(GuidanceSuggestion(
                    suggestion_type="improvement",
                    content=recommendation,
                    context={
                        "file": file_name,
                        "columns": inconsistent_type_cols,
                        "action": "fix_data_types"
                    },
                    priority=85
                ))
        
        # Relationship-based recommendations
        if relationships and len(file_contexts) > 1:
            # Suggest database schema
            file_names = [os.path.basename(path) for path in file_contexts.keys()]
            
            recommendation = (
                f"Consider organizing these datasets into a structured database schema with "
                f"proper foreign key relationships between tables: {', '.join(file_names)}"
            )
            
            recommendations.append(GuidanceSuggestion(
                suggestion_type="improvement",
                content=recommendation,
                context={
                    "files": file_names,
                    "relationships": relationships,
                    "action": "create_database_schema"
                },
                priority=75
            ))
        
        # Sort by priority and limit
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        return recommendations[:max_recommendations]
    
    def generate_preparation_steps(self, 
                                 data_contexts: Dict[str, Any],
                                 analysis_goal: Optional[str] = None,
                                 max_steps: int = 5) -> List[Dict[str, Any]]:
        """
        Generate data preparation steps based on analysis goals.
        
        Args:
            data_contexts: Data context information
            analysis_goal: Optional description of analysis goal
            max_steps: Maximum number of steps to generate
            
        Returns:
            List of preparation step dictionaries
        """
        steps = []
        
        # Extract file contexts and issues
        file_contexts = data_contexts.get("files", {})
        all_issues = set()
        file_issues = {}
        
        for file_path, file_info in file_contexts.items():
            file_name = os.path.basename(file_path)
            issues = file_info.get("data_quality", {}).get("issues", [])
            
            all_issues.update(issues)
            file_issues[file_name] = issues
        
        # Define standard preparation steps
        if "high_missing_data" in all_issues:
            steps.append({
                "step": 1,
                "title": "Handle Missing Values",
                "description": "Identify and address missing values by imputation or filtering.",
                "details": "For numerical columns, consider mean/median imputation. For categorical columns, consider mode imputation or a 'Missing' category. Remove rows only if missing data is minimal.",
                "priority": 90
            })
        
        if "inconsistent_types" in all_issues:
            steps.append({
                "step": 2,
                "title": "Fix Data Type Inconsistencies",
                "description": "Ensure consistent data types across columns.",
                "details": "Convert string representations to appropriate numeric or date types. Handle special characters and inconsistent formatting.",
                "priority": 95
            })
        
        # Multiple file preparation
        if len(file_contexts) > 1:
            steps.append({
                "step": 3,
                "title": "Join/Merge Datasets",
                "description": "Combine related datasets based on key relationships.",
                "details": "Identify appropriate join columns and merge types (inner, left, right, outer) based on analysis requirements.",
                "priority": 85
            })
        
        # Feature engineering
        if analysis_goal and ("predict" in analysis_goal.lower() or "forecast" in analysis_goal.lower()):
            steps.append({
                "step": 4,
                "title": "Feature Engineering",
                "description": "Create derived features to improve predictive power.",
                "details": "Generate features from existing columns, create interaction terms, and transform skewed distributions.",
                "priority": 80
            })
        
        # Always include standardization
        steps.append({
            "step": 5,
            "title": "Standardize and Normalize",
            "description": "Scale numerical features to improve analysis quality.",
            "details": "Apply standardization (z-score) or normalization (min-max scaling) to numeric columns as appropriate for the intended analysis.",
            "priority": 70
        })
        
        # Outlier handling
        steps.append({
            "step": 6,
            "title": "Handle Outliers",
            "description": "Identify and address extreme values.",
            "details": "Use statistical methods (z-score, IQR) to detect outliers and apply capping, removal, or transformation as appropriate.",
            "priority": 75
        })
        
        # If we know the analysis goal, add more specific steps
        if analysis_goal:
            if "classification" in analysis_goal.lower():
                steps.append({
                    "step": 7,
                    "title": "Balance Classes",
                    "description": "Address class imbalance for classification tasks.",
                    "details": "Apply oversampling, undersampling, or synthetic data generation techniques to balance target classes.",
                    "priority": 78
                })
            
            if "time series" in analysis_goal.lower():
                steps.append({
                    "step": 8,
                    "title": "Time Series Preprocessing",
                    "description": "Prepare data for time series analysis.",
                    "details": "Handle seasonality, trends, and ensure equally spaced time intervals. Create lag features and rolling statistics.",
                    "priority": 85
                })
        
        # Sort by priority and assign step numbers
        steps.sort(key=lambda s: s.get("priority", 0), reverse=True)
        for i, step in enumerate(steps[:max_steps]):
            step["step"] = i + 1
            
        return steps[:max_steps]
    
    def explain_data_patterns(self, 
                            pattern_description: str,
                            data_contexts: Dict[str, Any]) -> str:
        """
        Generate a simple explanation of complex data patterns.
        
        Args:
            pattern_description: Description of the pattern to explain
            data_contexts: Data context information
            
        Returns:
            User-friendly explanation
        """
        # Create a prompt for the LLM to explain the pattern
        system_prompt = (
            "You are a helpful data assistant that explains complex data patterns in simple terms. "
            "Provide clear, concise explanations using plain language while maintaining technical accuracy. "
            "Use analogies when appropriate. Limit explanations to 3-4 sentences."
        )
        
        # Extract relevant context
        context_summary = data_contexts.get("summary", "")
        
        user_prompt = f"""
        Please explain this data pattern in simple terms: "{pattern_description}"
        
        Context about the data:
        {context_summary}
        """
        
        # Generate explanation using LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            explanation = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.3  # Lower temperature for more consistent explanations
            )
            
            return explanation
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return f"I couldn't generate an explanation at this time. The pattern refers to {pattern_description}."
    
    def generate_guidance(self, 
                        data_contexts: Dict[str, Any], 
                        conversation_history: List[Dict[str, str]],
                        analysis_goal: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive guidance based on data context and conversation.
        
        Args:
            data_contexts: Data context information
            conversation_history: Conversation history
            analysis_goal: Optional description of analysis goal
            
        Returns:
            Dictionary with various guidance elements
        """
        guidance = {
            "questions": [],
            "exploration_suggestions": [],
            "schema_recommendations": [],
            "preparation_steps": []
        }
        
        # Generate questions
        questions = self.generate_questions(data_contexts, conversation_history)
        guidance["questions"] = [q.to_dict() for q in questions]
        
        # Generate exploration suggestions
        suggestions = self.generate_exploration_suggestions(data_contexts, conversation_history)
        guidance["exploration_suggestions"] = [s.to_dict() for s in suggestions]
        
        # Generate schema recommendations
        recommendations = self.generate_schema_recommendations(data_contexts)
        guidance["schema_recommendations"] = [r.to_dict() for r in recommendations]
        
        # Generate preparation steps
        steps = self.generate_preparation_steps(data_contexts, analysis_goal)
        guidance["preparation_steps"] = steps
        
        return guidance
