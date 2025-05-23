"""
Data context builder for creating intelligent data summaries and insights.

This module creates rich data context for LLM conversations by analyzing
data files, generating statistics, and surfacing relationships.
"""
import os
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from collections import defaultdict

from app.utils.logging_config import get_logger
from app.memory.memory_monitor import MemoryMonitor
from app.memory.resource_optimizer import ResourceOptimizer
from app.processors.base_processor import BaseProcessor
from app.processors.csv_processor import LargeCSVProcessor
from app.processors.excel_processor import ExcelProcessor

logger = get_logger(__name__)

@dataclass
class DataFileContext:
    """Context information about a data file."""
    file_path: str
    file_name: str
    file_type: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    column_names: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    sample_data: Optional[pd.DataFrame] = None
    sample_rows: int = 10
    data_quality: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_names": self.column_names,
            "column_types": self.column_types,
            "statistics": self.statistics,
            "sample_rows": self.sample_rows,
            "data_quality": self.data_quality
        }
        
        # Convert sample data to list for serialization
        if self.sample_data is not None:
            result["sample_data"] = self.sample_data.to_dict(orient='records')
        
        return result

@dataclass
class RelationshipInfo:
    """Information about a detected relationship between data files."""
    source_file: str
    target_file: str
    source_column: str
    target_column: str
    relationship_type: str  # "one-to-one", "one-to-many", "many-to-one", "many-to-many"
    confidence: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_file": self.source_file,
            "target_file": self.target_file,
            "source_column": self.source_column,
            "target_column": self.target_column,
            "relationship_type": self.relationship_type,
            "confidence": self.confidence
        }

class DataContextBuilder:
    """
    Creates intelligent data summaries and context for LLM discussions.
    
    This class:
    - Analyzes data files to create summaries
    - Generates statistics and samples for context
    - Builds relationship maps and schema suggestions
    - Provides data quality insights
    """
    
    def __init__(self, 
                memory_monitor: Optional[MemoryMonitor] = None,
                resource_optimizer: Optional[ResourceOptimizer] = None,
                temp_dir: Optional[str] = None):
        """
        Initialize the data context builder.
        
        Args:
            memory_monitor: Optional memory monitor instance
            resource_optimizer: Optional resource optimizer instance
            temp_dir: Directory for temporary files
        """
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.resource_optimizer = resource_optimizer or ResourceOptimizer(self.memory_monitor)
        self.temp_dir = temp_dir or os.path.join(os.getcwd(), 'tmp')
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize processors
        self.processors = {
            'csv': LargeCSVProcessor(memory_monitor=self.memory_monitor, resource_optimizer=self.resource_optimizer),
            'excel': ExcelProcessor(memory_monitor=self.memory_monitor, resource_optimizer=self.resource_optimizer)
        }
        
        logger.info("DataContextBuilder initialized")
    
    def build_context_for_files(self, file_paths: List[str], 
                               include_stats: bool = True,
                               include_samples: bool = True,
                               include_relationships: bool = True,
                               include_quality: bool = True) -> Dict[str, Any]:
        """
        Build comprehensive context for a list of data files.
        
        Args:
            file_paths: List of file paths to analyze
            include_stats: Whether to include statistics
            include_samples: Whether to include data samples
            include_relationships: Whether to try to detect relationships
            include_quality: Whether to include data quality metrics
            
        Returns:
            Dictionary with comprehensive data context
        """
        self.memory_monitor.start_tracking_step("build_data_context")
        
        try:
            # Process each file
            file_contexts = {}
            
            for file_path in file_paths:
                try:
                    context = self.analyze_file(file_path, include_stats, include_samples, include_quality)
                    file_contexts[file_path] = context
                except Exception as e:
                    logger.error(f"Error analyzing file {file_path}: {str(e)}")
            
            # Detect relationships if requested
            relationships = []
            if include_relationships and len(file_contexts) > 1:
                relationships = self.detect_relationships(file_contexts)
            
            # Build recommendations
            recommendations = self.generate_recommendations(file_contexts, relationships)
            
            # Create full context
            result = {
                "files": {path: context.to_dict() for path, context in file_contexts.items()},
                "relationships": [r.to_dict() for r in relationships],
                "recommendations": recommendations,
                "summary": self.create_context_summary(file_contexts, relationships)
            }
            
            return result
            
        finally:
            self.memory_monitor.end_tracking_step("build_data_context")
            
    def analyze_file(self, file_path: str, 
                    include_stats: bool = True,
                    include_samples: bool = True,
                    include_quality: bool = True) -> DataFileContext:
        """
        Analyze a single data file to build context.
        
        Args:
            file_path: Path to the file to analyze
            include_stats: Whether to include statistics
            include_samples: Whether to include data samples
            include_quality: Whether to include data quality metrics
            
        Returns:
            DataFileContext with analysis results
        """
        # Get file info
        file_name = os.path.basename(file_path)
        _, file_ext = os.path.splitext(file_name)
        file_type = file_ext.lower().lstrip('.')
        
        # Create context object
        context = DataFileContext(
            file_path=file_path,
            file_name=file_name,
            file_type=file_type
        )
        
        # Get appropriate processor
        processor = self._get_processor_for_file(file_path)
        if not processor:
            logger.warning(f"No processor available for file type: {file_type}")
            return context
            
        # Get file info using processor
        file_info = processor.get_file_info(file_path)
        
        # Fill basic info
        context.row_count = file_info.get("row_count")
        context.column_count = file_info.get("column_count")
        context.column_names = file_info.get("columns", [])
        
        # Generate sample if requested
        if include_samples:
            context.sample_data = self._get_sample_data(file_path, processor)
            
            # Infer column types from sample
            if context.sample_data is not None:
                context.column_types = {
                    col: str(context.sample_data[col].dtype) 
                    for col in context.sample_data.columns
                }
        
        # Generate statistics if requested
        if include_stats and context.sample_data is not None:
            context.statistics = self._compute_statistics(context.sample_data, file_info)
        
        # Assess data quality if requested
        if include_quality and context.sample_data is not None:
            context.data_quality = self._assess_data_quality(context.sample_data, file_info)
        
        return context
    
    def detect_relationships(self, file_contexts: Dict[str, DataFileContext]) -> List[RelationshipInfo]:
        """
        Detect potential relationships between data files.
        
        Args:
            file_contexts: Dictionary of file paths to context objects
            
        Returns:
            List of detected relationships
        """
        relationships = []
        
        # We need at least 2 files to detect relationships
        if len(file_contexts) < 2:
            return relationships
            
        # Get list of files
        files = list(file_contexts.keys())
        
        # Check each file pair
        for i, source_file in enumerate(files):
            source_context = file_contexts[source_file]
            
            # Skip if no sample data
            if source_context.sample_data is None:
                continue
                
            # Get source columns
            source_cols = set(source_context.column_names)
            
            # Check against other files
            for j in range(i + 1, len(files)):
                target_file = files[j]
                target_context = file_contexts[target_file]
                
                # Skip if no sample data
                if target_context.sample_data is None:
                    continue
                    
                # Find potential join columns based on name similarity
                potential_joins = self._find_potential_join_columns(source_context, target_context)
                
                # Analyze each potential join
                for source_col, target_col in potential_joins:
                    try:
                        relationship = self._analyze_column_relationship(
                            source_context, target_context, source_col, target_col)
                        
                        if relationship:
                            relationships.append(relationship)
                            
                    except Exception as e:
                        logger.error(f"Error analyzing relationship between {source_file}.{source_col} and {target_file}.{target_col}: {str(e)}")
        
        return relationships
    
    def create_context_summary(self, 
                              file_contexts: Dict[str, DataFileContext],
                              relationships: List[RelationshipInfo]) -> str:
        """
        Create a natural language summary of the data context.
        
        Args:
            file_contexts: Dictionary of file paths to context objects
            relationships: List of detected relationships
            
        Returns:
            String summary of the data context
        """
        summary = []
        
        # Summarize files
        summary.append(f"Data context contains {len(file_contexts)} file(s):")
        
        for path, context in file_contexts.items():
            file_desc = f"- {context.file_name}: "
            
            if context.row_count and context.column_count:
                file_desc += f"{context.row_count:,} rows × {context.column_count} columns. "
            
            # Add key columns
            if context.column_names:
                key_cols = context.column_names[:5]
                if len(context.column_names) > 5:
                    key_cols_str = ', '.join(key_cols) + f", and {len(context.column_names) - 5} more"
                else:
                    key_cols_str = ', '.join(key_cols)
                file_desc += f"Columns: {key_cols_str}. "
            
            # Add notable quality issues if any
            if context.data_quality and 'issues' in context.data_quality and context.data_quality['issues']:
                issues = context.data_quality['issues']
                if len(issues) > 2:
                    file_desc += f"Has {len(issues)} quality issues. "
                elif issues:
                    file_desc += f"Quality issues: {', '.join(issues[:2])}. "
            
            summary.append(file_desc)
        
        # Summarize relationships
        if relationships:
            summary.append(f"\nDetected {len(relationships)} potential relationships:")
            
            for rel in relationships:
                source_name = os.path.basename(rel.source_file)
                target_name = os.path.basename(rel.target_file)
                
                rel_desc = (f"- {rel.relationship_type} relationship between {source_name}.{rel.source_column} "
                           f"and {target_name}.{rel.target_column} (confidence: {rel.confidence:.2f})")
                summary.append(rel_desc)
        
        return "\n".join(summary)
    
    def generate_recommendations(self, 
                               file_contexts: Dict[str, DataFileContext],
                               relationships: List[RelationshipInfo]) -> List[str]:
        """
        Generate recommendations based on data analysis.
        
        Args:
            file_contexts: Dictionary of file paths to context objects
            relationships: List of detected relationships
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check for quality issues across all files
        quality_issues = False
        missing_data_files = []
        inconsistent_types = []
        
        for path, context in file_contexts.items():
            if context.data_quality and 'issues' in context.data_quality and context.data_quality['issues']:
                quality_issues = True
                
                # Check for high missing data
                if 'high_missing_data' in context.data_quality['issues']:
                    missing_data_files.append(context.file_name)
                
                # Check for inconsistent types
                if 'inconsistent_types' in context.data_quality['issues']:
                    inconsistent_types.append(context.file_name)
        
        # Add recommendations based on quality issues
        if missing_data_files:
            if len(missing_data_files) == 1:
                recommendations.append(f"Consider handling missing data in {missing_data_files[0]} before analysis.")
            else:
                recommendations.append(f"Missing data issues detected in {len(missing_data_files)} files. Consider implementing a consistent handling strategy.")
        
        if inconsistent_types:
            if len(inconsistent_types) == 1:
                recommendations.append(f"Fix data type inconsistencies in {inconsistent_types[0]}.")
            else:
                recommendations.append(f"Data type inconsistencies found in {len(inconsistent_types)} files. Consider standardizing data types.")
        
        # Recommendations based on relationships
        if relationships:
            # Check for potential join opportunities
            join_opportunities = defaultdict(list)
            
            for rel in relationships:
                source_name = os.path.basename(rel.source_file)
                target_name = os.path.basename(rel.target_file)
                
                if rel.confidence > 0.7:  # Only high confidence joins
                    key = f"{source_name} ↔ {target_name}"
                    join_opportunities[key].append((rel.source_column, rel.target_column))
            
            if join_opportunities:
                for key, joins in join_opportunities.items():
                    files = key.split(" ↔ ")
                    if len(joins) == 1:
                        source_col, target_col = joins[0]
                        recommendations.append(f"Consider joining {files[0]} and {files[1]} using columns {source_col} and {target_col}.")
                    else:
                        recommendations.append(f"Multiple potential join paths detected between {files[0]} and {files[1]}.")
        
        # General recommendations
        if len(file_contexts) == 1:
            # Single file recommendations
            context = list(file_contexts.values())[0]
            
            # Check for large file
            if context.row_count and context.row_count > 100000:
                recommendations.append("This is a large file. Consider using chunked processing for memory efficiency.")
            
            # Check for many columns
            if context.column_count and context.column_count > 50:
                recommendations.append("This file has many columns. Consider feature selection to focus analysis.")
        
        elif len(file_contexts) > 1:
            # Multiple file recommendations
            if not relationships:
                recommendations.append("Multiple files detected but no clear relationships found. Consider exploring potential connections between datasets.")
        
        return recommendations
    
    def _get_processor_for_file(self, file_path: str) -> Optional[BaseProcessor]:
        """Get appropriate processor for a file."""
        _, ext = os.path.splitext(file_path)
        file_type = ext.lower().lstrip('.')
        
        if file_type in ['csv', 'txt']:
            return self.processors.get('csv')
        elif file_type in ['xlsx', 'xls']:
            return self.processors.get('excel')
        else:
            return None
    
    def _get_sample_data(self, file_path: str, processor: BaseProcessor) -> Optional[pd.DataFrame]:
        """
        Get sample data from a file.
        
        Args:
            file_path: Path to the file
            processor: Appropriate file processor
            
        Returns:
            DataFrame with sample data or None if unable to sample
        """
        try:
            # Sample data using processor
            return processor.sample_data(file_path)
        except Exception as e:
            logger.error(f"Error sampling data from {file_path}: {str(e)}")
            return None
    
    def _compute_statistics(self, df: pd.DataFrame, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute statistics for a dataframe.
        
        Args:
            df: DataFrame to analyze
            file_info: File information from processor
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        try:
            # Basic statistics
            stats["row_count"] = len(df)
            stats["column_count"] = len(df.columns)
            
            # Column-specific stats
            column_stats = {}
            
            for col in df.columns:
                col_stats = {}
                
                # Get data type
                dtype = str(df[col].dtype)
                col_stats["type"] = dtype
                
                # Count unique values
                col_stats["unique_count"] = df[col].nunique()
                col_stats["unique_ratio"] = col_stats["unique_count"] / len(df) if len(df) > 0 else 0
                
                # Missing values
                col_stats["missing_count"] = df[col].isna().sum()
                col_stats["missing_ratio"] = col_stats["missing_count"] / len(df) if len(df) > 0 else 0
                
                # Type-specific statistics
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Numeric column statistics
                    col_stats["min"] = df[col].min() if not df[col].isna().all() else None
                    col_stats["max"] = df[col].max() if not df[col].isna().all() else None
                    col_stats["mean"] = df[col].mean() if not df[col].isna().all() else None
                    col_stats["median"] = df[col].median() if not df[col].isna().all() else None
                    col_stats["std"] = df[col].std() if not df[col].isna().all() else None
                
                elif pd.api.types.is_string_dtype(df[col]):
                    # String column statistics
                    sample_values = df[col].dropna().sample(min(5, len(df[col].dropna()))).tolist() if len(df[col].dropna()) > 0 else []
                    col_stats["sample_values"] = sample_values
                    
                    # Check if looks like categorical
                    if col_stats["unique_ratio"] < 0.1:
                        col_stats["appears_categorical"] = True
                        
                        # Get top categories
                        value_counts = df[col].value_counts(normalize=True).head(5)
                        col_stats["top_categories"] = {
                            str(k): float(v) for k, v in value_counts.items()
                        }
                
                # Store column stats
                column_stats[col] = col_stats
            
            stats["columns"] = column_stats
            
        except Exception as e:
            logger.error(f"Error computing statistics: {str(e)}")
        
        return stats
    
    def _assess_data_quality(self, df: pd.DataFrame, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess data quality for a dataframe.
        
        Args:
            df: DataFrame to analyze
            file_info: File information from processor
            
        Returns:
            Dictionary of data quality metrics and issues
        """
        quality = {
            "score": 0.0,  # Overall score (0-1)
            "metrics": {},  # Specific metrics
            "issues": [],   # List of issues
            "column_issues": {}  # Column-specific issues
        }
        
        try:
            # Calculate completeness (inverse of missing data ratio)
            missing_cells = df.isna().sum().sum()
            total_cells = df.size
            completeness = 1.0 - (missing_cells / total_cells if total_cells > 0 else 0)
            quality["metrics"]["completeness"] = completeness
            
            # Check for high missing data
            if completeness < 0.9:
                quality["issues"].append("high_missing_data")
                
            # Check columns for specific issues
            column_issues = {}
            
            for col in df.columns:
                col_issues = []
                
                # Check for missing values
                missing_ratio = df[col].isna().mean()
                if missing_ratio > 0.1:
                    col_issues.append(f"high_missing_values ({missing_ratio:.1%})")
                
                # Check for unique values
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                
                # Check if numeric column has inconsistent types (e.g., mixed strings and numbers)
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check for any non-numeric strings in original data
                    if 'object' in str(df[col].dtype) or df[col].apply(lambda x: isinstance(x, str)).any():
                        col_issues.append("inconsistent_types")
                        if "inconsistent_types" not in quality["issues"]:
                            quality["issues"].append("inconsistent_types")
                
                # Check for potential duplicate column
                for other_col in df.columns:
                    if col != other_col and df[col].equals(df[other_col]):
                        col_issues.append(f"duplicate_of_{other_col}")
                        break
                
                # Store issues if any
                if col_issues:
                    column_issues[col] = col_issues
            
            quality["column_issues"] = column_issues
            
            # Calculate overall score
            # Start with completeness score
            score = completeness
            
            # Reduce for number of issues
            issue_penalty = min(0.5, len(quality["issues"]) * 0.1)  # Cap at 0.5 reduction
            score = max(0.1, score - issue_penalty)  # Ensure score is at least 0.1
            
            quality["score"] = score
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            quality["issues"].append(f"assessment_error: {str(e)}")
        
        return quality
    
    def _find_potential_join_columns(self, 
                                    source_context: DataFileContext,
                                    target_context: DataFileContext) -> List[Tuple[str, str]]:
        """
        Find potential join columns between two data files.
        
        Args:
            source_context: Context for source file
            target_context: Context for target file
            
        Returns:
            List of (source_column, target_column) tuples
        """
        potential_joins = []
        
        # Skip if no sample data
        if source_context.sample_data is None or target_context.sample_data is None:
            return potential_joins
            
        # Check for exact column name matches
        common_names = set(source_context.column_names) & set(target_context.column_names)
        for col in common_names:
            potential_joins.append((col, col))
        
        # Check for typical join column patterns
        join_keywords = ['id', 'key', 'code', 'num', 'number']
        
        # Find columns with these keywords
        source_id_cols = [col for col in source_context.column_names 
                         if any(keyword in col.lower() for keyword in join_keywords)]
        target_id_cols = [col for col in target_context.column_names 
                         if any(keyword in col.lower() for keyword in join_keywords)]
        
        # Check combinations of potential ID columns
        for source_col in source_id_cols:
            for target_col in target_id_cols:
                # Skip if already added as exact match
                if (source_col, target_col) not in potential_joins:
                    # Check if column types are compatible
                    source_type = str(source_context.sample_data[source_col].dtype)
                    target_type = str(target_context.sample_data[target_col].dtype)
                    
                    # Only suggest compatible types
                    if 'int' in source_type and 'int' in target_type:
                        potential_joins.append((source_col, target_col))
                    elif 'str' in source_type and 'str' in target_type:
                        potential_joins.append((source_col, target_col))
                    elif source_type == target_type:
                        potential_joins.append((source_col, target_col))
        
        return potential_joins
    
    def _analyze_column_relationship(self,
                                   source_context: DataFileContext,
                                   target_context: DataFileContext,
                                   source_column: str,
                                   target_column: str) -> Optional[RelationshipInfo]:
        """
        Analyze the relationship between two columns.
        
        Args:
            source_context: Context for source file
            target_context: Context for target file
            source_column: Name of column in source file
            target_column: Name of column in target file
            
        Returns:
            RelationshipInfo or None if no relationship detected
        """
        # Skip if no sample data
        if source_context.sample_data is None or target_context.sample_data is None:
            return None
            
        # Get values
        source_values = set(source_context.sample_data[source_column].dropna().unique())
        target_values = set(target_context.sample_data[target_column].dropna().unique())
        
        # Skip if empty
        if not source_values or not target_values:
            return None
        
        # Check for overlapping values
        common_values = source_values & target_values
        
        # Calculate overlap ratios
        source_overlap_ratio = len(common_values) / len(source_values) if source_values else 0
        target_overlap_ratio = len(common_values) / len(target_values) if target_values else 0
        
        # Skip if no significant overlap
        if source_overlap_ratio < 0.1 and target_overlap_ratio < 0.1:
            return None
            
        # Determine relationship type
        relationship_type = "unknown"
        
        # Check uniqueness in source and target
        source_unique_ratio = len(source_values) / len(source_context.sample_data)
        target_unique_ratio = len(target_values) / len(target_context.sample_data)
        
        # Determine relationship type based on uniqueness
        if source_unique_ratio > 0.9 and target_unique_ratio > 0.9:
            # Both columns have mostly unique values
            relationship_type = "one-to-one"
        elif source_unique_ratio > 0.9:
            # Source has unique values, target may have duplicates
            relationship_type = "one-to-many"
        elif target_unique_ratio > 0.9:
            # Target has unique values, source may have duplicates
            relationship_type = "many-to-one"
        else:
            # Both have duplicate values
            relationship_type = "many-to-many"
        
        # Calculate confidence based on overlap and uniqueness
        confidence = min(source_overlap_ratio, target_overlap_ratio) * 0.7 + (source_unique_ratio * target_unique_ratio) * 0.3
        
        # Create relationship info
        return RelationshipInfo(
            source_file=source_context.file_path,
            target_file=target_context.file_path,
            source_column=source_column,
            target_column=target_column,
            relationship_type=relationship_type,
            confidence=confidence
        )
