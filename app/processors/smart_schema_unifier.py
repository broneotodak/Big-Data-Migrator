"""
Smart Schema Unifier - Uses LLM for schema inference and programmatic processing for data migration.

This module addresses the core problem: LLMs should infer schemas and relationships, 
but actual data processing should be deterministic and programmatic.
"""
import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib

import pandas as pd
import numpy as np
from tqdm import tqdm

from app.utils.logging_config import get_logger
from app.memory.memory_monitor import MemoryMonitor
from app.processors.multi_file_processor import MultiFileProcessor

logger = get_logger(__name__)

@dataclass
class ColumnMapping:
    """Represents a mapping between source and target columns."""
    source_column: str
    target_column: str
    source_file: str
    data_type: str
    transformation_rule: Optional[str] = None
    confidence_score: float = 1.0
    sample_values: List[Any] = None

@dataclass
class UnifiedSchema:
    """Represents a unified database schema across multiple files."""
    table_name: str
    columns: Dict[str, str]  # column_name -> data_type
    primary_keys: List[str]
    foreign_keys: Dict[str, str]  # column -> referenced_table.column
    indexes: List[str]
    constraints: Dict[str, str]
    file_mappings: Dict[str, List[ColumnMapping]]  # file_path -> column mappings
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class SmartSchemaUnifier:
    """
    Unifies schemas across multiple file formats using LLM intelligence 
    combined with programmatic data processing.
    """
    
    def __init__(self, 
                 llm_system=None,
                 memory_monitor: MemoryMonitor = None,
                 confidence_threshold: float = 0.8,
                 max_sample_rows: int = 100):
        """
        Initialize the Smart Schema Unifier.
        
        Args:
            llm_system: LLM conversation system for schema inference
            memory_monitor: Memory monitoring instance
            confidence_threshold: Minimum confidence for automatic mappings
            max_sample_rows: Maximum rows to sample for schema inference
        """
        self.llm_system = llm_system
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.confidence_threshold = confidence_threshold
        self.max_sample_rows = max_sample_rows
        
        # File processor for reading different formats
        self.file_processor = MultiFileProcessor(
            memory_monitor=self.memory_monitor,
            show_progress=True
        )
        
        # Cache for schema inference results
        self._schema_cache = {}
        
    def analyze_files_for_unification(self, 
                                    file_paths: List[str],
                                    conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze multiple files to create a unified schema.
        
        Args:
            file_paths: List of file paths to analyze
            conversation_id: Optional conversation ID for LLM context
            
        Returns:
            Analysis results with unified schema proposal
        """
        logger.info(f"Starting schema unification analysis for {len(file_paths)} files")
        
        # Step 1: Extract schema and samples from each file
        file_schemas = {}
        file_samples = {}
        
        for file_path in tqdm(file_paths, desc="Analyzing file schemas"):
            try:
                schema_info = self._extract_file_schema(file_path)
                file_schemas[file_path] = schema_info
                
                # Get sample data
                sample_data = self._get_sample_data(file_path, self.max_sample_rows)
                file_samples[file_path] = sample_data
                
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {str(e)}")
                continue
        
        # Step 2: Use LLM to infer unified schema and relationships
        unified_schema = None
        if self.llm_system:
            unified_schema = self._llm_infer_unified_schema(
                file_schemas, file_samples, conversation_id
            )
        
        # Step 3: Create programmatic mappings
        column_mappings = self._create_column_mappings(file_schemas, unified_schema)
        
        # Step 4: Validate mappings and suggest improvements
        validation_results = self._validate_mappings(file_schemas, column_mappings, file_samples)
        
        return {
            "unified_schema": unified_schema,
            "file_schemas": file_schemas,
            "column_mappings": column_mappings,
            "validation_results": validation_results,
            "file_samples": file_samples,
            "recommendations": self._generate_recommendations(validation_results)
        }
    
    def create_supabase_migration(self, 
                                unified_schema: UnifiedSchema,
                                file_paths: List[str],
                                batch_size: int = 1000) -> Dict[str, Any]:
        """
        Create Supabase migration with actual data using programmatic processing.
        
        Args:
            unified_schema: The unified schema to use
            file_paths: Files to migrate
            batch_size: Batch size for data insertion
            
        Returns:
            Migration results
        """
        logger.info(f"Starting Supabase migration for {len(file_paths)} files")
        
        migration_results = {
            "table_created": False,
            "total_rows_migrated": 0,
            "files_processed": 0,
            "failed_files": [],
            "data_quality_issues": [],
            "processing_time": 0,
            "schema_sql": "",
            "migration_log": []
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Generate and execute table creation SQL
            create_table_sql = self._generate_create_table_sql(unified_schema)
            migration_results["schema_sql"] = create_table_sql
            
            # TODO: Execute SQL on Supabase
            # For now, we'll simulate table creation
            migration_results["table_created"] = True
            migration_results["migration_log"].append("Table schema created successfully")
            
            # Step 2: Process and migrate data from each file
            for file_path in file_paths:
                try:
                    file_result = self._migrate_file_data(
                        file_path, unified_schema, batch_size
                    )
                    
                    migration_results["total_rows_migrated"] += file_result["rows_migrated"]
                    migration_results["files_processed"] += 1
                    migration_results["data_quality_issues"].extend(file_result["quality_issues"])
                    migration_results["migration_log"].append(
                        f"File {os.path.basename(file_path)}: {file_result['rows_migrated']} rows migrated"
                    )
                    
                except Exception as e:
                    logger.error(f"Error migrating file {file_path}: {str(e)}")
                    migration_results["failed_files"].append({
                        "file_path": file_path,
                        "error": str(e)
                    })
            
            migration_results["processing_time"] = time.time() - start_time
            logger.info(f"Migration completed in {migration_results['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            migration_results["error"] = str(e)
        
        return migration_results
    
    def _extract_file_schema(self, file_path: str) -> Dict[str, Any]:
        """Extract schema information from a file."""
        file_info = self.file_processor.get_file_info(file_path)
        
        # Get sample data to infer data types
        sample_df = None
        try:
            for chunk in self.file_processor.read_file(file_path):
                sample_df = chunk.head(self.max_sample_rows)
                break
        except Exception as e:
            logger.warning(f"Could not read sample data from {file_path}: {str(e)}")
        
        schema_info = {
            "file_path": file_path,
            "file_type": file_info.get("file_type", "unknown"),
            "columns": {},
            "row_count": file_info.get("row_count"),
            "file_size_mb": file_info.get("file_size_mb", 0)
        }
        
        if sample_df is not None:
            for column in sample_df.columns:
                # Infer data type
                dtype = str(sample_df[column].dtype)
                python_type = self._pandas_to_sql_type(dtype)
                
                # Get sample values (non-null)
                sample_values = sample_df[column].dropna().head(5).tolist()
                
                schema_info["columns"][column] = {
                    "data_type": python_type,
                    "pandas_dtype": dtype,
                    "null_count": int(sample_df[column].isnull().sum()),
                    "sample_values": sample_values,
                    "unique_count": int(sample_df[column].nunique())
                }
        
        return schema_info
    
    def _get_sample_data(self, file_path: str, max_rows: int) -> pd.DataFrame:
        """Get sample data from a file."""
        try:
            for chunk in self.file_processor.read_file(file_path):
                return chunk.head(max_rows)
        except Exception as e:
            logger.error(f"Error reading sample data from {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def _llm_infer_unified_schema(self, 
                                file_schemas: Dict[str, Dict],
                                file_samples: Dict[str, pd.DataFrame],
                                conversation_id: Optional[str] = None) -> Optional[UnifiedSchema]:
        """Use LLM to infer unified schema across files."""
        if not self.llm_system:
            logger.warning("No LLM system available for schema inference")
            return None
        
        try:
            # Prepare data for LLM analysis
            schema_summary = self._prepare_schema_summary_for_llm(file_schemas, file_samples)
            
            # Create or use existing conversation
            if not conversation_id:
                conversation_id = self.llm_system.create_conversation(
                    title="Schema Unification Analysis",
                    data_files=list(file_schemas.keys())
                )
            
            # Ask LLM to analyze and unify schemas
            prompt = f"""
Analyze these file schemas and create a unified database schema. The files contain similar data but in different formats:

{schema_summary}

Please provide:
1. A unified table structure with appropriate column names and data types
2. Primary key recommendations
3. Column mappings from each file to the unified schema
4. Data relationships and foreign keys if applicable
5. Any data transformation rules needed

Focus on creating a clean, normalized schema that can accommodate all the data.
"""
            
            response = self.llm_system.add_message(prompt, conversation_id)
            
            # Parse LLM response (this would need more sophisticated parsing)
            # For now, return a basic unified schema
            return self._parse_llm_schema_response(response["response"], file_schemas)
            
        except Exception as e:
            logger.error(f"Error in LLM schema inference: {str(e)}")
            return None
    
    def _prepare_schema_summary_for_llm(self, 
                                      file_schemas: Dict[str, Dict],
                                      file_samples: Dict[str, pd.DataFrame]) -> str:
        """Prepare a concise schema summary for LLM analysis."""
        summary_parts = []
        
        for file_path, schema in file_schemas.items():
            file_name = os.path.basename(file_path)
            summary_parts.append(f"\n**File: {file_name}**")
            summary_parts.append(f"Type: {schema['file_type']}")
            summary_parts.append(f"Rows: {schema.get('row_count', 'unknown')}")
            summary_parts.append("Columns:")
            
            for col_name, col_info in schema["columns"].items():
                sample_vals = col_info.get("sample_values", [])[:3]
                summary_parts.append(
                    f"  - {col_name} ({col_info['data_type']}): {sample_vals}"
                )
        
        return "\n".join(summary_parts)
    
    def _parse_llm_schema_response(self, 
                                 llm_response: str,
                                 file_schemas: Dict[str, Dict]) -> UnifiedSchema:
        """Parse LLM response to extract unified schema (simplified version)."""
        # This is a simplified implementation
        # In a real system, you'd want more sophisticated parsing
        
        # Create a basic unified schema by merging all columns
        all_columns = {}
        file_mappings = {}
        
        for file_path, schema in file_schemas.items():
            file_mappings[file_path] = []
            for col_name, col_info in schema["columns"].items():
                # Normalize column name
                unified_col_name = col_name.lower().replace(" ", "_")
                all_columns[unified_col_name] = col_info["data_type"]
                
                # Create mapping
                mapping = ColumnMapping(
                    source_column=col_name,
                    target_column=unified_col_name,
                    source_file=file_path,
                    data_type=col_info["data_type"],
                    confidence_score=0.9
                )
                file_mappings[file_path].append(mapping)
        
        return UnifiedSchema(
            table_name="unified_data",
            columns=all_columns,
            primary_keys=["id"],  # Default primary key
            foreign_keys={},
            indexes=[],
            constraints={},
            file_mappings=file_mappings,
            relationships=[],
            metadata={
                "created_by": "smart_schema_unifier",
                "llm_response": llm_response,
                "confidence": "medium"
            }
        )
    
    def _create_column_mappings(self, 
                              file_schemas: Dict[str, Dict],
                              unified_schema: Optional[UnifiedSchema]) -> Dict[str, List[ColumnMapping]]:
        """Create programmatic column mappings."""
        if unified_schema and unified_schema.file_mappings:
            return unified_schema.file_mappings
        
        # Fallback: create basic mappings
        mappings = {}
        for file_path, schema in file_schemas.items():
            mappings[file_path] = []
            for col_name, col_info in schema["columns"].items():
                mapping = ColumnMapping(
                    source_column=col_name,
                    target_column=col_name.lower().replace(" ", "_"),
                    source_file=file_path,
                    data_type=col_info["data_type"],
                    confidence_score=0.7
                )
                mappings[file_path].append(mapping)
        
        return mappings
    
    def _validate_mappings(self, 
                         file_schemas: Dict[str, Dict],
                         column_mappings: Dict[str, List[ColumnMapping]],
                         file_samples: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate column mappings and identify potential issues."""
        validation_results = {
            "total_mappings": 0,
            "high_confidence_mappings": 0,
            "low_confidence_mappings": 0,
            "data_type_conflicts": [],
            "missing_mappings": [],
            "duplicate_targets": [],
            "recommendations": []
        }
        
        # Track target columns to detect duplicates
        target_column_sources = defaultdict(list)
        
        for file_path, mappings in column_mappings.items():
            validation_results["total_mappings"] += len(mappings)
            
            for mapping in mappings:
                # Check confidence
                if mapping.confidence_score >= self.confidence_threshold:
                    validation_results["high_confidence_mappings"] += 1
                else:
                    validation_results["low_confidence_mappings"] += 1
                
                # Track target columns
                target_column_sources[mapping.target_column].append(mapping)
        
        # Check for duplicate targets (multiple sources mapping to same target)
        for target_col, source_mappings in target_column_sources.items():
            if len(source_mappings) > 1:
                # Check if data types are compatible
                data_types = set(m.data_type for m in source_mappings)
                if len(data_types) > 1:
                    validation_results["data_type_conflicts"].append({
                        "target_column": target_col,
                        "source_mappings": [asdict(m) for m in source_mappings],
                        "conflicting_types": list(data_types)
                    })
        
        return validation_results
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if validation_results["low_confidence_mappings"] > 0:
            recommendations.append(
                f"Review {validation_results['low_confidence_mappings']} low-confidence column mappings"
            )
        
        if validation_results["data_type_conflicts"]:
            recommendations.append(
                f"Resolve {len(validation_results['data_type_conflicts'])} data type conflicts"
            )
        
        if validation_results["total_mappings"] == 0:
            recommendations.append("No column mappings found - files may have very different structures")
        
        return recommendations
    
    def _pandas_to_sql_type(self, pandas_dtype: str) -> str:
        """Convert pandas dtype to SQL data type."""
        type_mapping = {
            'int64': 'INTEGER',
            'float64': 'DECIMAL',
            'object': 'TEXT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'category': 'TEXT'
        }
        return type_mapping.get(pandas_dtype, 'TEXT')
    
    def _generate_create_table_sql(self, unified_schema: UnifiedSchema) -> str:
        """Generate CREATE TABLE SQL for unified schema."""
        sql_parts = [f"CREATE TABLE {unified_schema.table_name} ("]
        
        # Add columns
        column_definitions = []
        for col_name, data_type in unified_schema.columns.items():
            column_definitions.append(f"  {col_name} {data_type}")
        
        sql_parts.append(",\n".join(column_definitions))
        
        # Add primary key
        if unified_schema.primary_keys:
            pk_cols = ", ".join(unified_schema.primary_keys)
            sql_parts.append(f",\n  PRIMARY KEY ({pk_cols})")
        
        sql_parts.append("\n);")
        
        return "\n".join(sql_parts)
    
    def _migrate_file_data(self, 
                         file_path: str,
                         unified_schema: UnifiedSchema,
                         batch_size: int) -> Dict[str, Any]:
        """Migrate data from a single file to the unified schema."""
        logger.info(f"Migrating data from {os.path.basename(file_path)}")
        
        result = {
            "rows_migrated": 0,
            "batches_processed": 0,
            "quality_issues": [],
            "processing_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Get column mappings for this file
            file_mappings = unified_schema.file_mappings.get(file_path, [])
            if not file_mappings:
                raise ValueError(f"No column mappings found for file {file_path}")
            
            # Create mapping dictionary
            column_map = {m.source_column: m.target_column for m in file_mappings}
            
            # Process file in chunks
            for chunk_df in self.file_processor.read_file(file_path):
                # Transform chunk according to mappings
                transformed_chunk = self._transform_chunk(chunk_df, column_map, unified_schema)
                
                # Validate data quality
                quality_issues = self._validate_chunk_quality(transformed_chunk, unified_schema)
                result["quality_issues"].extend(quality_issues)
                
                # TODO: Insert into Supabase
                # For now, just simulate insertion
                result["rows_migrated"] += len(transformed_chunk)
                result["batches_processed"] += 1
                
                # Memory cleanup
                del transformed_chunk
                self.memory_monitor.cleanup_memory()
        
        except Exception as e:
            logger.error(f"Error migrating file {file_path}: {str(e)}")
            raise
        
        result["processing_time"] = time.time() - start_time
        return result
    
    def _transform_chunk(self, 
                       chunk_df: pd.DataFrame,
                       column_map: Dict[str, str],
                       unified_schema: UnifiedSchema) -> pd.DataFrame:
        """Transform a data chunk according to column mappings."""
        # Rename columns according to mapping
        mapped_df = chunk_df.rename(columns=column_map)
        
        # Ensure all target columns exist
        for target_col in unified_schema.columns.keys():
            if target_col not in mapped_df.columns:
                mapped_df[target_col] = None
        
        # Select only target columns
        target_columns = list(unified_schema.columns.keys())
        result_df = mapped_df[target_columns].copy()
        
        return result_df
    
    def _validate_chunk_quality(self, 
                              chunk_df: pd.DataFrame,
                              unified_schema: UnifiedSchema) -> List[Dict[str, Any]]:
        """Validate data quality of a transformed chunk."""
        issues = []
        
        for col_name, expected_type in unified_schema.columns.items():
            if col_name in chunk_df.columns:
                # Check for null values in non-nullable columns
                null_count = chunk_df[col_name].isnull().sum()
                if null_count > 0:
                    issues.append({
                        "type": "null_values",
                        "column": col_name,
                        "count": int(null_count),
                        "percentage": float(null_count / len(chunk_df) * 100)
                    })
        
        return issues 