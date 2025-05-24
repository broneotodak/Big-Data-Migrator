"""
Smart Query Processor for Big Data Migrator

This module detects user query intent and processes data directly using pandas/SQL
instead of overwhelming LLMs with raw data context.
"""
import re
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class QueryIntent:
    """Represents detected user query intent"""
    def __init__(self, intent_type: str, confidence: float, parameters: Dict[str, Any]):
        self.intent_type = intent_type
        self.confidence = confidence
        self.parameters = parameters

class ProcessedResults:
    """Results from direct data processing"""
    def __init__(self, primary_answer: str, detailed_results: Dict[str, Any], 
                 formatted_summary: str, calculation_method: str):
        self.primary_answer = primary_answer
        self.detailed_results = detailed_results
        self.formatted_summary = formatted_summary
        self.calculation_method = calculation_method

class SmartQueryProcessor:
    """
    Intelligent query processor that handles data calculations directly
    instead of relying on LLM context processing.
    """
    
    def __init__(self):
        self.intent_patterns = {
            'comparison': [
                r'\b(compare|comparison|difference|diff|vs|versus)\b',
                r'\bmissing\b.*\b(between|from)\b',
                r'\bhow\s+much\s+is\s+missing\b',
                r'\bdifference\s+between\b',
                r'\bmissing.*\b(transactions?|records?|entries?)\b',
                r'\bhow\s+many.*\bmissing\b',
                r'\bfind.*\bmissing\b',
                r'\bcheck.*\bmissing\b',
                r'\bmissing.*\bcount\b',
                r'\bmissing.*\bRM\b'
            ],
            'aggregation': [
                r'\b(total|sum|count|average|mean)\b(?!.*\bmissing\b)',
                r'\bhow\s+much\b.*\b(total|altogether)\b(?!.*\bmissing\b)',
                r'\bgrand\s+total\b(?!.*\bmissing\b)'
            ],
            'analysis': [
                r'\banalyze|analysis\b',
                r'\bbreakdown\b',
                r'\binsights?\b',
                r'\bpatterns?\b',
                r'\brelation(s|ship)?\b',
                r'\blogical\b.*\brelation\b'
            ],
            'transaction_matching': [
                r'\bmatch\b.*\btransactions?\b',
                r'\brelated\b.*\btransactions?\b',
                r'\bcommon\b.*\btransactions?\b'
            ]
        }
        
        # Common amount column names to look for
        self.amount_columns = [
            'Transaction Amount (RM)', 'transaction_amount', 'amount', 'Amount',
            'Gross payments', 'Net payments', 'Settlement Amount (RM)',
            'value', 'Value', 'total', 'Total'
        ]
        
        # Common ID column names for matching
        self.id_columns = [
            'Transaction ID', 'transaction_id', 'id', 'ID', 'reference',
            'Reference', 'Transaction Reference', 'Merchant Reference'
        ]
    
    def detect_intent(self, query: str, available_files: List[str]) -> QueryIntent:
        """
        Detect user query intent using pattern matching
        
        Args:
            query: User's query text
            available_files: List of available data files
            
        Returns:
            QueryIntent object with detected intent and confidence
        """
        query_lower = query.lower()
        best_intent = None
        best_confidence = 0.0
        best_parameters = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            confidence = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    confidence += 1.0
            
            # Normalize confidence by number of patterns
            confidence = confidence / len(patterns)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent_type
                
                # Extract parameters based on intent
                if intent_type == 'comparison' and len(available_files) >= 2:
                    best_parameters = {
                        'files': available_files[:2],  # Take first two files
                        'comparison_type': 'missing_analysis' if 'missing' in query_lower else 'general'
                    }
                elif intent_type == 'aggregation':
                    best_parameters = {
                        'files': available_files,
                        'operation': self._extract_aggregation_type(query_lower)
                    }
        
        # If no strong pattern match, but multiple files available, assume comparison
        if best_confidence < 0.3 and len(available_files) >= 2:
            best_intent = 'comparison'
            best_confidence = 0.5
            best_parameters = {
                'files': available_files,
                'comparison_type': 'general'
            }
        
        return QueryIntent(
            intent_type=best_intent or 'analysis',
            confidence=best_confidence,
            parameters=best_parameters
        )
    
    def _extract_aggregation_type(self, query: str) -> str:
        """Extract what type of aggregation user wants"""
        if any(word in query for word in ['total', 'sum']):
            return 'sum'
        elif any(word in query for word in ['count', 'number']):
            return 'count'
        elif any(word in query for word in ['average', 'mean']):
            return 'mean'
        else:
            return 'sum'  # Default
    
    def process_query(self, query: str, data_files: Dict[str, pd.DataFrame]) -> ProcessedResults:
        """
        Process user query by doing actual data calculations
        
        Args:
            query: User's query
            data_files: Dictionary mapping file paths to DataFrames
            
        Returns:
            ProcessedResults with calculated answers
        """
        try:
            # Validate input query
            if not isinstance(query, str):
                logger.error(f"Invalid query type: {type(query)}, value: {query}")
                return ProcessedResults(
                    primary_answer=f"Invalid query format: expected string, got {type(query)}",
                    detailed_results={},
                    formatted_summary="Query validation failed.",
                    calculation_method="validation_error"
                )
            
            if len(query.strip()) == 0:
                logger.error("Empty query provided")
                return ProcessedResults(
                    primary_answer="Empty query provided",
                    detailed_results={},
                    formatted_summary="No query to process.",
                    calculation_method="validation_error"
                )
                
            # Log the actual query for debugging
            logger.info(f"Processing query: '{query}'")
            logger.info(f"Available data files: {len(data_files)}")
            
            file_list = list(data_files.keys())
            intent = self.detect_intent(query, file_list)
            
            logger.info(f"Detected intent: {intent.intent_type} (confidence: {intent.confidence:.2f})")
            
            if intent.intent_type == 'comparison':
                return self._process_comparison(query, data_files, intent.parameters)
            elif intent.intent_type == 'aggregation':
                return self._process_aggregation(query, data_files, intent.parameters)
            elif intent.intent_type == 'transaction_matching':
                return self._process_transaction_matching(query, data_files, intent.parameters)
            else:
                return self._process_general_analysis(query, data_files)
                
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ProcessedResults(
                primary_answer=f"Error processing query: {str(e)}",
                detailed_results={"error": str(e), "query": query},
                formatted_summary="Unable to process query due to error.",
                calculation_method="error_handling"
            )
    
    def _process_comparison(self, query: str, data_files: Dict[str, pd.DataFrame], 
                          parameters: Dict[str, Any]) -> ProcessedResults:
        """Process file comparison queries"""
        files = parameters.get('files', list(data_files.keys())[:2])
        
        if len(files) < 2:
            return ProcessedResults(
                primary_answer="Need at least 2 files for comparison",
                detailed_results={},
                formatted_summary="Insufficient files for comparison.",
                calculation_method="validation_error"
            )
        
        file1_path, file2_path = files[0], files[1]
        raw_df1, raw_df2 = data_files[file1_path], data_files[file2_path]
        
        # Fix CSV headers for both files
        df1 = self._fix_csv_headers(raw_df1)
        df2 = self._fix_csv_headers(raw_df2)
        
        logger.info(f"File 1 columns after header fix: {list(df1.columns)}")
        logger.info(f"File 2 columns after header fix: {list(df2.columns)}")
        
        # Find amount columns
        amount_col1 = self._find_amount_column(df1)
        amount_col2 = self._find_amount_column(df2)
        
        logger.info(f"Detected amount columns: File1='{amount_col1}', File2='{amount_col2}'")
        
        if not amount_col1 or not amount_col2:
            return ProcessedResults(
                primary_answer=f"Could not find amount columns. File1: {amount_col1}, File2: {amount_col2}. Available columns: File1={list(df1.columns)}, File2={list(df2.columns)}",
                detailed_results={},
                formatted_summary="Unable to locate monetary amount columns for comparison.",
                calculation_method="column_detection_error"
            )
        
        # Calculate basic statistics
        file1_total = df1[amount_col1].sum()
        file2_total = df2[amount_col2].sum()
        file1_count = len(df1)
        file2_count = len(df2)
        
        # Calculate difference
        amount_difference = abs(file1_total - file2_total)
        count_difference = abs(file1_count - file2_count)
        
        # Try to find matching transactions if possible
        matching_info = self._find_matching_transactions(df1, df2, amount_col1, amount_col2)
        
        # Format results
        file1_name = Path(file1_path).name
        file2_name = Path(file2_path).name
        
        primary_answer = f"""Direct comparison results:
• {file1_name}: {file1_count:,} transactions, RM {file1_total:,.2f}
• {file2_name}: {file2_count:,} transactions, RM {file2_total:,.2f}
• Difference: {count_difference:,} transactions, RM {amount_difference:,.2f}"""

        if matching_info:
            primary_answer += f"\n• Common transactions: {matching_info['common_count']} found"
            if matching_info['missing_count'] > 0:
                primary_answer += f"\n• Missing from {file2_name}: {matching_info['missing_count']} transactions, RM {matching_info['missing_amount']:,.2f}"
        
        detailed_results = {
            'file1': {
                'name': file1_name,
                'count': file1_count,
                'total': file1_total,
                'amount_column': amount_col1
            },
            'file2': {
                'name': file2_name,
                'count': file2_count,
                'total': file2_total,
                'amount_column': amount_col2
            },
            'differences': {
                'count_difference': count_difference,
                'amount_difference': amount_difference
            },
            'matching': matching_info
        }
        
        formatted_summary = f"""Compared {file1_name} vs {file2_name}:
File 1: {file1_count:,} transactions totaling RM {file1_total:,.2f}
File 2: {file2_count:,} transactions totaling RM {file2_total:,.2f}
Numerical difference: {count_difference:,} transactions, RM {amount_difference:,.2f}"""
        
        return ProcessedResults(
            primary_answer=primary_answer,
            detailed_results=detailed_results,
            formatted_summary=formatted_summary,
            calculation_method="pandas_comparison"
        )
    
    def _process_aggregation(self, query: str, data_files: Dict[str, pd.DataFrame], 
                           parameters: Dict[str, Any]) -> ProcessedResults:
        """Process aggregation queries (sum, count, average)"""
        operation = parameters.get('operation', 'sum')
        
        results = {}
        grand_total = 0
        grand_count = 0
        
        for file_path, df in data_files.items():
            file_name = Path(file_path).name
            amount_col = self._find_amount_column(df)
            
            if amount_col:
                if operation == 'sum':
                    value = df[amount_col].sum()
                    grand_total += value
                elif operation == 'count':
                    value = len(df)
                    grand_count += value
                elif operation == 'mean':
                    value = df[amount_col].mean()
                
                results[file_name] = {
                    'value': value,
                    'count': len(df),
                    'amount_column': amount_col
                }
            else:
                results[file_name] = {
                    'value': 0,
                    'count': len(df),
                    'amount_column': 'Not found'
                }
        
        if operation == 'sum':
            primary_answer = f"Total across all files: RM {grand_total:,.2f}"
        elif operation == 'count':
            primary_answer = f"Total transactions across all files: {sum(r['count'] for r in results.values()):,}"
        else:
            avg_total = sum(r['value'] for r in results.values() if r['value'] > 0) / len([r for r in results.values() if r['value'] > 0])
            primary_answer = f"Average across all files: RM {avg_total:,.2f}"
        
        # Create detailed breakdown
        breakdown = []
        for file_name, data in results.items():
            if operation == 'sum':
                breakdown.append(f"• {file_name}: RM {data['value']:,.2f} ({data['count']:,} transactions)")
            elif operation == 'count':
                breakdown.append(f"• {file_name}: {data['count']:,} transactions")
            else:
                breakdown.append(f"• {file_name}: RM {data['value']:,.2f} average")
        
        formatted_summary = f"{operation.title()} calculation results:\n" + "\n".join(breakdown)
        
        return ProcessedResults(
            primary_answer=primary_answer,
            detailed_results=results,
            formatted_summary=formatted_summary,
            calculation_method=f"pandas_{operation}"
        )
    
    def _find_amount_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the most likely amount/monetary column in DataFrame"""
        
        # First, check if the first row contains header information
        df_to_search = self._fix_csv_headers(df)
        
        # Look for predefined amount column names
        for col_name in self.amount_columns:
            if col_name in df_to_search.columns:
                # Verify it's numeric (skip first row if it contains headers)
                numeric_data = pd.to_numeric(df_to_search[col_name], errors='coerce')
                if not numeric_data.dropna().empty:
                    return col_name
        
        # Fallback: look for any numeric column with amount keywords
        for col in df_to_search.columns:
            # Check if column contains amount-related keywords
            if any(keyword in str(col).lower() for keyword in ['amount', 'rm', 'total', 'sum', 'payment', 'gross', 'net']):
                # Try to convert to numeric
                numeric_data = pd.to_numeric(df_to_search[col], errors='coerce')
                if not numeric_data.dropna().empty:
                    return col
        
        # Last resort: look for any numeric column that might be amounts
        for col in df_to_search.columns:
            try:
                numeric_data = pd.to_numeric(df_to_search[col], errors='coerce')
                # Check if it looks like monetary amounts (has decimals, reasonable range)
                if not numeric_data.dropna().empty:
                    non_zero_values = numeric_data.dropna()
                    if len(non_zero_values) > 0:
                        # Check if values look like currency amounts
                        mean_val = non_zero_values.mean()
                        if 0.01 <= mean_val <= 100000:  # Reasonable currency range
                            return col
            except:
                continue
        
        return None
    
    def _fix_csv_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix CSV files where headers are in the first data row"""
        try:
            # Check if first row contains likely header information
            first_row = df.iloc[0] if len(df) > 0 else None
            
            if first_row is not None:
                # Check if first row contains header-like strings
                first_row_strings = [str(val) for val in first_row if pd.notna(val)]
                
                # Look for header indicators
                header_indicators = ['no', 'date', 'transaction', 'amount', 'reference', 'type', 'payment']
                has_header_words = any(
                    any(indicator in str(val).lower() for indicator in header_indicators)
                    for val in first_row_strings
                )
                
                if has_header_words and len(first_row_strings) > 3:
                    # First row likely contains headers, use it
                    new_df = df.copy()
                    new_df.columns = [str(val) if pd.notna(val) else f"Column_{i}" for i, val in enumerate(first_row)]
                    # Remove the header row from data
                    new_df = new_df.iloc[1:].reset_index(drop=True)
                    
                    # Try to convert numeric columns
                    for col in new_df.columns:
                        try:
                            new_df[col] = pd.to_numeric(new_df[col])
                        except (ValueError, TypeError):
                            # Keep as string if conversion fails
                            pass
                    
                    return new_df
            
            return df
            
        except Exception as e:
            logger.warning(f"Error fixing CSV headers: {str(e)}")
            return df
    
    def _find_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the most likely ID column in DataFrame"""
        for col_name in self.id_columns:
            if col_name in df.columns:
                return col_name
        
        # Fallback: look for columns with 'id' in name
        for col in df.columns:
            if 'id' in col.lower():
                return col
        
        return None
    
    def _find_matching_transactions(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                  amount_col1: str, amount_col2: str) -> Optional[Dict[str, Any]]:
        """Find matching transactions between two DataFrames"""
        try:
            # Try to match by amount (with small tolerance for floating point)
            tolerance = 0.01
            
            # Get unique amounts from each file
            amounts1 = set(df1[amount_col1].round(2))
            amounts2 = set(df2[amount_col2].round(2))
            
            # Find common amounts
            common_amounts = amounts1.intersection(amounts2)
            
            # Count transactions with common amounts
            common_transactions1 = df1[df1[amount_col1].round(2).isin(common_amounts)]
            common_transactions2 = df2[df2[amount_col2].round(2).isin(common_amounts)]
            
            # Find missing from each file
            missing_amounts = amounts1 - amounts2
            missing_transactions = df1[df1[amount_col1].round(2).isin(missing_amounts)]
            
            return {
                'common_count': len(common_transactions1),
                'common_amount': common_transactions1[amount_col1].sum(),
                'missing_count': len(missing_transactions),
                'missing_amount': missing_transactions[amount_col1].sum() if len(missing_transactions) > 0 else 0,
                'match_method': 'amount_based'
            }
            
        except Exception as e:
            logger.warning(f"Could not perform transaction matching: {str(e)}")
            return None
    
    def _process_transaction_matching(self, query: str, data_files: Dict[str, pd.DataFrame], 
                                    parameters: Dict[str, Any]) -> ProcessedResults:
        """Process transaction matching queries"""
        # Similar to comparison but focused on finding exact matches
        return self._process_comparison(query, data_files, parameters)
    
    def _process_general_analysis(self, query: str, data_files: Dict[str, pd.DataFrame]) -> ProcessedResults:
        """Process general analysis queries"""
        summary_stats = {}
        
        for file_path, df in data_files.items():
            file_name = Path(file_path).name
            amount_col = self._find_amount_column(df)
            
            stats = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns)
            }
            
            if amount_col:
                stats.update({
                    'amount_column': amount_col,
                    'total_amount': df[amount_col].sum(),
                    'average_amount': df[amount_col].mean(),
                    'min_amount': df[amount_col].min(),
                    'max_amount': df[amount_col].max()
                })
            
            summary_stats[file_name] = stats
        
        # Create summary
        total_rows = sum(stats['row_count'] for stats in summary_stats.values())
        total_amount = sum(stats.get('total_amount', 0) for stats in summary_stats.values())
        
        primary_answer = f"""Dataset Overview:
• Total files: {len(data_files)}
• Total transactions: {total_rows:,}
• Total amount: RM {total_amount:,.2f}"""
        
        formatted_summary = "File-by-file breakdown:\n"
        for file_name, stats in summary_stats.items():
            formatted_summary += f"• {file_name}: {stats['row_count']:,} rows"
            if 'total_amount' in stats:
                formatted_summary += f", RM {stats['total_amount']:,.2f} total"
            formatted_summary += "\n"
        
        return ProcessedResults(
            primary_answer=primary_answer,
            detailed_results=summary_stats,
            formatted_summary=formatted_summary,
            calculation_method="pandas_analysis"
        )
    
    def create_llm_context(self, query: str, processed_results: ProcessedResults) -> str:
        """
        Create optimized context for LLM that contains only calculated results,
        not raw data.
        """
        context = f"""The user asked: "{query}"

I have processed the data directly and calculated the following results:

PRIMARY ANSWER: {processed_results.primary_answer}

DETAILED BREAKDOWN:
{processed_results.formatted_summary}

CALCULATION METHOD: {processed_results.calculation_method}

Your task: Explain these calculated results to the user in a clear, conversational way. 
The calculations have already been done correctly - just interpret and explain the numbers.
Do NOT suggest Excel or external tools. The analysis is complete."""

        return context 