"""
Visualization utilities for data chat interface.

This module provides functions to create visualizations for the data chat interface,
including schema diagrams, relationship maps, and other visual elements.
"""
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Union
import base64
import streamlit as st


def create_schema_diagram(data):
    """Create a simple schema diagram."""
    return "Schema diagram placeholder"


def create_relationship_diagram(data):
    """Create a simple relationship diagram."""
    return "Relationship diagram placeholder"


def create_data_quality_chart(quality_metrics: Dict[str, Dict[str, float]]) -> bytes:
    """
    Create a data quality visualization chart.
    
    Args:
        quality_metrics: Dictionary of quality metrics by file
        
    Returns:
        Image bytes
    """
    # Check if metrics exist
    if not quality_metrics:
        return None
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    files = list(quality_metrics.keys())
    metrics = ['completeness', 'consistency', 'accuracy', 'uniqueness']
    
    # Filter metrics to only those present
    available_metrics = set()
    for file_metrics in quality_metrics.values():
        available_metrics.update(file_metrics.keys())
    metrics = [m for m in metrics if m in available_metrics]
    
    x = np.arange(len(files))
    width = 0.8 / len(metrics)
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        values = [quality_metrics[f].get(metric, 0) for f in files]
        ax.bar(x + i * width - (len(metrics) - 1) * width / 2, values, width, label=metric.capitalize())
    
    # Add labels and legend
    ax.set_ylabel('Score')
    ax.set_title('Data Quality Metrics by File')
    ax.set_xticks(x)
    ax.set_xticklabels(files, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    
    # Close figure to prevent display
    plt.close()
    
    return buf.getvalue()


def create_distribution_plot(df: pd.DataFrame, column: str) -> bytes:
    """
    Create a distribution plot for a column.
    
    Args:
        df: DataFrame containing the column
        column: Column name to plot
        
    Returns:
        Image bytes
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check data type
    if pd.api.types.is_numeric_dtype(df[column]):
        # Numeric column - plot histogram
        ax.hist(df[column].dropna(), bins=30, alpha=0.7, color='steelblue')
        
        # Add mean and median lines
        mean_val = df[column].mean()
        median_val = df[column].median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
        
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
    else:
        # Categorical column - plot bar chart
        value_counts = df[column].value_counts().sort_values(ascending=False).head(15)
        value_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_xlabel('')
        ax.set_ylabel('Count')
        
    ax.set_title(f'Distribution of {column}')
    ax.legend()
    plt.tight_layout()
    
    # Save figure to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    
    # Close figure
    plt.close()
    
    return buf.getvalue()


def create_correlation_heatmap(df: pd.DataFrame, columns: List[str] = None) -> bytes:
    """
    Create a correlation heatmap for numeric columns.
    
    Args:
        df: DataFrame to analyze
        columns: Optional list of columns to include
        
    Returns:
        Image bytes
    """
    # Select numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    
    # Filter columns if specified
    if columns:
        numeric_df = numeric_df[columns]
    
    # Need at least 2 columns
    if numeric_df.shape[1] < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='coolwarm')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Add ticks and labels
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                    ha="center", va="center", 
                    color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
    
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    
    # Save figure to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    
    # Close figure
    plt.close()
    
    return buf.getvalue()


def create_side_by_side_comparison(df1: pd.DataFrame, df2: pd.DataFrame, 
                                  label1: str, label2: str) -> bytes:
    """
    Create a side-by-side comparison of two dataframes.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        label1: Label for first DataFrame
        label2: Label for second DataFrame
        
    Returns:
        Image bytes
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create summary statistics
    stats1 = pd.DataFrame({
        'Column': df1.columns,
        'Type': df1.dtypes.astype(str),
        'Non-Null': df1.count(),
        'Null': df1.isna().sum(),
        'Unique': [df1[col].nunique() for col in df1.columns]
    })
    
    stats2 = pd.DataFrame({
        'Column': df2.columns,
        'Type': df2.dtypes.astype(str),
        'Non-Null': df2.count(),
        'Null': df2.isna().sum(),
        'Unique': [df2[col].nunique() for col in df2.columns]
    })
    
    # Create tables
    ax1.axis('tight')
    ax1.axis('off')
    ax1_table = ax1.table(
        cellText=stats1.values,
        colLabels=stats1.columns,
        loc='center',
        cellLoc='center'
    )
    ax1_table.auto_set_font_size(False)
    ax1_table.set_fontsize(9)
    ax1_table.scale(1, 1.5)
    ax1.set_title(f"{label1} ({len(df1)} rows)")
    
    ax2.axis('tight')
    ax2.axis('off')
    ax2_table = ax2.table(
        cellText=stats2.values,
        colLabels=stats2.columns,
        loc='center',
        cellLoc='center'
    )
    ax2_table.auto_set_font_size(False)
    ax2_table.set_fontsize(9)
    ax2_table.scale(1, 1.5)
    ax2.set_title(f"{label2} ({len(df2)} rows)")
    
    plt.tight_layout()
    
    # Save figure to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    
    # Close figure
    plt.close()
    
    return buf.getvalue()
