"""
Data-specific visualization utilities.

This module provides specialized visualization functions for data exploration,
focused on data quality, patterns, and relationships.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Any, Union, Optional, Tuple
import io
import base64


def create_data_quality_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a data quality chart showing missing values and other quality metrics.
    
    Args:
        df: Pandas dataframe to analyze
        
    Returns:
        Plotly figure object
    """
    # Calculate missing values
    missing = df.isna().sum() / len(df) * 100
    
    # Create a horizontal bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=missing.index,
        x=missing.values,
        orientation='h',
        name='Missing Values',
        marker_color='crimson'
    ))
    
    # Add categoricals to represent data types
    dtypes = df.dtypes.astype(str)
    dtype_categories = {
        'float': 'numeric',
        'int': 'numeric',
        'object': 'categorical',
        'datetime': 'datetime',
        'bool': 'boolean'
    }
    
    # Map types to categories
    dtype_mapped = []
    for dtype in dtypes:
        for key, value in dtype_categories.items():
            if key in dtype:
                dtype_mapped.append(value)
                break
        else:
            dtype_mapped.append('other')
    
    # Add hover information
    hover_texts = [f"{col}<br>Type: {dtype_mapped[i]}<br>Missing: {missing[col]:.1f}%" 
                  for i, col in enumerate(missing.index)]
    
    # Update the figure
    fig.update_traces(hovertext=hover_texts)
    fig.update_layout(
        title='Data Quality Assessment',
        xaxis_title='Missing Values (%)',
        yaxis_title='Column',
        height=max(400, len(df.columns) * 25),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    return fig


def create_correlations_chart(df: pd.DataFrame, threshold: float = 0.0) -> go.Figure:
    """
    Create a correlation heatmap for numeric columns.
    
    Args:
        df: Pandas dataframe to analyze
        threshold: Minimum correlation strength to display (absolute value)
        
    Returns:
        Plotly figure object
    """
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=np.number)
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Apply threshold mask
    mask = np.abs(corr.values) < threshold
    corr_filtered = corr.copy()
    corr_filtered.values[mask] = np.nan
    
    # Create heatmap
    fig = px.imshow(
        corr_filtered,
        x=corr_filtered.columns,
        y=corr_filtered.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation")
    )
    
    # Update layout
    fig.update_layout(
        title='Feature Correlation Heatmap',
        height=max(500, len(numeric_df.columns) * 30),
        width=max(500, len(numeric_df.columns) * 30),
    )
    
    return fig


def create_distribution_plots(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, go.Figure]:
    """
    Create distribution plots for selected columns based on data types.
    
    Args:
        df: Pandas dataframe to analyze
        columns: List of columns to plot (if None, select 5 representative columns)
        
    Returns:
        Dictionary of column names to plotly figures
    """
    # If no columns specified, select a sample of representative columns
    if not columns:
        # Try to include different types
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=np.datetime64).columns.tolist()
        
        # Select a mix if available
        columns = []
        if numeric_cols:
            columns.extend(numeric_cols[:min(2, len(numeric_cols))])
        if cat_cols:
            columns.extend(cat_cols[:min(2, len(cat_cols))])
        if date_cols:
            columns.extend(date_cols[:min(1, len(date_cols))])
            
        # Ensure we have at least some columns
        if not columns and len(df.columns) > 0:
            columns = df.columns[:5].tolist()
    
    # Create figures
    figures = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        # Check data type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Create histogram for numeric columns
            fig = px.histogram(
                df, 
                x=col, 
                title=f"Distribution of {col}",
                marginal="box",  # Include box plot on the margin
                color_discrete_sequence=['#3366CC']
            )
            fig.update_layout(showlegend=False)
            
        elif pd.api.types.is_datetime64_dtype(df[col]):
            # Create timeline for datetime columns
            fig = px.histogram(
                df, 
                x=col, 
                title=f"Timeline of {col}",
                color_discrete_sequence=['#33CC99']
            )
            
        else:
            # Create bar chart for categorical columns
            value_counts = df[col].value_counts()
            if len(value_counts) > 15:
                # Take just top 15 categories if there are too many
                value_counts = value_counts[:15]
                
            fig = px.bar(
                value_counts,
                title=f"Distribution of {col}",
                color_discrete_sequence=['#9966FF'],
                labels={'value': 'Count', 'index': col}
            )
        
        figures[col] = fig
    
    return figures


def create_anomaly_detection_chart(df: pd.DataFrame, column: str) -> go.Figure:
    """
    Create a chart highlighting potential anomalies in a numeric column.
    
    Args:
        df: Pandas dataframe to analyze
        column: Column name to examine
        
    Returns:
        Plotly figure object
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="Column not found or not numeric",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Calculate outlier boundaries using IQR method
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Identify anomalies
    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    # Create the figure
    fig = go.Figure()
    
    # Add normal data points
    fig.add_trace(go.Box(
        x=df[column],
        name='Distribution',
        boxpoints='outliers',
        jitter=0.3,
        pointpos=-1.8
    ))
    
    # Add vertical lines for bounds
    fig.add_shape(
        type='line',
        x0=lower_bound, y0=0, x1=lower_bound, y1=1,
        yref='paper',
        line=dict(color='red', width=2, dash='dash')
    )
    
    fig.add_shape(
        type='line',
        x0=upper_bound, y0=0, x1=upper_bound, y1=1,
        yref='paper',
        line=dict(color='red', width=2, dash='dash')
    )
    
    # Add annotations for bounds
    fig.add_annotation(
        x=lower_bound, y=1,
        yref='paper',
        text=f"Lower bound: {lower_bound:.2f}",
        showarrow=True,
        arrowhead=1
    )
    
    fig.add_annotation(
        x=upper_bound, y=1,
        yref='paper',
        text=f"Upper bound: {upper_bound:.2f}",
        showarrow=True,
        arrowhead=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"Anomaly Detection for {column}",
        height=400,
        showlegend=False,
        annotations=[
            dict(
                x=0.5, y=-0.15,
                xref='paper', yref='paper',
                text=f'Anomaly count: {len(anomalies)} ({len(anomalies)/len(df)*100:.1f}%)',
                showarrow=False
            )
        ]
    )
    
    return fig


def create_data_comparison_chart(df1: pd.DataFrame, df2: pd.DataFrame, column: str) -> go.Figure:
    """
    Create a chart comparing distributions of the same column in two dataframes.
    
    Args:
        df1: First dataframe
        df2: Second dataframe
        column: Column to compare
        
    Returns:
        Plotly figure object
    """
    if column not in df1.columns or column not in df2.columns:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="Column not found in one or both datasets",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Check data type
    if pd.api.types.is_numeric_dtype(df1[column]) and pd.api.types.is_numeric_dtype(df2[column]):
        # Create histograms for numeric columns
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df1[column],
            name='Dataset 1',
            opacity=0.7,
            marker_color='blue'
        ))
        fig.add_trace(go.Histogram(
            x=df2[column],
            name='Dataset 2',
            opacity=0.7,
            marker_color='red'
        ))
        
        fig.update_layout(
            title=f"Comparison of {column} Distribution",
            xaxis_title=column,
            yaxis_title='Count',
            barmode='overlay'
        )
    else:
        # Create bar charts for categorical columns
        counts1 = df1[column].value_counts()
        counts2 = df2[column].value_counts()
        
        # Get all unique categories
        all_categories = sorted(set(counts1.index) | set(counts2.index))
        
        # Prepare data for plotting
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=all_categories,
            y=[counts1.get(cat, 0) for cat in all_categories],
            name='Dataset 1',
            marker_color='blue'
        ))
        fig.add_trace(go.Bar(
            x=all_categories,
            y=[counts2.get(cat, 0) for cat in all_categories],
            name='Dataset 2',
            marker_color='red'
        ))
        
        fig.update_layout(
            title=f"Comparison of {column} Distribution",
            xaxis_title=column,
            yaxis_title='Count',
            barmode='group',
            xaxis={'categoryorder':'total descending'}
        )
    
    return fig


def create_data_profile_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a summary data profile with key statistics.
    
    Args:
        df: Pandas dataframe to analyze
        
    Returns:
        Dictionary of profile metrics
    """
    # Basic metrics
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # in MB
        "missing_values_pct": (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
        "duplicate_rows_pct": (df.duplicated().sum() / len(df)) * 100,
        "column_types": {},
    }
    
    # Column type counts
    type_counts = {}
    for dtype in df.dtypes:
        dtype_str = str(dtype)
        if 'float' in dtype_str:
            key = 'numeric'
        elif 'int' in dtype_str:
            key = 'numeric'
        elif 'object' in dtype_str:
            key = 'categorical'
        elif 'datetime' in dtype_str:
            key = 'datetime'
        elif 'bool' in dtype_str:
            key = 'boolean'
        else:
            key = 'other'
            
        type_counts[key] = type_counts.get(key, 0) + 1
    
    profile["column_types"] = type_counts
    
    # Column-specific statistics for numeric fields
    numeric_stats = {}
    for col in df.select_dtypes(include=np.number).columns:
        numeric_stats[col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "missing_pct": (df[col].isna().sum() / len(df)) * 100
        }
    
    profile["numeric_stats"] = numeric_stats
    
    # Column-specific statistics for categorical fields
    cat_stats = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        value_counts = df[col].value_counts()
        cat_stats[col] = {
            "unique_values": df[col].nunique(),
            "top_value": value_counts.index[0] if len(value_counts) > 0 else None,
            "top_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
            "missing_pct": (df[col].isna().sum() / len(df)) * 100
        }
    
    profile["categorical_stats"] = cat_stats
    
    return profile
