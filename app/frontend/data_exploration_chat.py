"""
Data exploration chat for advanced data discussions.

This module extends the base chat interface with specialized data exploration
capabilities, allowing users to interactively explore their datasets through
natural language conversation.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import plotly.express as px
import plotly.graph_objects as go

# Simplified imports to avoid circular dependencies
try:
    from chat_interface import ChatInterface
except ImportError:
    # Create a fallback if needed
    class ChatInterface:
        def __init__(self, api_url=None):
            self.api_url = api_url

# Fallback visualization functions
def create_data_quality_chart(df):
    """Create a simple data quality chart."""
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        fig = px.bar(x=missing_data.index, y=missing_data.values, 
                     title="Missing Values by Column")
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(text="No missing values found - 100% complete!", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_correlations_chart(df, threshold=0.3):
    """Create a simple correlation chart."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix")
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric columns for correlation", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_distribution_plots(df, columns):
    """Create simple distribution plots."""
    plots = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        else:
            value_counts = df[col].value_counts().head(10)
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                        title=f"Top 10 Values in {col}")
        plots[col] = fig
    return plots

def create_anomaly_detection_chart(df, column):
    """Create a simple anomaly detection chart."""
    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.box(df, y=column, title=f"Outlier Detection for {column}")
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(text="Anomaly detection only available for numeric columns", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_data_comparison_chart(df1, df2, column):
    """Create a simple data comparison chart."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df1[column], name="Dataset 1", opacity=0.7))
    fig.add_trace(go.Histogram(x=df2[column], name="Dataset 2", opacity=0.7))
    fig.update_layout(title=f"Comparison of {column}", barmode='overlay')
    return fig

def create_data_profile_summary(df):
    """Create a simple data profile summary."""
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        "missing_values_pct": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        "duplicate_rows_pct": (df.duplicated().sum() / len(df)) * 100,
        "column_types": df.dtypes.value_counts().to_dict()
    }
    
    # Convert numpy dtypes to strings for JSON serialization
    profile["column_types"] = {str(k): v for k, v in profile["column_types"].items()}
    
    return profile

class DataExplorationChat(ChatInterface):
    """
    Specialized chat interface for data exploration.
    
    This class:
    - Allows users to ask questions about their data structure
    - Provides interactive data visualization within chat
    - Offers schema suggestions with explanations
    - Handles complex data relationship discussions
    - Generates actionable migration recommendations
    """
    
    def __init__(self, api_url: str = None):
        """
        Initialize the data exploration chat.
        
        Args:
            api_url: URL for the API server
        """
        super().__init__(api_url)
        
        # Initialize additional components for data exploration
        if "visualization_history" not in st.session_state:
            st.session_state.visualization_history = []
        if "schema_recommendations" not in st.session_state:
            st.session_state.schema_recommendations = []
        if "data_relationships" not in st.session_state:
            st.session_state.data_relationships = []
        if "active_exploration" not in st.session_state:
            st.session_state.active_exploration = None
        
    def render(self):
        """Render the data exploration chat interface."""
        # Main layout - Split into 2/3 for chat and 1/3 for exploration
        chat_col, exploration_col = st.columns([2, 1])
        
        with chat_col:
            # Render base chat interface
            st.header("Data Exploration Chat")
            
            # Render conversation history
            self._render_chat_history()
            
            # Message input area
            self._render_message_input()
        
        with exploration_col:
            # Data exploration sidebar
            st.header("Data Explorer")
            
            # Data file selection
            self._render_file_selector()
            
        # Data visualization tools
            self._render_visualization_tools()
            
            # Schema recommendations
            with st.expander("Schema Recommendations", expanded=False):
                self._render_schema_recommendations()
            
            # Relationship mapping
            with st.expander("Data Relationships", expanded=False):
                self._render_relationship_mapping()
            with st.expander("Data Quality", expanded=False):
                self._render_data_quality()
    
    def _render_file_selector(self):
        """Render the data file selector."""
        if not st.session_state.data_preview:
            self._render_file_upload()
            return
        
        # File selector
        file_names = list(st.session_state.data_preview.keys())
        
        if file_names:
            selected_file = st.selectbox(
                "Select a data file to explore:",
                file_names,
                key="selected_data_file"
            )
            
            if selected_file:
                df = st.session_state.data_preview[selected_file]
                st.session_state.active_exploration = selected_file
                
                # Display data preview
                with st.expander("Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
                    
                    # Column info
                    col_info = pd.DataFrame({
                        "Type": df.dtypes,
                        "Non-Null": df.count(),
                        "Null": df.isna().sum()
                    })
                    st.dataframe(col_info)
    
    def _render_file_upload(self):
        """Render simple file upload interface."""
        st.info("Upload data files to start exploring")
        uploaded_files = st.file_uploader(
            "Upload data files",
            accept_multiple_files=True,
            type=["csv", "xlsx", "xls", "json", "pdf", "docx", "txt"]
        )
        
        if uploaded_files:
            # Simple file processing
            for uploaded_file in uploaded_files:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file, nrows=1000)
                    elif uploaded_file.name.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(uploaded_file, nrows=1000)
                    elif uploaded_file.name.endswith(".json"):
                        df = pd.read_json(uploaded_file, nrows=1000)
                    else:
                        df = pd.DataFrame({"Info": [f"File type: {uploaded_file.type}"]})
                    
                    st.session_state.data_preview[uploaded_file.name] = df
                    st.success(f"✅ Loaded {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")
    
    def _render_visualization_tools(self):
        """Render data visualization tools."""
        if not st.session_state.active_exploration:
            return
        
        # Get the active dataframe
        df = st.session_state.data_preview[st.session_state.active_exploration]
        
        st.write("### Data Visualization")
        
        # Select visualization type
        viz_type = st.selectbox(
            "Visualization type:",
            ["Select...", "Data Profile", "Data Quality", "Distributions", "Correlations", 
             "Anomaly Detection", "Data Comparison", "Scatter Plot", "Time Series"],
            key="viz_type"
        )
        
        if viz_type == "Data Profile":
            # Render data profile summary
            self._render_data_profile(df)
        
        elif viz_type == "Data Quality":
            # Render data quality visualization
            self._render_data_quality_visualization(df)
            
        elif viz_type == "Distributions":
            # Render distribution visualizations
            self._render_distribution_visualization(df)
            
        elif viz_type == "Correlations":
            # Render correlation visualization
            self._render_correlation_visualization(df)
            
        elif viz_type == "Anomaly Detection":
            # Render anomaly detection
            self._render_anomaly_detection(df)
            
        elif viz_type == "Data Comparison":
            # Render data comparison
            self._render_data_comparison(df)
            
        elif viz_type == "Column Distribution":
            # Column selection
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column:", numeric_cols)
                
                # Create histogram
                fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Add to chat button
                if st.button("Add to chat", key="add_distribution"):
                    self._add_visualization_to_chat(
                        f"Here's the distribution of {selected_col}:",
                        "chart", 
                        {
                            "chart_type": "bar",
                            "x": df[selected_col].value_counts().index.tolist(),
                            "y": df[selected_col].value_counts().tolist()
                        }
                    )
        
        elif viz_type == "Correlation Plot":
            # Select columns for correlation
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) > 1:
                selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:4])
                
                if selected_cols and len(selected_cols) > 1:
                    # Create correlation matrix
                    corr_matrix = df[selected_cols].corr()
                    
                    # Plot heatmap
                    fig = px.imshow(
                        corr_matrix, 
                        text_auto=True, 
                        title="Correlation Matrix", 
                        color_continuous_scale='RdBu_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add to chat button
                    if st.button("Add to chat", key="add_correlation"):
                        self._add_visualization_to_chat(
                            "Here's the correlation matrix between the selected columns:",
                            "chart", 
                            {
                                "chart_type": "heatmap",
                                "data": corr_matrix.to_dict()
                            }
                        )
            else:
                st.info("Need at least 2 numeric columns for correlation")
        
        elif viz_type == "Scatter Plot":
            # Select columns for scatter plot
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) >= 2:
                col_x = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
                col_y = st.selectbox("Y-axis:", numeric_cols, key="scatter_y")
                
                col_color = st.selectbox("Color by (optional):", ["None"] + df.columns.tolist(), key="scatter_color")
                color_column = None if col_color == "None" else col_color
                
                # Create scatter plot
                if color_column:
                    fig = px.scatter(df, x=col_x, y=col_y, color=color_column, title=f"{col_y} vs {col_x}")
                else:
                    fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_y} vs {col_x}")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add to chat button
                if st.button("Add to chat", key="add_scatter"):
                    self._add_visualization_to_chat(
                        f"Here's a scatter plot of {col_y} vs {col_x}:",
                        "chart", 
                        {
                            "chart_type": "scatter",
                            "x": df[col_x].tolist(),
                            "y": df[col_y].tolist()
                        }
                    )
            else:
                st.info("Need at least 2 numeric columns for scatter plot")
        
        elif viz_type == "Time Series":
            # Check for date columns
            date_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
            
            # Also check for columns that might be dates but not detected as such
            for col in df.columns:
                if col not in date_cols and 'date' in col.lower():
                    try:
                        pd.to_datetime(df[col])
                        date_cols.append(col)
                    except:
                        pass
            
            if date_cols:
                date_col = st.selectbox("Date column:", date_cols, key="time_date")
                value_col = st.selectbox("Value column:", df.select_dtypes(include=np.number).columns.tolist(), key="time_value")
                
                # Convert to datetime if needed
                if df[date_col].dtype != 'datetime64[ns]':
                    try:
                        df_plot = df.copy()
                        df_plot[date_col] = pd.to_datetime(df_plot[date_col])
                    except:
                        st.error(f"Could not convert {date_col} to datetime")
                        return
                else:
                    df_plot = df
                
                # Create time series plot
                fig = px.line(df_plot.sort_values(date_col), x=date_col, y=value_col, title=f"{value_col} over time")
                st.plotly_chart(fig, use_container_width=True)
                
                # Add to chat button
                if st.button("Add to chat", key="add_timeseries"):
                    self._add_visualization_to_chat(
                        f"Here's the trend of {value_col} over time:",
                        "chart", 
                        {
                            "chart_type": "line",
                            "x": df_plot.sort_values(date_col)[date_col].tolist(),
                            "y": df_plot.sort_values(date_col)[value_col].tolist()
                        }
                    )
            else:
                st.info("No date columns found for time series")
    
    def _render_schema_recommendations(self):
        """Render schema recommendations."""
        if st.button("Generate Schema Recommendations"):
            with st.spinner("Analyzing data and generating recommendations..."):
                # In a real implementation, this would call the API for recommendations
                # Here we'll simulate some recommendations
                try:
                    response = self._get_schema_recommendations()
                    
                    # Store recommendations
                    st.session_state.schema_recommendations = response
                    
                    # Add to chat
                    self._add_visualization_to_chat(
                        "I've analyzed your data and here are my schema recommendations:",
                        "schema",
                        response
                    )
                except Exception as e:
                    st.error(f"Error generating schema recommendations: {str(e)}")
        
        # Display existing recommendations
        if st.session_state.schema_recommendations:
            recommendations = st.session_state.schema_recommendations
            
            # Display high-level recommendations
            if "general" in recommendations:
                for rec in recommendations["general"]:
                    st.info(rec)
            
            # Display table-specific recommendations
            if "tables" in recommendations:
                for table_name, table_rec in recommendations["tables"].items():
                    with st.expander(f"Table: {table_name}"):
                        st.write("**Recommended structure:**")
                        st.json(table_rec)
    
    def _render_relationship_mapping(self):
        """Render relationship mapping between data files."""
        if st.button("Detect Relationships"):
            with st.spinner("Analyzing relationships between data files..."):
                try:
                    # In real implementation, call API
                    relationships = self._get_relationships()
                    
                    # Store relationships
                    st.session_state.data_relationships = relationships
                    
                    # Add to chat
                    self._add_visualization_to_chat(
                        "I've detected the following relationships between your data files:",
                        "schema",
                        {"relationships": relationships}
                    )
                except Exception as e:
                    st.error(f"Error detecting relationships: {str(e)}")
        
        # Display existing relationships
        if st.session_state.data_relationships:
            relationships = st.session_state.data_relationships
            
            # Display as table
            rel_data = []
            for rel in relationships:
                rel_data.append({
                    "Source File": rel["source_file"],
                    "Source Column": rel["source_column"],
                    "Target File": rel["target_file"],
                    "Target Column": rel["target_column"],
                    "Type": rel["relationship_type"],
                    "Confidence": f"{rel['confidence']:.2f}"
                })
            
            if rel_data:
                st.dataframe(pd.DataFrame(rel_data))
            else:
                st.info("No relationships detected")
    
    def _render_data_quality(self):
        """Render data quality metrics."""
        if not st.session_state.active_exploration:
            st.info("Select a data file to view quality metrics")
            return
            
        df = st.session_state.data_preview[st.session_state.active_exploration]
        
        # Calculate quality metrics
        metrics = {
            "Completion Rate": (1 - df.isna().sum() / len(df)).mean(),
            "Duplicate Rows": df.duplicated().sum() / len(df) if len(df) > 0 else 0,
        }
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Completion Rate", f"{metrics['Completion Rate']:.1%}")
        with col2:
            st.metric("Duplicate Rate", f"{metrics['Duplicate Rows']:.1%}")
        
        # Column-level quality
        st.write("### Column Quality")
        
        quality_data = []
        for column in df.columns:
            null_rate = df[column].isna().sum() / len(df) if len(df) > 0 else 0
            quality_data.append({
                "Column": column,
                "Type": str(df[column].dtype),
                "Non-Null": df[column].count(),
                "Null": df[column].isna().sum(),
                "Null %": f"{null_rate:.1%}",
                "Unique Values": df[column].nunique()
            })
        
        st.dataframe(pd.DataFrame(quality_data))
        
        if st.button("Add Quality Report to Chat"):
            # Create quality report
            report = f"## Data Quality Report for {st.session_state.active_exploration}\n\n"
            report += f"- Overall completion rate: {metrics['Completion Rate']:.1%}\n"
            report += f"- Duplicate rows: {df.duplicated().sum()} ({metrics['Duplicate Rows']:.1%})\n\n"
            
            # Add column quality
            report += "### Column Quality Issues\n\n"
            for col_data in quality_data:
                if float(col_data['Null %'].strip('%')) > 5:
                    report += f"- **{col_data['Column']}**: {col_data['Null %']} missing values\n"
            
            # Add to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": report
            })
    
    def _add_visualization_to_chat(self, message: str, viz_type: str, data: dict):
        """
        Add a visualization to the chat history.
        
        Args:
            message: Message text
            viz_type: Type of visualization
            data: Visualization data
        """
        st.session_state.messages.append({
            "role": "assistant",
            "content": message,
            "visualization": {
                "type": viz_type,
                "data": data
            }
        })
    
    def _get_schema_recommendations(self):
        """
        Get schema recommendations from API.
        
        Returns:
            Schema recommendations
        """
        try:
            # In a real implementation, this would call the API
            # For now, simulate response
            file_names = list(st.session_state.data_preview.keys())
            
            recommendations = {
                "general": [
                    "Consider normalizing your database schema to reduce data redundancy.",
                    "Add proper foreign key constraints to maintain data integrity."
                ],
                "tables": {}
            }
            
            for file_name in file_names:
                df = st.session_state.data_preview[file_name]
                table_name = file_name.split('.')[0]
                
                columns = {}
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    
                    if "int" in dtype:
                        sql_type = "INTEGER"
                    elif "float" in dtype:
                        sql_type = "DECIMAL"
                    elif "datetime" in dtype:
                        sql_type = "TIMESTAMP"
                    elif "bool" in dtype:
                        sql_type = "BOOLEAN"
                    else:
                        sql_type = "VARCHAR(255)"
                    
                    # Detect likely primary key
                    is_pk = False
                    if col.lower().endswith('_id') or col.lower() == 'id':
                        if df[col].nunique() == len(df):
                            is_pk = True
                    
                    columns[col] = {
                        "type": sql_type,
                        "nullable": df[col].isna().any(),
                        "primary_key": is_pk
                    }
                
                recommendations["tables"][table_name] = {
                    "columns": columns,
                    "primary_key": next((col for col, info in columns.items() if info["primary_key"]), None),
                    "indexes": []
                }
            
            return recommendations
        except Exception as e:
            st.error(f"Error getting schema recommendations: {str(e)}")
            return {}
    
    def _get_relationships(self):
        """
        Detect relationships between data files.
        
        Returns:
            List of relationships
        """
        try:
            # In a real implementation, this would call the API
            # For now, simulate relationship detection
            relationships = []
            file_names = list(st.session_state.data_preview.keys())
            
            for i in range(len(file_names) - 1):
                for j in range(i + 1, len(file_names)):
                    # Try to find common column names as potential relationships
                    df1 = st.session_state.data_preview[file_names[i]]
                    df2 = st.session_state.data_preview[file_names[j]]
                    
                    common_columns = set(df1.columns).intersection(set(df2.columns))
                    
                    for col in common_columns:
                        # Skip common generic column names
                        if col.lower() in ['id', 'name', 'date', 'year', 'month', 'day']:
                            continue
                            
                        # Check for potential foreign key relationship
                        if col.lower().endswith('_id') or df1[col].isin(df2[col]).mean() > 0.5:
                            # Determine relationship type
                            unique_ratio_1 = df1[col].nunique() / len(df1) if len(df1) > 0 else 0
                            unique_ratio_2 = df2[col].nunique() / len(df2) if len(df2) > 0 else 0
                            
                            if unique_ratio_1 > 0.8 and unique_ratio_2 > 0.8:
                                rel_type = "one-to-one"
                            elif unique_ratio_1 > 0.8:
                                rel_type = "one-to-many"
                            elif unique_ratio_2 > 0.8:
                                rel_type = "many-to-one"
                            else:
                                rel_type = "many-to-many"
                            
                            relationships.append({
                                "source_file": file_names[i],
                                "target_file": file_names[j],
                                "source_column": col,
                                "target_column": col,
                                "relationship_type": rel_type,
                                "confidence": 0.7
                            })
            
            return relationships
        except Exception as e:
            st.error(f"Error detecting relationships: {str(e)}")
            return []
    
    def _render_data_quality_visualization(self, df: pd.DataFrame):
        """Render data quality visualization for a dataframe."""
        with st.expander("Data Quality Analysis", expanded=True):
            st.write("Analyzing data quality metrics across columns...")
            
            # Create data quality chart
            fig = create_data_quality_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add quality metrics to conversation
            if st.button("Add to Conversation", key="add_quality_viz"):
                quality_data = {
                    "type": "data_quality",
                    "timestamp": time.time(),
                    "data": {
                        "missing_values": df.isna().sum().to_dict(),
                        "total_rows": len(df)
                    }
                }
                st.session_state.visualization_history.append(quality_data)
                st.success("Data quality visualization added to conversation!")
    
    def _render_correlation_visualization(self, df: pd.DataFrame):
        """Render correlation visualization for numeric columns."""
        with st.expander("Correlation Analysis", expanded=False):
            # Threshold slider
            threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.3, 0.05)
            
            # Only proceed if we have numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
                return
                
            # Create correlation chart
            fig = create_correlations_chart(df, threshold)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add to conversation
            if st.button("Add to Conversation", key="add_corr_viz"):
                corr_data = {
                    "type": "correlation",
                    "timestamp": time.time(),
                    "data": {
                        "correlation_matrix": df.corr().to_dict(),
                        "threshold": threshold
                    }
                }
                st.session_state.visualization_history.append(corr_data)
                st.success("Correlation analysis added to conversation!")
    
    def _render_distribution_visualization(self, df: pd.DataFrame):
        """Render distribution visualizations for selected columns."""
        with st.expander("Distribution Analysis", expanded=False):
            # Column selection
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect(
                "Select columns for distribution analysis", 
                all_columns,
                default=all_columns[:min(3, len(all_columns))]
            )
            
            if not selected_columns:
                st.info("Please select at least one column.")
                return
                
            # Create distribution plots
            distribution_figs = create_distribution_plots(df, selected_columns)
            
            # Display plots
            for col, fig in distribution_figs.items():
                st.subheader(f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button(f"Add to Conversation", key=f"add_dist_{col}"):
                    dist_data = {
                        "type": "distribution",
                        "timestamp": time.time(),
                        "column": col,
                        "data": {
                            "column": col,
                            "summary_stats": df[col].describe().to_dict() if pd.api.types.is_numeric_dtype(df[col]) else None,
                            "value_counts": df[col].value_counts().to_dict() if not pd.api.types.is_numeric_dtype(df[col]) else None
                        }
                    }
                    st.session_state.visualization_history.append(dist_data)
                    st.success(f"Distribution of {col} added to conversation!")
    
    def _render_anomaly_detection(self, df: pd.DataFrame):
        """Render anomaly detection for numeric columns."""
        with st.expander("Anomaly Detection", expanded=False):
            # Only show for numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns available for anomaly detection.")
                return
                
            # Column selection
            selected_column = st.selectbox(
                "Select column for anomaly detection",
                numeric_cols
            )
            
            if not selected_column:
                return
                
            # Create anomaly detection chart
            fig = create_anomaly_detection_chart(df, selected_column)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add to conversation
            if st.button("Add to Conversation", key="add_anomaly_viz"):
                # Calculate outlier boundaries using IQR method
                q1 = df[selected_column].quantile(0.25)
                q3 = df[selected_column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Identify anomalies
                anomalies = df[(df[selected_column] < lower_bound) | (df[selected_column] > upper_bound)]
                
                anomaly_data = {
                    "type": "anomaly",
                    "timestamp": time.time(),
                    "column": selected_column,
                    "data": {
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "anomaly_count": len(anomalies),
                        "anomaly_percentage": len(anomalies)/len(df)*100
                    }
                }
                st.session_state.visualization_history.append(anomaly_data)
                st.success(f"Anomaly detection for {selected_column} added to conversation!")
    
    def _render_data_comparison(self, df: pd.DataFrame):
        """Render data comparison with previous datasets if available."""
        with st.expander("Data Comparison", expanded=False):
            # Check if we have previous datasets stored
            if "previous_datasets" not in st.session_state:
                st.info("No previous datasets available for comparison.")
                return
                
            previous_datasets = st.session_state.previous_datasets
            if not previous_datasets:
                st.info("No previous datasets available for comparison.")
                return
                
            # Dataset selection
            dataset_names = list(previous_datasets.keys())
            selected_dataset = st.selectbox(
                "Select dataset to compare with",
                dataset_names
            )
            
            if not selected_dataset:
                return
                
            df2 = previous_datasets[selected_dataset]
            
            # Find common columns
            common_cols = list(set(df.columns) & set(df2.columns))
            if not common_cols:
                st.warning("No common columns found between datasets.")
                return
                
            # Column selection
            selected_column = st.selectbox(
                "Select column to compare",
                common_cols
            )
            
            if not selected_column:
                return
                
            # Create comparison chart
            fig = create_data_comparison_chart(df, df2, selected_column)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add to conversation
            if st.button("Add to Conversation", key="add_comparison_viz"):
                comparison_data = {
                    "type": "comparison",
                    "timestamp": time.time(),
                    "column": selected_column,
                    "data": {
                        "dataset1_name": "Current Dataset",
                        "dataset2_name": selected_dataset,
                        "column": selected_column
                    }
                }
                st.session_state.visualization_history.append(comparison_data)
                st.success(f"Comparison of {selected_column} added to conversation!")
    
    def _render_data_profile(self, df: pd.DataFrame):
        """Render data profile summary."""
        with st.expander("Data Profile Summary", expanded=True):
            # Create data profile summary
            profile = create_data_profile_summary(df)
            
            # Basic metrics
            st.subheader("Basic Metrics")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Rows", profile["row_count"])
            with metrics_col2:
                st.metric("Columns", profile["column_count"])
            with metrics_col3:
                st.metric("Memory Usage", f"{profile['memory_usage']:.2f} MB")
                
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Missing Values", f"{profile['missing_values_pct']:.2f}%")
            with metrics_col2:
                st.metric("Duplicate Rows", f"{profile['duplicate_rows_pct']:.2f}%")
            
            # Column type breakdown
            st.subheader("Column Types")
            col_types = profile["column_types"]
            col_type_fig = go.Figure(data=[
                go.Pie(
                    labels=list(col_types.keys()),
                    values=list(col_types.values()),
                    hole=.3
                )
            ])
            col_type_fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(col_type_fig, use_container_width=True)
            
            # Add to conversation
            if st.button("Add to Conversation", key="add_profile_viz"):
                profile_data = {
                    "type": "profile",
                    "timestamp": time.time(),
                    "data": profile
                }
                st.session_state.visualization_history.append(profile_data)
                st.success("Data profile summary added to conversation!")
