import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any

def generate_histogram(df: pd.DataFrame, column: str, bins: int = 20, 
                       log_scale: bool = False, kde: bool = True) -> Dict[str, Any]:
    """
    Generate a histogram for a numeric column.
    
    Args:
        df: Input DataFrame
        column: Column name to plot
        bins: Number of bins
        log_scale: Whether to use log scale on y-axis
        kde: Whether to overlay KDE
        
    Returns:
        Dictionary with plotly figure and statistics
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset"}
    
    if df[column].dtype.kind not in 'ifc':
        return {"error": f"Column '{column}' is not numeric"}
    
    # Calculate statistics
    stats = {
        "mean": float(df[column].mean()),
        "median": float(df[column].median()),
        "std": float(df[column].std()),
        "min": float(df[column].min()),
        "max": float(df[column].max()),
        "skew": float(df[column].skew()),
        "kurtosis": float(df[column].kurtosis()),
        "missing_count": int(df[column].isnull().sum()),
        "missing_percent": float((df[column].isnull().sum() / len(df)) * 100)
    }
    
    # Create plotly figure
    fig = px.histogram(
        df, 
        x=column,
        nbins=bins,
        marginal="box" if not kde else "violin",
        title=f"Distribution of {column}",
        opacity=0.7,
        log_y=log_scale,
        color_discrete_sequence=['#636EFA']
    )
    
    # Add mean and median lines
    fig.add_vline(x=stats["mean"], line_dash="dash", line_color="red", 
                 annotation_text="Mean", annotation_position="top right")
    fig.add_vline(x=stats["median"], line_dash="dash", line_color="green", 
                 annotation_text="Median", annotation_position="top left")
    
    # Update layout
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
        template="plotly_white",
        height=500
    )
    
    return {
        "figure": fig,
        "stats": stats,
        "plot_type": "histogram"
    }

def generate_boxplot(df: pd.DataFrame, column: str, 
                    group_by: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a boxplot for a numeric column, optionally grouped by a categorical column.
    
    Args:
        df: Input DataFrame
        column: Numeric column name to plot
        group_by: Optional categorical column to group by
        
    Returns:
        Dictionary with plotly figure and statistics
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset"}
    
    if df[column].dtype.kind not in 'ifc':
        return {"error": f"Column '{column}' is not numeric"}
    
    if group_by and group_by not in df.columns:
        return {"error": f"Column '{group_by}' not found in dataset"}
    
    # Calculate statistics
    stats = {
        "mean": float(df[column].mean()),
        "median": float(df[column].median()),
        "q1": float(df[column].quantile(0.25)),
        "q3": float(df[column].quantile(0.75)),
        "iqr": float(df[column].quantile(0.75) - df[column].quantile(0.25)),
        "outliers_count": len(df[(df[column] < df[column].quantile(0.25) - 1.5 * (df[column].quantile(0.75) - df[column].quantile(0.25))) | 
                              (df[column] > df[column].quantile(0.75) + 1.5 * (df[column].quantile(0.75) - df[column].quantile(0.25)))])
    }
    
    # Create plotly figure
    if group_by:
        fig = px.box(
            df, 
            x=group_by,
            y=column,
            title=f"Boxplot of {column} by {group_by}",
            color=group_by,
            notched=True,
            points="outliers"
        )
    else:
        fig = px.box(
            df, 
            y=column,
            title=f"Boxplot of {column}",
            notched=True,
            points="outliers"
        )
    
    # Update layout
    fig.update_layout(
        yaxis_title=column,
        template="plotly_white",
        height=500
    )
    
    return {
        "figure": fig,
        "stats": stats,
        "plot_type": "boxplot"
    }

def generate_scatter_plot(df: pd.DataFrame, x_column: str, y_column: str, 
                         color_by: Optional[str] = None, size_by: Optional[str] = None,
                         trendline: bool = False) -> Dict[str, Any]:
    """
    Generate a scatter plot between two numeric columns.
    
    Args:
        df: Input DataFrame
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        color_by: Optional column to color points by
        size_by: Optional column to size points by
        trendline: Whether to add a trendline
        
    Returns:
        Dictionary with plotly figure and statistics
    """
    if x_column not in df.columns or y_column not in df.columns:
        return {"error": f"One or both columns not found in dataset"}
    
    if df[x_column].dtype.kind not in 'ifc' or df[y_column].dtype.kind not in 'ifc':
        return {"error": f"Both columns must be numeric"}
    
    if color_by and color_by not in df.columns:
        return {"error": f"Column '{color_by}' not found in dataset"}
    
    if size_by and size_by not in df.columns:
        return {"error": f"Column '{size_by}' not found in dataset"}
    
    # Calculate statistics
    correlation = df[[x_column, y_column]].corr().iloc[0, 1]
    
    stats = {
        "correlation": float(correlation),
        "x_mean": float(df[x_column].mean()),
        "y_mean": float(df[y_column].mean()),
        "x_std": float(df[x_column].std()),
        "y_std": float(df[y_column].std())
    }
    
    # Create plotly figure
    fig = px.scatter(
        df, 
        x=x_column,
        y=y_column,
        color=color_by,
        size=size_by,
        title=f"Scatter Plot: {x_column} vs {y_column}",
        trendline="ols" if trendline else None,
        opacity=0.7,
        hover_data=df.columns
    )
    
    # Add correlation annotation
    fig.add_annotation(
        x=0.95,
        y=0.05,
        xref="paper",
        yref="paper",
        text=f"Correlation: {correlation:.2f}",
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        template="plotly_white",
        height=500
    )
    
    return {
        "figure": fig,
        "stats": stats,
        "plot_type": "scatter"
    }

def generate_correlation_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None,
                                method: str = "pearson") -> Dict[str, Any]:
    """
    Generate a correlation heatmap for numeric columns in the dataframe.
    
    Args:
        df: The dataframe to analyze
        columns: Optional list of columns to include. If None, all numeric columns are used.
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        Dictionary with figure and statistics
    """
    # Validate inputs
    if df.empty:
        return {"error": "Empty dataframe"}
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        return {"error": "No numeric columns found for correlation analysis"}
    
    # Filter columns if specified
    if columns:
        numeric_cols = [col for col in columns if col in numeric_cols]
        
        if not numeric_cols:
            return {"error": "None of the specified columns are numeric"}
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr(method=method)
    
    # Find highest correlations (excluding self-correlations)
    corr_values = corr_matrix.unstack()
    corr_values = corr_values[corr_values < 1.0]  # Remove self-correlations
    highest_corrs = corr_values.sort_values(ascending=False)[:5]
    
    stats = {
        "highest_correlations": {f"{idx[0]}__{idx[1]}": val for idx, val in highest_corrs.items()},
        "method": method,
        "num_features": len(numeric_cols)
    }
    
    # Create a custom heatmap using go.Heatmap instead of px.imshow to avoid np.bool deprecation
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmin=-1, zmax=1
    ))
    
    # Add text annotations manually
    annotations = []
    for i, row in enumerate(corr_matrix.index):
        for j, col in enumerate(corr_matrix.columns):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{corr_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")
                )
            )
    
    # Update layout with title and annotations
    fig.update_layout(
        title=f"Correlation Heatmap ({method.capitalize()})",
        template="plotly_white",
        height=600,
        width=700,
        annotations=annotations
    )
    
    return {
        "figure": fig,
        "stats": stats,
        "type": "correlation_heatmap"
    }

def generate_bar_chart(df: pd.DataFrame, column: str, 
                      top_n: int = 10, horizontal: bool = False) -> Dict[str, Any]:
    """
    Generate a bar chart for a categorical column.
    
    Args:
        df: Input DataFrame
        column: Categorical column name
        top_n: Number of top categories to show
        horizontal: Whether to use horizontal bars
        
    Returns:
        Dictionary with plotly figure and statistics
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset"}
    
    # Get value counts
    value_counts = df[column].value_counts().head(top_n)
    
    stats = {
        "unique_count": df[column].nunique(),
        "top_categories": value_counts.to_dict(),
        "missing_count": int(df[column].isnull().sum()),
        "missing_percent": float((df[column].isnull().sum() / len(df)) * 100)
    }
    
    # Create plotly figure
    if horizontal:
        fig = px.bar(
            x=value_counts.values,
            y=value_counts.index,
            title=f"Top {top_n} Categories in {column}",
            orientation='h',
            color=value_counts.values,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(yaxis_title=column, xaxis_title="Count")
    else:
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Top {top_n} Categories in {column}",
            color=value_counts.values,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(xaxis_title=column, yaxis_title="Count")
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        height=500,
        coloraxis_showscale=False
    )
    
    return {
        "figure": fig,
        "stats": stats,
        "plot_type": "bar"
    }

def generate_pie_chart(df: pd.DataFrame, column: str, top_n: int = 10) -> Dict[str, Any]:
    """
    Generate a pie chart for a categorical column.
    
    Args:
        df: Input DataFrame
        column: Categorical column name
        top_n: Number of top categories to show
        
    Returns:
        Dictionary with plotly figure and statistics
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset"}
    
    # Get value counts
    value_counts = df[column].value_counts().head(top_n)
    
    # If there are more categories, add an "Other" category
    if df[column].nunique() > top_n:
        other_count = df[column].value_counts().iloc[top_n:].sum()
        value_counts["Other"] = other_count
    
    stats = {
        "unique_count": df[column].nunique(),
        "top_categories": value_counts.to_dict(),
        "missing_count": int(df[column].isnull().sum()),
        "missing_percent": float((df[column].isnull().sum() / len(df)) * 100)
    }
    
    # Create plotly figure
    fig = px.pie(
        names=value_counts.index,
        values=value_counts.values,
        title=f"Distribution of {column}",
        hole=0.4
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        height=500
    )
    
    # Update traces
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return {
        "figure": fig,
        "stats": stats,
        "plot_type": "pie"
    }

def generate_line_plot(df: pd.DataFrame, x_column: str, y_column: str,
                      group_by: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a line plot between two columns.
    
    Args:
        df: Input DataFrame
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        group_by: Optional column to group by
        
    Returns:
        Dictionary with plotly figure and statistics
    """
    if x_column not in df.columns or y_column not in df.columns:
        return {"error": f"One or both columns not found in dataset"}
    
    if group_by and group_by not in df.columns:
        return {"error": f"Column '{group_by}' not found in dataset"}
    
    # Create plotly figure
    if group_by:
        fig = px.line(
            df, 
            x=x_column,
            y=y_column,
            color=group_by,
            title=f"Line Plot: {y_column} vs {x_column} by {group_by}",
            markers=True
        )
    else:
        fig = px.line(
            df, 
            x=x_column,
            y=y_column,
            title=f"Line Plot: {y_column} vs {x_column}",
            markers=True
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        template="plotly_white",
        height=500
    )
    
    # Calculate basic statistics
    stats = {
        "x_min": float(df[x_column].min()),
        "x_max": float(df[x_column].max()),
        "y_min": float(df[y_column].min()),
        "y_max": float(df[y_column].max()),
        "y_mean": float(df[y_column].mean()),
        "trend": "increasing" if df[y_column].corr(df[x_column]) > 0 else "decreasing"
    }
    
    return {
        "figure": fig,
        "stats": stats,
        "plot_type": "line"
    }

def generate_pairplot(df: pd.DataFrame, columns: List[str], 
                     hue: Optional[str] = None, n_samples: int = 1000) -> Dict[str, Any]:
    """
    Generate a pairplot for multiple numeric columns.
    
    Args:
        df: Input DataFrame
        columns: List of numeric columns to include
        hue: Optional categorical column for coloring
        n_samples: Number of samples to use (for performance)
        
    Returns:
        Dictionary with plotly figure and statistics
    """
    if not all(col in df.columns for col in columns):
        return {"error": "One or more columns not found in dataset"}
    
    if hue and hue not in df.columns:
        return {"error": f"Column '{hue}' not found in dataset"}
    
    # Sample data if needed
    if len(df) > n_samples:
        df_sample = df.sample(n_samples, random_state=42)
    else:
        df_sample = df
    
    # Create pairplot using plotly
    fig = make_subplots(rows=len(columns), cols=len(columns), 
                       shared_xaxes=True, shared_yaxes=True,
                       subplot_titles=[f"{col1} vs {col2}" for col1 in columns for col2 in columns])
    
    # Calculate correlations
    corr_matrix = df[columns].corr()
    
    # Add traces for each subplot
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:  # Diagonal - histogram
                fig.add_trace(
                    go.Histogram(x=df_sample[col1], name=col1),
                    row=i+1, col=j+1
                )
            else:  # Off-diagonal - scatter plot
                if hue:
                    for hue_val in df_sample[hue].unique():
                        mask = df_sample[hue] == hue_val
                        fig.add_trace(
                            go.Scatter(
                                x=df_sample[col2][mask],
                                y=df_sample[col1][mask],
                                mode='markers',
                                marker=dict(size=5, opacity=0.5),
                                name=f"{hue}={hue_val}"
                            ),
                            row=i+1, col=j+1
                        )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=df_sample[col2],
                            y=df_sample[col1],
                            mode='markers',
                            marker=dict(size=5, opacity=0.5)
                        ),
                        row=i+1, col=j+1
                    )
    
    # Update layout
    fig.update_layout(
        title="Pairplot",
        template="plotly_white",
        height=200 * len(columns),
        width=200 * len(columns),
        showlegend=False
    )
    
    stats = {
        "correlations": corr_matrix.to_dict(),
        "num_features": len(columns)
    }
    
    return {
        "figure": fig,
        "stats": stats,
        "plot_type": "pairplot"
    }

def generate_count_plot(df: pd.DataFrame, column: str, 
                       hue: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a count plot for a categorical column.
    
    Args:
        df: Input DataFrame
        column: Categorical column name
        hue: Optional categorical column for grouping
        
    Returns:
        Dictionary with plotly figure and statistics
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset"}
    
    if hue and hue not in df.columns:
        return {"error": f"Column '{hue}' not found in dataset"}
    
    # Create plotly figure
    if hue:
        fig = px.histogram(
            df,
            x=column,
            color=hue,
            barmode="group",
            title=f"Count Plot of {column} by {hue}",
            category_orders={column: df[column].value_counts().index.tolist()}
        )
    else:
        fig = px.histogram(
            df,
            x=column,
            title=f"Count Plot of {column}",
            category_orders={column: df[column].value_counts().index.tolist()}
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
        template="plotly_white",
        height=500
    )
    
    # Calculate statistics
    stats = {
        "unique_count": df[column].nunique(),
        "value_counts": df[column].value_counts().to_dict(),
        "missing_count": int(df[column].isnull().sum()),
        "missing_percent": float((df[column].isnull().sum() / len(df)) * 100)
    }
    
    return {
        "figure": fig,
        "stats": stats,
        "plot_type": "count"
    }

def generate_violin_plot(df: pd.DataFrame, column: str, 
                        group_by: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a violin plot for a numeric column.
    
    Args:
        df: Input DataFrame
        column: Numeric column name
        group_by: Optional categorical column for grouping
        
    Returns:
        Dictionary with plotly figure and statistics
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset"}
    
    if df[column].dtype.kind not in 'ifc':
        return {"error": f"Column '{column}' is not numeric"}
    
    if group_by and group_by not in df.columns:
        return {"error": f"Column '{group_by}' not found in dataset"}
    
    # Create plotly figure
    if group_by:
        fig = px.violin(
            df,
            x=group_by,
            y=column,
            color=group_by,
            box=True,
            points="all",
            title=f"Violin Plot of {column} by {group_by}"
        )
    else:
        fig = px.violin(
            df,
            y=column,
            box=True,
            points="all",
            title=f"Violin Plot of {column}"
        )
    
    # Update layout
    fig.update_layout(
        yaxis_title=column,
        template="plotly_white",
        height=500
    )
    
    # Calculate statistics
    stats = {
        "mean": float(df[column].mean()),
        "median": float(df[column].median()),
        "std": float(df[column].std()),
        "skew": float(df[column].skew()),
        "kurtosis": float(df[column].kurtosis())
    }
    
    if group_by:
        group_stats = {}
        for group in df[group_by].unique():
            group_data = df[df[group_by] == group][column]
            group_stats[str(group)] = {
                "mean": float(group_data.mean()),
                "median": float(group_data.median()),
                "std": float(group_data.std())
            }
        stats["group_stats"] = group_stats
    
    return {
        "figure": fig,
        "stats": stats,
        "plot_type": "violin"
    }

def get_available_plots(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get available plot types for each column in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping column names to list of applicable plot types
    """
    available_plots = {}
    
    for col in df.columns:
        plots = []
        
        # Check column type
        if df[col].dtype.kind in 'ifc':  # numeric
            plots.extend(["histogram", "boxplot", "violin"])
            
            # Add plots that require another numeric column
            if len(df.select_dtypes(include=['number']).columns) > 1:
                plots.extend(["scatter", "line"])
            
            # Add plots that can use grouping if categorical columns exist
            if len(df.select_dtypes(include=['object', 'category']).columns) > 0:
                plots.extend(["grouped_boxplot", "grouped_violin"])
        
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':  # categorical
            plots.extend(["bar", "pie", "count"])
            
            # Add plots that can use this column for grouping if numeric columns exist
            if len(df.select_dtypes(include=['number']).columns) > 0:
                plots.extend(["grouped_histogram", "grouped_boxplot"])
        
        available_plots[col] = plots
    
    # Add special plot types that use multiple columns
    if len(df.select_dtypes(include=['number']).columns) > 1:
        available_plots["_special"] = ["correlation_heatmap", "pairplot"]
    
    return available_plots

def generate_plot(df: pd.DataFrame, plot_type: str, config: Dict) -> Dict[str, Any]:
    """
    Generate a plot based on type and configuration.
    
    Args:
        df: Input DataFrame
        plot_type: Type of plot to generate
        config: Plot configuration
        
    Returns:
        Dictionary with plot data
    """
    if plot_type == "histogram":
        return generate_histogram(
            df, 
            column=config.get("column"),
            bins=config.get("bins", 20),
            log_scale=config.get("log_scale", False),
            kde=config.get("kde", True)
        )
    
    elif plot_type == "boxplot":
        return generate_boxplot(
            df,
            column=config.get("column"),
            group_by=config.get("group_by")
        )
    
    elif plot_type == "scatter":
        return generate_scatter_plot(
            df,
            x_column=config.get("x_column"),
            y_column=config.get("y_column"),
            color_by=config.get("color_by"),
            size_by=config.get("size_by"),
            trendline=config.get("trendline", False)
        )
    
    elif plot_type == "correlation_heatmap":
        return generate_correlation_heatmap(
            df,
            columns=config.get("columns"),
            method=config.get("method", "pearson")
        )
    
    elif plot_type == "bar":
        return generate_bar_chart(
            df,
            column=config.get("column"),
            top_n=config.get("top_n", 10),
            horizontal=config.get("horizontal", False)
        )
    
    elif plot_type == "pie":
        return generate_pie_chart(
            df,
            column=config.get("column"),
            top_n=config.get("top_n", 10)
        )
    
    elif plot_type == "line":
        return generate_line_plot(
            df,
            x_column=config.get("x_column"),
            y_column=config.get("y_column"),
            group_by=config.get("group_by")
        )
    
    elif plot_type == "pairplot":
        return generate_pairplot(
            df,
            columns=config.get("columns"),
            hue=config.get("hue"),
            n_samples=config.get("n_samples", 1000)
        )
    
    elif plot_type == "count":
        return generate_count_plot(
            df,
            column=config.get("column"),
            hue=config.get("hue")
        )
    
    elif plot_type == "violin":
        return generate_violin_plot(
            df,
            column=config.get("column"),
            group_by=config.get("group_by")
        )
    
    else:
        return {"error": f"Unknown plot type: {plot_type}"}
