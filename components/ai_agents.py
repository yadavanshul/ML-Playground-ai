import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
import openai
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key or openai_api_key == "your_api_key_here":
    logger.warning("No OpenAI API key found. Using mock functionality.")
    USE_MOCK = True
else:
    openai.api_key = openai_api_key
    USE_MOCK = False

# Helper function to check if an API key is valid
def is_valid_api_key(api_key):
    return api_key and isinstance(api_key, str) and api_key.startswith("sk-")

# Helper function to convert numpy types to Python native types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# Initialize ChromaDB client with optional embedding function
try:
    client = chromadb.PersistentClient(path="./ai_eda_pipeline/chromadb_store")
    
    # Create embedding function if API key is available
    if not USE_MOCK:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-ada-002"
        )
    else:
        # Use a simple mock embedding function when no API key is available
        class MockEmbeddingFunction:
            def __call__(self, texts):
                # Return a simple mock embedding for each text
                return [[0.1] * 768 for _ in texts]
        
        openai_ef = MockEmbeddingFunction()
    
    # Create collections if they don't exist
    try:
        dataset_collection = client.get_collection("dataset_metadata")
    except:
        dataset_collection = client.create_collection(
            name="dataset_metadata",
            embedding_function=openai_ef
        )

    try:
        insights_collection = client.get_collection("ai_insights")
    except:
        insights_collection = client.create_collection(
            name="ai_insights",
            embedding_function=openai_ef
        )

    try:
        preprocessing_collection = client.get_collection("preprocessing_steps")
    except:
        preprocessing_collection = client.create_collection(
            name="preprocessing_steps",
            embedding_function=openai_ef
        )
    
    CHROMA_INITIALIZED = True
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {str(e)}")
    CHROMA_INITIALIZED = False

class MainAIAgent:
    """
    Main AI agent that coordinates the mini agents and provides high-level insights.
    """
    
    def __init__(self, api_key=None):
        self.dataset_metadata = None
        self.dataset_name = None
        
        # Configure OpenAI API key if provided
        self.api_key = api_key
        if is_valid_api_key(api_key):
            openai.api_key = api_key
            self.use_mock = False
        else:
            self.use_mock = USE_MOCK
        
        # Initialize mini agents with the API key
        self.mini_agents = {
            "eda": EDAMiniAgent(api_key=api_key),
            "preprocessing": PreprocessingMiniAgent(api_key=api_key),
            "error_detection": ErrorDetectionAgent(api_key=api_key)
        }
    
    def analyze_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Analyze a dataset and store its metadata in ChromaDB.
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset analysis
        """
        from ..utils.data_utils import get_dataset_metadata, detect_data_issues, suggest_preprocessing_steps
        
        # Get dataset metadata
        self.dataset_metadata = get_dataset_metadata(df)
        self.dataset_name = dataset_name
        
        # Detect data issues
        data_issues = detect_data_issues(df)
        
        # Get preprocessing suggestions
        preprocessing_suggestions = suggest_preprocessing_steps(df, data_issues)
        
        # Combine metadata, issues, and suggestions
        analysis = {
            "metadata": self.dataset_metadata,
            "issues": data_issues,
            "suggestions": preprocessing_suggestions
        }
        
        # Store in ChromaDB if initialized
        if CHROMA_INITIALIZED:
            try:
                # Convert numpy types to Python native types before serialization
                serializable_analysis = convert_numpy_types(analysis)
                metadata_str = json.dumps(serializable_analysis)
                
                # Create a summary for embedding
                summary = f"Dataset: {dataset_name}\n"
                summary += f"Shape: {self.dataset_metadata['shape']}\n"
                summary += f"Columns: {', '.join(self.dataset_metadata['columns'])}\n"
                summary += f"Numeric columns: {', '.join(self.dataset_metadata['numeric_columns'])}\n"
                summary += f"Categorical columns: {', '.join(self.dataset_metadata['categorical_columns'])}\n"
                
                # Add issues summary
                if data_issues["missing_values"]:
                    summary += f"Missing values in columns: {', '.join(data_issues['missing_values'].keys())}\n"
                if data_issues["outliers"]:
                    summary += f"Outliers in columns: {', '.join(data_issues['outliers'].keys())}\n"
                if data_issues["high_correlation"]:
                    corr_pairs = [f"{item['col1']}-{item['col2']}" for item in data_issues["high_correlation"]]
                    summary += f"High correlation between: {', '.join(corr_pairs)}\n"
                
                # Store in ChromaDB
                dataset_collection.upsert(
                    ids=[dataset_name],
                    documents=[summary],
                    metadatas=[{"full_metadata": metadata_str, "timestamp": time.time()}]
                )
            except Exception as e:
                logger.error(f"Error storing dataset in ChromaDB: {str(e)}")
        
        return analysis
    
    def get_dataset_summary(self) -> str:
        """
        Get a human-readable summary of the dataset.
        
        Returns:
            String with dataset summary
        """
        if not self.dataset_metadata:
            return "No dataset has been analyzed yet."
        
        summary = f"## Dataset Summary: {self.dataset_name}\n\n"
        
        # Basic info
        summary += f"- **Rows**: {self.dataset_metadata['shape'][0]}\n"
        summary += f"- **Columns**: {self.dataset_metadata['shape'][1]}\n\n"
        
        # Column types
        summary += f"- **Numeric Columns**: {len(self.dataset_metadata['numeric_columns'])}\n"
        summary += f"- **Categorical Columns**: {len(self.dataset_metadata['categorical_columns'])}\n"
        summary += f"- **Datetime Columns**: {len(self.dataset_metadata['datetime_columns'])}\n\n"
        
        # Missing values
        missing_cols = {col: count for col, count in self.dataset_metadata["missing_values"].items() if count > 0}
        if missing_cols:
            summary += "### Missing Values\n"
            for col, count in missing_cols.items():
                percent = (count / self.dataset_metadata['shape'][0]) * 100
                summary += f"- **{col}**: {count} ({percent:.1f}%)\n"
            summary += "\n"
        
        return summary
    
    def get_similar_datasets(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Find similar datasets based on a query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar datasets
        """
        results = dataset_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        similar_datasets = []
        for i, (id, metadata, distance) in enumerate(zip(results["ids"][0], results["metadatas"][0], results["distances"][0])):
            similar_datasets.append({
                "dataset_name": id,
                "similarity": 1 - distance,
                "metadata": json.loads(metadata["full_metadata"])
            })
        
        return similar_datasets
    
    def get_eda_agent(self):
        """Get the EDA mini agent."""
        return self.mini_agents["eda"]
    
    def get_preprocessing_agent(self):
        """Get the preprocessing mini agent."""
        return self.mini_agents["preprocessing"]
    
    def get_error_detection_agent(self):
        """Get the error detection mini agent."""
        return self.mini_agents["error_detection"]
    
    def recommend_visualizations(self, df: pd.DataFrame) -> List[Dict]:
        """
        Recommend optimal visualizations based on dataset characteristics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of recommended visualization configurations
        """
        if df is None or self.dataset_metadata is None:
            return []
        
        recommendations = []
        
        # If using mock functionality, return rule-based recommendations
        if self.use_mock:
            return self._generate_enhanced_mock_recommendations(df)
        
        try:
            # Get column types
            numeric_cols = self.dataset_metadata['numeric_columns']
            categorical_cols = self.dataset_metadata['categorical_columns']
            datetime_cols = self.dataset_metadata['datetime_columns']
            
            # Analyze column statistics for better recommendations
            column_stats = {}
            for col in numeric_cols:
                column_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "skew": float(df[col].skew()) if hasattr(df[col], 'skew') else 0,
                    "unique_ratio": float(df[col].nunique() / len(df)),
                    "missing_ratio": float(df[col].isna().mean())
                }
            
            # 1. For numeric columns with high skewness, recommend histograms with KDE
            for col in numeric_cols:
                if abs(column_stats[col]["skew"]) > 1.0:  # Significant skewness
                    recommendations.append({
                        "type": "histogram",
                        "config": {
                            "column": col,
                            "bins": 25,
                            "kde": True,
                            "log_scale": column_stats[col]["skew"] > 2.0  # Use log scale for highly skewed data
                        },
                        "reason": f"Histogram with KDE recommended for {col} which shows {'positive' if column_stats[col]['skew'] > 0 else 'negative'} skewness ({column_stats[col]['skew']:.2f})"
                    })
            
            # 2. For numeric columns with potential outliers, recommend boxplots
            if "issues" in self.dataset_metadata and "outliers" in self.dataset_metadata["issues"]:
                for col, info in self.dataset_metadata["issues"]["outliers"].items():
                    if info["percent"] > 0.5:  # If more than 0.5% outliers
                        # If we have categorical columns, recommend grouped boxplots
                        if categorical_cols and len(categorical_cols) > 0:
                            # Find a categorical column with reasonable cardinality
                            for cat_col in categorical_cols:
                                if df[cat_col].nunique() <= 8:  # Not too many categories
                                    recommendations.append({
                                        "type": "boxplot",
                                        "config": {
                                            "column": col,
                                            "group_by": cat_col
                                        },
                                        "reason": f"Boxplot grouped by {cat_col} recommended to visualize outliers in {col} ({info['percent']:.1f}% outliers) across different categories"
                                    })
                                    break
                            else:  # No suitable categorical column found
                                recommendations.append({
                                    "type": "boxplot",
                                    "config": {"column": col},
                                    "reason": f"Boxplot recommended to visualize outliers in {col} ({info['percent']:.1f}% outliers detected)"
                                })
                        else:
                            recommendations.append({
                                "type": "boxplot",
                                "config": {"column": col},
                                "reason": f"Boxplot recommended to visualize outliers in {col} ({info['percent']:.1f}% outliers detected)"
                            })
            
            # 3. For high correlation pairs, recommend scatter plots with trendlines
            if "issues" in self.dataset_metadata and "high_correlation" in self.dataset_metadata["issues"]:
                for i, corr_info in enumerate(self.dataset_metadata["issues"]["high_correlation"][:3]):  # Top 3 correlations
                    col1, col2 = corr_info["col1"], corr_info["col2"]
                    
                    # If we have categorical columns, recommend colored scatter plots
                    color_by = None
                    if categorical_cols:
                        for cat_col in categorical_cols:
                            if df[cat_col].nunique() <= 6:  # Not too many categories for coloring
                                color_by = cat_col
                                break
                    
                    recommendations.append({
                        "type": "scatter",
                        "config": {
                            "x_column": col1,
                            "y_column": col2,
                            "trendline": True,
                            "color_by": color_by
                        },
                        "reason": f"Scatter plot with trendline recommended to visualize strong {'positive' if corr_info['correlation'] > 0 else 'negative'} correlation ({corr_info['correlation']:.2f}) between {col1} and {col2}" + (f", colored by {color_by}" if color_by else "")
                    })
            
            # 4. For categorical columns with reasonable cardinality, recommend bar charts
            for col in categorical_cols:
                nunique = df[col].nunique()
                if nunique <= 15:  # Not too many categories
                    # If we have numeric columns, recommend bar charts with numeric values
                    if numeric_cols:
                        # Find the most relevant numeric column (e.g., one with least missing values)
                        best_numeric_col = min(numeric_cols, key=lambda x: column_stats.get(x, {}).get("missing_ratio", 1.0))
                        
                        recommendations.append({
                            "type": "bar",
                            "config": {
                                "column": col, 
                                "top_n": min(nunique, 12),
                                "horizontal": nunique > 8  # Use horizontal bars for many categories
                            },
                            "reason": f"Bar chart recommended to visualize distribution of {col} categories (showing top {min(nunique, 12)} categories)"
                        })
                    else:
                        recommendations.append({
                            "type": "bar",
                            "config": {
                                "column": col, 
                                "top_n": min(nunique, 12),
                                "horizontal": nunique > 8
                            },
                            "reason": f"Bar chart recommended to visualize distribution of {col} categories"
                        })
            
            # 5. For numeric columns, recommend histograms (if not already added due to skewness)
            added_histograms = set(rec["config"]["column"] for rec in recommendations if rec["type"] == "histogram")
            for col in numeric_cols:
                if col not in added_histograms:
                    recommendations.append({
                        "type": "histogram",
                        "config": {"column": col, "bins": 20, "kde": True},
                        "reason": f"Histogram recommended to visualize distribution of {col}"
                    })
            
            # 6. Always recommend correlation heatmap for datasets with multiple numeric columns
            if len(numeric_cols) > 3:
                # If too many numeric columns, select a subset
                selected_cols = None
                if len(numeric_cols) > 10:
                    # Select columns with highest correlations or most important columns
                    if "issues" in self.dataset_metadata and "high_correlation" in self.dataset_metadata["issues"]:
                        # Extract columns involved in high correlations
                        corr_cols = set()
                        for corr_info in self.dataset_metadata["issues"]["high_correlation"]:
                            corr_cols.add(corr_info["col1"])
                            corr_cols.add(corr_info["col2"])
                        
                        # Take up to 8 columns
                        selected_cols = list(corr_cols)[:8]
                
                recommendations.append({
                    "type": "correlation_heatmap",
                    "config": {"method": "pearson", "columns": selected_cols},
                    "reason": "Correlation heatmap recommended to visualize relationships between numeric variables" + 
                              (f" (showing {len(selected_cols)} most correlated columns)" if selected_cols else "")
                })
            
            # 7. For datetime columns, recommend time series plots
            for dt_col in datetime_cols:
                if numeric_cols:
                    # Find numeric columns with interesting patterns over time
                    best_numeric_col = None
                    
                    # If we have analysis results, use them to find the best column
                    if "issues" in self.dataset_metadata and "high_correlation" in self.dataset_metadata["issues"]:
                        # Look for numeric columns that might be correlated with time
                        for corr_info in self.dataset_metadata["issues"]["high_correlation"]:
                            if corr_info["col1"] == dt_col and corr_info["col2"] in numeric_cols:
                                best_numeric_col = corr_info["col2"]
                                break
                            elif corr_info["col2"] == dt_col and corr_info["col1"] in numeric_cols:
                                best_numeric_col = corr_info["col1"]
                                break
                    
                    # If no correlation found, use the first numeric column
                    if not best_numeric_col and numeric_cols:
                        best_numeric_col = numeric_cols[0]
                    
                    if best_numeric_col:
                        recommendations.append({
                            "type": "line",
                            "config": {
                                "x_column": dt_col,
                                "y_column": best_numeric_col
                            },
                            "reason": f"Time series plot recommended to visualize {best_numeric_col} over time ({dt_col})"
                        })
            
            # 8. For datasets with categorical and numeric columns, recommend grouped bar charts
            if categorical_cols and numeric_cols:
                # Find a categorical column with reasonable cardinality
                for cat_col in categorical_cols:
                    if 2 <= df[cat_col].nunique() <= 6:  # Good number of categories for grouping
                        # Find a relevant numeric column
                        for num_col in numeric_cols[:2]:  # Consider first two numeric columns
                            recommendations.append({
                                "type": "bar",
                                "config": {
                                    "column": cat_col,
                                    "numeric_column": num_col,
                                    "top_n": df[cat_col].nunique(),
                                    "horizontal": df[cat_col].nunique() > 4
                                },
                                "reason": f"Grouped bar chart recommended to compare {num_col} across different {cat_col} categories"
                            })
                        break
            
            # 9. For categorical columns with few unique values, recommend pie charts
            for col in categorical_cols:
                if 2 <= df[col].nunique() <= 6:  # Ideal for pie charts
                    recommendations.append({
                        "type": "pie",
                        "config": {"column": col},
                        "reason": f"Pie chart recommended to visualize distribution of {col} categories"
                    })
            
            # 10. For datasets with multiple numeric columns, recommend pairplot for key columns
            if len(numeric_cols) >= 3:
                # Select a subset of columns for pairplot (too many would be unreadable)
                selected_cols = numeric_cols[:4]  # Take first 4 numeric columns
                
                # If we have a categorical column with few categories, use it for coloring
                hue = None
                if categorical_cols:
                    for cat_col in categorical_cols:
                        if df[cat_col].nunique() <= 5:
                            hue = cat_col
                            break
                
                recommendations.append({
                    "type": "pairplot",
                    "config": {
                        "columns": selected_cols,
                        "hue": hue
                    },
                    "reason": f"Pairplot recommended to visualize relationships between key numeric variables" + 
                              (f", colored by {hue}" if hue else "")
                })
            
            # Sort recommendations by relevance (custom logic could be added here)
            # For now, we'll prioritize correlation plots, then boxplots, then histograms
            def get_plot_priority(rec):
                plot_type = rec["type"]
                if plot_type == "scatter" or plot_type == "correlation_heatmap":
                    return 1
                elif plot_type == "boxplot":
                    return 2
                elif plot_type == "histogram":
                    return 3
                elif plot_type == "bar" or plot_type == "pie":
                    return 4
                else:
                    return 5
            
            recommendations.sort(key=get_plot_priority)
            
            # Limit to top 8 recommendations
            return recommendations[:8]
            
        except Exception as e:
            logger.error(f"Error generating visualization recommendations: {str(e)}")
            return self._generate_enhanced_mock_recommendations(df)
    
    def _generate_enhanced_mock_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """
        Generate enhanced rule-based visualization recommendations when OpenAI API is not available.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of recommended visualization configurations
        """
        recommendations = []
        
        # Get column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Calculate basic statistics for better recommendations
        column_stats = {}
        for col in numeric_cols:
            try:
                column_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "skew": float(df[col].skew()) if hasattr(df[col], 'skew') else 0,
                    "unique_ratio": float(df[col].nunique() / len(df)),
                    "missing_ratio": float(df[col].isna().mean())
                }
            except:
                # Handle any calculation errors
                column_stats[col] = {
                    "mean": 0, "std": 0, "skew": 0, "unique_ratio": 0, "missing_ratio": 0
                }
        
        # 1. For numeric columns with high skewness, recommend histograms with KDE
        for col in numeric_cols:
            if col in column_stats and abs(column_stats[col]["skew"]) > 1.0:
                recommendations.append({
                    "type": "histogram",
                    "config": {
                        "column": col,
                        "bins": 25,
                        "kde": True,
                        "log_scale": column_stats[col]["skew"] > 2.0
                    },
                    "reason": f"Histogram with KDE recommended for {col} which shows {'positive' if column_stats[col]['skew'] > 0 else 'negative'} skewness"
                })
        
        # 2. For numeric columns, recommend histograms
        added_histograms = set(rec["config"]["column"] for rec in recommendations if rec["type"] == "histogram")
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            if col not in added_histograms:
                recommendations.append({
                    "type": "histogram",
                    "config": {"column": col, "bins": 20, "kde": True},
                    "reason": f"Histogram recommended to visualize distribution of {col}"
                })
        
        # 3. For categorical columns, recommend bar charts
        for col in categorical_cols[:2]:  # Top 2 categorical columns
            if df[col].nunique() <= 15:  # Not too many categories
                recommendations.append({
                    "type": "bar",
                    "config": {
                        "column": col, 
                        "top_n": min(df[col].nunique(), 12),
                        "horizontal": df[col].nunique() > 8
                    },
                    "reason": f"Bar chart recommended to visualize distribution of {col} categories"
                })
        
        # 4. If multiple numeric columns, recommend scatter plot of first two
        if len(numeric_cols) >= 2:
            # Find the two numeric columns with highest variance
            sorted_cols = sorted(numeric_cols, key=lambda c: column_stats.get(c, {}).get("std", 0), reverse=True)
            col1, col2 = sorted_cols[0], sorted_cols[1]
            
            # If we have categorical columns, recommend colored scatter plots
            color_by = None
            if categorical_cols:
                for cat_col in categorical_cols:
                    if df[cat_col].nunique() <= 6:  # Not too many categories for coloring
                        color_by = cat_col
                        break
            
            recommendations.append({
                "type": "scatter",
                "config": {
                    "x_column": col1,
                    "y_column": col2,
                    "trendline": True,
                    "color_by": color_by
                },
                "reason": f"Scatter plot recommended to visualize relationship between {col1} and {col2}" + 
                          (f", colored by {color_by}" if color_by else "")
            })
        
        # 5. If multiple numeric columns, recommend correlation heatmap
        if len(numeric_cols) > 3:
            # If too many numeric columns, select a subset
            selected_cols = None
            if len(numeric_cols) > 10:
                selected_cols = numeric_cols[:8]  # Take first 8 columns
            
            recommendations.append({
                "type": "correlation_heatmap",
                "config": {"method": "pearson", "columns": selected_cols},
                "reason": "Correlation heatmap recommended to visualize relationships between numeric variables" + 
                          (f" (showing {len(selected_cols)} columns)" if selected_cols else "")
            })
        
        # 6. If categorical and numeric columns, recommend boxplot
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Only use group_by if the categorical column has reasonable cardinality
            group_by = cat_col if df[cat_col].nunique() <= 8 else None
            
            recommendations.append({
                "type": "boxplot",
                "config": {
                    "column": num_col,
                    "group_by": group_by
                },
                "reason": f"Boxplot recommended to visualize distribution of {num_col}" + 
                          (f" across different {cat_col} categories" if group_by else "")
            })
        
        # 7. For categorical columns with few unique values, recommend pie charts
        for col in categorical_cols:
            if 2 <= df[col].nunique() <= 6:  # Ideal for pie charts
                recommendations.append({
                    "type": "pie",
                    "config": {"column": col},
                    "reason": f"Pie chart recommended to visualize distribution of {col} categories"
                })
        
        # 8. For datasets with multiple numeric columns, recommend pairplot for key columns
        if len(numeric_cols) >= 3:
            # Select a subset of columns for pairplot
            selected_cols = numeric_cols[:3]  # Take first 3 numeric columns
            
            # If we have a categorical column with few categories, use it for coloring
            hue = None
            if categorical_cols:
                for cat_col in categorical_cols:
                    if df[cat_col].nunique() <= 5:
                        hue = cat_col
                        break
            
            recommendations.append({
                "type": "pairplot",
                "config": {
                    "columns": selected_cols,
                    "hue": hue
                },
                "reason": f"Pairplot recommended to visualize relationships between key numeric variables" + 
                          (f", colored by {hue}" if hue else "")
            })
        
        # Sort recommendations by relevance
        def get_plot_priority(rec):
            plot_type = rec["type"]
            if plot_type == "scatter" or plot_type == "correlation_heatmap":
                return 1
            elif plot_type == "boxplot":
                return 2
            elif plot_type == "histogram":
                return 3
            elif plot_type == "bar" or plot_type == "pie":
                return 4
            else:
                return 5
        
        recommendations.sort(key=get_plot_priority)
        
        # Return top 8 recommendations
        return recommendations[:8]


class EDAMiniAgent:
    """
    Mini AI agent specialized in exploratory data analysis.
    """
    
    def __init__(self, api_key=None):
        # Configure OpenAI API key if provided
        self.api_key = api_key
        if is_valid_api_key(api_key):
            self.use_mock = False
        else:
            self.use_mock = USE_MOCK
    
    def generate_insight(self, df: pd.DataFrame, plot_data: Dict, dataset_name: str) -> str:
        """
        Generate an insight for a specific visualization.
        
        Args:
            df: Input DataFrame
            plot_data: Plot data including statistics
            dataset_name: Name of the dataset
            
        Returns:
            String with insight
        """
        if "error" in plot_data:
            return f"Error: {plot_data['error']}"
        
        plot_type = plot_data.get("plot_type", "unknown")
        stats = plot_data.get("stats", {})
        
        # If using mock functionality, return a generic insight
        if self.use_mock:
            return self._generate_mock_insight(plot_type, stats)
        
        # Create a prompt based on the plot type and statistics
        prompt = f"Generate a concise, data-driven insight about the following {plot_type} visualization for the dataset '{dataset_name}'.\n\n"
        prompt += "Statistics:\n"
        
        for key, value in stats.items():
            if isinstance(value, dict):
                prompt += f"- {key}: {json.dumps(value)[:100]}...\n"
            else:
                prompt += f"- {key}: {value}\n"
        
        prompt += "\nProvide a concise, factual insight based ONLY on these statistics. Do not make assumptions beyond what the data shows."
        prompt += "\nFocus on patterns, outliers, distributions, or correlations as appropriate for this visualization type."
        prompt += "\nKeep the insight to 2-3 sentences maximum."
        
        # Get similar insights from ChromaDB for context if initialized
        if CHROMA_INITIALIZED:
            try:
                similar_insights = insights_collection.query(
                    query_texts=[prompt],
                    n_results=3
                )
                
                if similar_insights["ids"][0]:
                    prompt += "\n\nHere are some similar insights for context (but generate a new insight specific to this data):\n"
                    for insight in similar_insights["documents"][0]:
                        prompt += f"- {insight}\n"
            except Exception as e:
                logger.error(f"Error querying insights collection: {str(e)}")
        
        # Generate insight using OpenAI
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analysis assistant that provides concise, factual insights based only on the statistics provided. Do not make assumptions beyond what the data shows."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.5
            )
            
            insight = response.choices[0].message.content.strip()
            
            # Store in ChromaDB if initialized
            if CHROMA_INITIALIZED:
                try:
                    insight_id = f"{dataset_name}_{plot_type}_{int(time.time())}"
                    insights_collection.upsert(
                        ids=[insight_id],
                        documents=[insight],
                        metadatas=[{
                            "dataset": dataset_name,
                            "plot_type": plot_type,
                            "timestamp": time.time()
                        }]
                    )
                except Exception as e:
                    logger.error(f"Error storing insight in ChromaDB: {str(e)}")
            
            return insight
        
        except Exception as e:
            logger.error(f"Error generating insight with OpenAI: {str(e)}")
            return self._generate_mock_insight(plot_type, stats)
    
    def _generate_mock_insight(self, plot_type: str, stats: Dict) -> str:
        """
        Generate a mock insight when OpenAI API is not available.
        
        Args:
            plot_type: Type of plot
            stats: Statistics from the plot
            
        Returns:
            String with mock insight
        """
        if plot_type == "histogram":
            if "mean" in stats and "median" in stats:
                mean = stats.get("mean", 0)
                median = stats.get("median", 0)
                skew_direction = "right" if mean > median else "left" if mean < median else "not"
                return f"The distribution is {skew_direction} skewed with a mean of {mean:.2f} and median of {median:.2f}. The data shows a central tendency around these values."
            return "This histogram shows the distribution of values across different bins. The height of each bar represents the frequency or count of observations in that bin."
        
        elif plot_type == "boxplot":
            return "This boxplot shows the distribution of the data through quartiles. The box represents the interquartile range (IQR), with the median shown as a line inside the box. Whiskers extend to show the range of the data, and points beyond the whiskers may represent outliers."
        
        elif plot_type == "scatter":
            return "This scatter plot shows the relationship between two variables. Each point represents an observation with its position determined by the values of the two variables. Patterns in the scatter plot may indicate correlation or other relationships between the variables."
        
        elif plot_type == "correlation_heatmap":
            return "This correlation heatmap shows the strength of relationships between pairs of variables. Darker colors typically indicate stronger correlations, with positive correlations shown in one color and negative correlations in another."
        
        elif plot_type == "bar" or plot_type == "pie":
            return "This chart shows the distribution of a categorical variable. Each segment represents a category, with the size proportional to the frequency or percentage of that category in the dataset."
        
        else:
            return f"This {plot_type} visualization shows patterns and relationships in the data. Examine the axes and legend to understand what is being displayed."


class PreprocessingMiniAgent:
    """
    Mini AI agent specialized in preprocessing recommendations.
    """
    
    def __init__(self, api_key=None):
        # Configure OpenAI API key if provided
        self.api_key = api_key
        if is_valid_api_key(api_key):
            self.use_mock = False
        else:
            self.use_mock = USE_MOCK
    
    def suggest_preprocessing_steps(self, df: pd.DataFrame, issues: Dict) -> List[Dict]:
        """
        Suggest preprocessing steps based on dataset and detected issues.
        
        Args:
            df: Input DataFrame
            issues: Dictionary of detected issues
            
        Returns:
            List of suggested preprocessing steps
        """
        from ..utils.data_utils import suggest_preprocessing_steps
        
        # Use the utility function to get suggestions
        suggestions = suggest_preprocessing_steps(df, issues)
        
        # Store in ChromaDB if initialized
        if CHROMA_INITIALIZED and not self.use_mock:
            try:
                for i, suggestion in enumerate(suggestions):
                    step_id = f"suggestion_{int(time.time())}_{i}"
                    step_str = json.dumps(suggestion)
                    
                    preprocessing_collection.upsert(
                        ids=[step_id],
                        documents=[step_str],
                        metadatas=[{
                            "step_type": suggestion.get("step", "unknown"),
                            "timestamp": time.time()
                        }]
                    )
            except Exception as e:
                logger.error(f"Error storing preprocessing suggestion in ChromaDB: {str(e)}")
        
        return suggestions
    
    def evaluate_preprocessing_pipeline(self, steps: List[Dict]) -> Dict:
        """
        Evaluate a preprocessing pipeline and provide feedback.
        
        Args:
            steps: List of preprocessing steps
            
        Returns:
            Dictionary with evaluation results
        """
        if not steps:
            return {
                "score": 0,
                "feedback": "No preprocessing steps provided."
            }
        
        # If using mock functionality, return a generic evaluation
        if self.use_mock:
            return self._generate_mock_evaluation(steps)
        
        # Create a prompt for evaluation
        prompt = "Evaluate the following preprocessing pipeline and provide feedback:\n\n"
        
        for i, step in enumerate(steps):
            step_type = step.get("step", "unknown")
            reason = step.get("reason", "No reason provided")
            
            if step_type == "impute_missing":
                column = step.get("column", "unknown")
                method = step.get("method", "unknown")
                prompt += f"{i+1}. Impute missing values in column '{column}' using {method} method. Reason: {reason}\n"
            
            elif step_type == "drop_column":
                column = step.get("column", "unknown")
                prompt += f"{i+1}. Drop column '{column}'. Reason: {reason}\n"
            
            elif step_type == "handle_outliers":
                column = step.get("column", "unknown")
                method = step.get("method", "unknown")
                prompt += f"{i+1}. Handle outliers in column '{column}' using {method} method. Reason: {reason}\n"
            
            elif step_type == "transform":
                column = step.get("column", "unknown")
                method = step.get("method", "unknown")
                prompt += f"{i+1}. Transform column '{column}' using {method} transform. Reason: {reason}\n"
            
            elif step_type == "encode":
                column = step.get("column", "unknown")
                method = step.get("method", "unknown")
                prompt += f"{i+1}. Encode column '{column}' using {method} encoding. Reason: {reason}\n"
            
            elif step_type == "scale":
                columns = step.get("columns", [])
                method = step.get("method", "unknown")
                prompt += f"{i+1}. Scale columns {columns} using {method} scaling. Reason: {reason}\n"
            
            else:
                prompt += f"{i+1}. {step_type} step with parameters {step}. Reason: {reason}\n"
        
        prompt += "\nEvaluate this pipeline on a scale of 1-10 and provide specific feedback on its effectiveness, potential issues, and suggestions for improvement."
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data preprocessing expert that evaluates preprocessing pipelines and provides constructive feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.5
            )
            
            feedback = response.choices[0].message.content.strip()
            
            # Extract score from feedback (assuming it's mentioned in the format "score: X" or similar)
            score = 5  # Default score
            import re
            score_match = re.search(r'(\d+)(?:\s*\/\s*10|\s*out of\s*10)?', feedback)
            if score_match:
                try:
                    score = int(score_match.group(1))
                    if score < 1:
                        score = 1
                    elif score > 10:
                        score = 10
                except:
                    pass
            
            return {
                "score": score,
                "feedback": feedback
            }
        
        except Exception as e:
            logger.error(f"Error evaluating preprocessing pipeline with OpenAI: {str(e)}")
            return self._generate_mock_evaluation(steps)
    
    def _generate_mock_evaluation(self, steps: List[Dict]) -> Dict:
        """
        Generate a mock evaluation when OpenAI API is not available.
        
        Args:
            steps: List of preprocessing steps
            
        Returns:
            Dictionary with mock evaluation results
        """
        # Count steps by type
        step_types = {}
        for step in steps:
            step_type = step.get("step", "unknown")
            step_types[step_type] = step_types.get(step_type, 0) + 1
        
        # Generate feedback based on step types
        feedback = "Preprocessing Pipeline Evaluation:\n\n"
        
        if "impute_missing" in step_types:
            feedback += "✓ Good: The pipeline handles missing values.\n"
        else:
            feedback += "⚠️ Consider: Check if the dataset has missing values that need to be addressed.\n"
        
        if "handle_outliers" in step_types:
            feedback += "✓ Good: The pipeline addresses outliers.\n"
        
        if "encode" in step_types:
            feedback += "✓ Good: Categorical variables are being encoded.\n"
        
        if "scale" in step_types:
            feedback += "✓ Good: Features are being scaled, which is important for many machine learning algorithms.\n"
        
        if "drop_column" in step_types and step_types["drop_column"] > 2:
            feedback += "⚠️ Caution: Multiple columns are being dropped. Ensure this doesn't remove important information.\n"
        
        if len(steps) > 10:
            feedback += "⚠️ Caution: The pipeline is quite complex with many steps. Consider simplifying if possible.\n"
        elif len(steps) < 3:
            feedback += "⚠️ Consider: The pipeline is minimal. Additional preprocessing steps might be beneficial.\n"
        
        # Calculate a mock score based on pipeline composition
        score = 5  # Base score
        
        # Adjust score based on pipeline composition
        if "impute_missing" in step_types:
            score += 1
        if "handle_outliers" in step_types:
            score += 1
        if "encode" in step_types:
            score += 1
        if "scale" in step_types:
            score += 1
        if len(steps) > 10:
            score -= 1
        
        # Ensure score is within 1-10 range
        score = max(1, min(10, score))
        
        feedback += f"\nOverall Score: {score}/10"
        
        return {
            "score": score,
            "feedback": feedback
        }


class ErrorDetectionAgent:
    """
    Mini AI agent specialized in detecting and fixing errors in the dataset.
    """
    
    def __init__(self, api_key=None):
        # Configure OpenAI API key if provided
        self.api_key = api_key
        if is_valid_api_key(api_key):
            self.use_mock = False
        else:
            self.use_mock = USE_MOCK
    
    def detect_errors(self, df: pd.DataFrame) -> Dict:
        """
        Detect errors in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with detected errors
        """
        from ..utils.data_utils import detect_data_issues
        
        # Use the utility function to detect issues
        issues = detect_data_issues(df)
        
        # Convert issues to errors format
        errors = {
            "missing_values": [],
            "outliers": [],
            "inconsistent_types": [],
            "high_cardinality": [],
            "zero_variance": [],
            "high_correlation": []
        }
        
        # Missing values
        for col, info in issues["missing_values"].items():
            if info["percent"] > 0:
                errors["missing_values"].append({
                    "column": col,
                    "count": info["count"],
                    "percent": info["percent"],
                    "suggestion": self._suggest_missing_value_fix(df, col, info["percent"])
                })
        
        # Outliers
        for col, info in issues["outliers"].items():
            errors["outliers"].append({
                "column": col,
                "count": info["count"],
                "percent": info["percent"],
                "lower_bound": info["lower_bound"],
                "upper_bound": info["upper_bound"],
                "suggestion": self._suggest_outlier_fix(df, col, info["percent"])
            })
        
        # Inconsistent types
        for col, info in issues["inconsistent_types"].items():
            errors["inconsistent_types"].append({
                "column": col,
                "percent_numeric": info["percent_numeric"],
                "suggestion": self._suggest_type_fix(df, col, info["percent_numeric"])
            })
        
        # High cardinality
        for col, info in issues["high_cardinality"].items():
            errors["high_cardinality"].append({
                "column": col,
                "unique_count": info["unique_count"],
                "unique_ratio": info["unique_ratio"],
                "suggestion": f"Consider grouping rare categories or encoding this column."
            })
        
        # Zero variance
        for col in issues["zero_variance"]:
            errors["zero_variance"].append({
                "column": col,
                "suggestion": f"Consider dropping this column as it has constant value."
            })
        
        # High correlation
        for corr_info in issues["high_correlation"]:
            errors["high_correlation"].append({
                "columns": [corr_info["col1"], corr_info["col2"]],
                "correlation": corr_info["correlation"],
                "suggestion": f"Consider dropping one of these columns or creating a composite feature."
            })
        
        return errors
    
    def _suggest_missing_value_fix(self, df: pd.DataFrame, column: str, percent: float) -> str:
        """Suggest a fix for missing values."""
        if percent < 5:
            if df[column].dtype.kind in 'ifc':  # numeric
                return f"Impute with median or mean as the missing percentage is low ({percent:.1f}%)."
            else:
                return f"Impute with mode or create a 'Missing' category as the missing percentage is low ({percent:.1f}%)."
        elif percent < 30:
            return f"Consider advanced imputation methods like KNN or model-based imputation ({percent:.1f}% missing)."
        else:
            return f"Consider dropping this column as it has too many missing values ({percent:.1f}%)."
    
    def _suggest_outlier_fix(self, df: pd.DataFrame, column: str, percent: float) -> str:
        """Suggest a fix for outliers."""
        if percent < 1:
            return f"Remove outliers as they are very few ({percent:.1f}%)."
        elif percent < 5:
            return f"Winsorize (clip) outliers to the boundaries ({percent:.1f}% outliers)."
        else:
            return f"Apply transformation (log, sqrt) as there are many outliers ({percent:.1f}%), suggesting a skewed distribution."
    
    def _suggest_type_fix(self, df: pd.DataFrame, column: str, percent_numeric: float) -> str:
        """Suggest a fix for inconsistent types."""
        if percent_numeric > 80:
            return f"Convert to numeric type as most values ({percent_numeric:.1f}%) are numeric."
        else:
            return f"Convert to categorical type as many values ({100-percent_numeric:.1f}%) are non-numeric."
    
    def fix_error(self, df: pd.DataFrame, error_type: str, error_info: Dict) -> Tuple[pd.DataFrame, str]:
        """
        Fix an error in the dataset.
        
        Args:
            df: Input DataFrame
            error_type: Type of error to fix
            error_info: Information about the error
            
        Returns:
            Tuple of (fixed_df, message)
        """
        from ..utils.data_utils import apply_preprocessing_step
        
        if error_type == "missing_values":
            column = error_info["column"]
            
            # Determine method based on data type and missing percentage
            if df[column].dtype.kind in 'ifc':  # numeric
                if error_info["percent"] < 5:
                    method = "median"
                else:
                    method = "knn"
            else:  # categorical
                if error_info["percent"] < 5:
                    method = "mode"
                else:
                    method = "new_category"
            
            step = {
                "step": "impute_missing",
                "column": column,
                "method": method,
                "reason": f"Fix {error_info['percent']:.1f}% missing values in {column}"
            }
            
            return apply_preprocessing_step(df, step)
        
        elif error_type == "outliers":
            column = error_info["column"]
            
            # Determine method based on outlier percentage
            if error_info["percent"] < 1:
                method = "remove"
            elif error_info["percent"] < 5:
                method = "winsorize"
            else:
                method = "transform"
                
            step = {
                "step": "handle_outliers",
                "column": column,
                "method": method,
                "reason": f"Fix {error_info['percent']:.1f}% outliers in {column}"
            }
            
            return apply_preprocessing_step(df, step)
        
        elif error_type == "inconsistent_types":
            column = error_info["column"]
            
            # Determine method based on percentage of numeric values
            if error_info["percent_numeric"] > 80:
                method = "to_numeric"
            else:
                method = "to_categorical"
                
            step = {
                "step": "convert_type",
                "column": column,
                "method": method,
                "reason": f"Fix inconsistent types in {column}"
            }
            
            return apply_preprocessing_step(df, step)
        
        elif error_type == "high_cardinality":
            column = error_info["column"]
            
            step = {
                "step": "reduce_cardinality",
                "column": column,
                "method": "group_rare",
                "threshold": 0.01,
                "reason": f"Reduce high cardinality in {column}"
            }
            
            return apply_preprocessing_step(df, step)
        
        elif error_type == "zero_variance":
            column = error_info["column"]
            
            step = {
                "step": "drop_column",
                "column": column,
                "reason": f"Drop zero variance column {column}"
            }
            
            return apply_preprocessing_step(df, step)
        
        elif error_type == "high_correlation":
            columns = error_info["columns"]
            
            step = {
                "step": "handle_correlation",
                "columns": columns,
                "method": "drop_one",
                "reason": f"Handle high correlation between {columns[0]} and {columns[1]}"
            }
            
            return apply_preprocessing_step(df, step)
        
        else:
            return df, f"Unknown error type: {error_type}" 