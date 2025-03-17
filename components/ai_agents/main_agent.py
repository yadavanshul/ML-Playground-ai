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


