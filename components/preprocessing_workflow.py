import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from ..utils.data_utils import apply_preprocessing_step

# Preprocessing step definitions
PREPROCESSING_STEPS = {
    "impute_missing": {
        "name": "Impute Missing Values",
        "description": "Fill missing values in a column",
        "icon": "🔄",
        "methods": {
            "mean": "Mean (numeric)",
            "median": "Median (numeric)",
            "mode": "Mode (categorical)",
            "constant": "Constant value",
            "knn": "KNN imputation (numeric)",
            "new_category": "New category (categorical)",
            "forward_fill": "Forward fill (time series)",
            "backward_fill": "Backward fill (time series)",
            "interpolate": "Interpolation (numeric)"
        },
        "requires_column": True,
        "category": "Data Cleaning"
    },
    "drop_column": {
        "name": "Drop Column",
        "description": "Remove a column from the dataset",
        "icon": "🗑️",
        "methods": {},
        "requires_column": True,
        "category": "Feature Selection"
    },
    "remove_duplicates": {
        "name": "Remove Duplicates",
        "description": "Remove duplicate rows from the dataset",
        "icon": "🔍",
        "methods": {
            "all_columns": "Consider all columns",
            "subset": "Consider subset of columns",
            "keep_first": "Keep first occurrence",
            "keep_last": "Keep last occurrence"
        },
        "requires_column": False,
        "category": "Data Cleaning"
    },
    "handle_outliers": {
        "name": "Handle Outliers",
        "description": "Detect and handle outliers in a column",
        "icon": "📊",
        "methods": {
            "remove": "Remove outliers",
            "winsorize": "Winsorize (clip to boundaries)",
            "transform": "Apply transformation",
            "zscore": "Z-score method",
            "iqr": "IQR method"
        },
        "requires_column": True,
        "category": "Data Cleaning"
    },
    "transform": {
        "name": "Transform Column",
        "description": "Apply mathematical transformation to a column",
        "icon": "📈",
        "methods": {
            "log": "Log transform",
            "sqrt": "Square root transform",
            "box-cox": "Box-Cox transform",
            "yeo-johnson": "Yeo-Johnson transform",
            "reciprocal": "Reciprocal (1/x)",
            "square": "Square (x²)",
            "cube": "Cube (x³)"
        },
        "requires_column": True,
        "category": "Data Transformation"
    },
    "convert_type": {
        "name": "Convert Type",
        "description": "Convert column to a different data type",
        "icon": "🔄",
        "methods": {
            "to_numeric": "To numeric",
            "to_categorical": "To categorical",
            "to_datetime": "To datetime",
            "to_boolean": "To boolean"
        },
        "requires_column": True,
        "category": "Data Cleaning"
    },
    "reduce_cardinality": {
        "name": "Reduce Cardinality",
        "description": "Reduce the number of unique values in a categorical column",
        "icon": "📉",
        "methods": {
            "group_rare": "Group rare categories",
            "kmeans_clustering": "K-means clustering",
            "hierarchical_clustering": "Hierarchical clustering"
        },
        "requires_column": True,
        "category": "Feature Engineering"
    },
    "encode": {
        "name": "Encode Categorical",
        "description": "Encode categorical variables for modeling",
        "icon": "🔠",
        "methods": {
            "label": "Label encoding",
            "onehot": "One-hot encoding",
            "target": "Target encoding",
            "frequency": "Frequency encoding",
            "binary": "Binary encoding",
            "ordinal": "Ordinal encoding",
            "helmert": "Helmert encoding",
            "count": "Count encoding"
        },
        "requires_column": True,
        "category": "Feature Engineering"
    },
    "scale": {
        "name": "Scale Features",
        "description": "Scale numeric features",
        "icon": "⚖️",
        "methods": {
            "standard": "Standardization (z-score)",
            "minmax": "Min-Max scaling",
            "robust": "Robust scaling",
            "maxabs": "Max-Abs scaling",
            "quantile": "Quantile transformer",
            "power": "Power transformer"
        },
        "requires_column": False,
        "category": "Feature Scaling"
    },
    "handle_correlation": {
        "name": "Handle Correlation",
        "description": "Handle highly correlated features",
        "icon": "🔗",
        "methods": {
            "drop_one": "Drop one feature",
            "pca": "Apply PCA",
            "vif": "Variance Inflation Factor"
        },
        "requires_column": False,
        "category": "Feature Selection"
    },
    "binning": {
        "name": "Binning",
        "description": "Convert continuous variables into discrete bins",
        "icon": "📊",
        "methods": {
            "equal_width": "Equal-width binning",
            "equal_frequency": "Equal-frequency binning",
            "kmeans": "K-means binning",
            "custom": "Custom bins"
        },
        "requires_column": True,
        "category": "Data Transformation"
    },
    "feature_interaction": {
        "name": "Feature Interaction",
        "description": "Create interaction features between columns",
        "icon": "🔀",
        "methods": {
            "multiplication": "Multiplication",
            "addition": "Addition",
            "subtraction": "Subtraction",
            "division": "Division",
            "polynomial": "Polynomial features"
        },
        "requires_column": False,
        "category": "Feature Engineering"
    },
    "dimensionality_reduction": {
        "name": "Dimensionality Reduction",
        "description": "Reduce the number of features while preserving information",
        "icon": "📉",
        "methods": {
            "pca": "Principal Component Analysis",
            "lda": "Linear Discriminant Analysis",
            "tsne": "t-SNE",
            "umap": "UMAP"
        },
        "requires_column": False,
        "category": "Dimensionality Reduction"
    },
    "feature_selection": {
        "name": "Feature Selection",
        "description": "Select the most important features",
        "icon": "🎯",
        "methods": {
            "correlation": "Correlation-based",
            "chi2": "Chi-square test",
            "rfe": "Recursive Feature Elimination",
            "lasso": "LASSO",
            "random_forest": "Random Forest importance"
        },
        "requires_column": False,
        "category": "Feature Selection"
    },
    "handle_imbalance": {
        "name": "Handle Imbalanced Data",
        "description": "Address class imbalance in the target variable",
        "icon": "⚖️",
        "methods": {
            "smote": "SMOTE oversampling",
            "adasyn": "ADASYN oversampling",
            "random_undersampling": "Random undersampling",
            "class_weights": "Class weighting"
        },
        "requires_column": True,
        "category": "Imbalanced Data"
    },
    "time_features": {
        "name": "Extract Time Features",
        "description": "Extract features from datetime columns",
        "icon": "🕒",
        "methods": {
            "date_parts": "Extract date parts (year, month, day)",
            "time_parts": "Extract time parts (hour, minute, second)",
            "cyclical": "Cyclical encoding (sin/cos)",
            "holidays": "Holiday indicators",
            "lag_features": "Lag features"
        },
        "requires_column": True,
        "category": "Feature Engineering"
    },
    "text_processing": {
        "name": "Text Processing",
        "description": "Process text data for analysis",
        "icon": "📝",
        "methods": {
            "tokenize": "Tokenization",
            "stopwords": "Remove stopwords",
            "stemming": "Stemming",
            "lemmatization": "Lemmatization",
            "tfidf": "TF-IDF vectorization",
            "word_embeddings": "Word embeddings"
        },
        "requires_column": True,
        "category": "Feature Engineering"
    }
}

# Group preprocessing steps by category
PREPROCESSING_CATEGORIES = {
    "Data Cleaning": ["impute_missing", "remove_duplicates", "handle_outliers", "convert_type"],
    "Feature Scaling": ["scale"],
    "Feature Engineering": ["reduce_cardinality", "encode", "feature_interaction", "time_features", "text_processing"],
    "Feature Selection": ["drop_column", "handle_correlation", "feature_selection"],
    "Data Transformation": ["transform", "binning"],
    "Dimensionality Reduction": ["dimensionality_reduction"],
    "Imbalanced Data": ["handle_imbalance"]
}

def get_preprocessing_steps_by_category():
    """Group preprocessing steps by category."""
    steps_by_category = {}
    for category, step_ids in PREPROCESSING_CATEGORIES.items():
        steps_by_category[category] = [
            {"id": step_id, **PREPROCESSING_STEPS[step_id]}
            for step_id in step_ids
        ]
    return steps_by_category

def get_step_config_ui(step_id: str, df: pd.DataFrame, step_config: Dict = None) -> Dict:
    """
    Generate UI for configuring a preprocessing step.
    
    Args:
        step_id: ID of the preprocessing step
        df: DataFrame to preprocess
        step_config: Existing configuration (if any)
        
    Returns:
        Dictionary with step configuration
    """
    if step_id not in PREPROCESSING_STEPS:
        st.error(f"Unknown preprocessing step: {step_id}")
        return {}
    
    step_info = PREPROCESSING_STEPS[step_id]
    config = step_config.copy() if step_config else {"step": step_id}
    
    # Step title
    st.markdown(f"### {step_info['icon']} {step_info['name']}")
    st.markdown(f"*{step_info['description']}*")
    
    # Column selection if required
    if step_info["requires_column"]:
        if step_id == "handle_correlation":
            # For correlation, select two columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for correlation handling")
                return {}
            
            # Allow selecting multiple column pairs for correlation handling
            st.markdown("#### Select columns with high correlation")
            
            # Option to select multiple pairs or use AI-detected correlations
            correlation_option = st.radio(
                "Correlation selection method:",
                ["Select column pairs manually", "Select multiple columns and find correlations"],
                key=f"corr_option_{step_id}"
            )
            
            if correlation_option == "Select column pairs manually":
                # Manual selection of column pairs
                col1 = st.selectbox("First column", numeric_cols, key=f"col1_{step_id}")
                col2 = st.selectbox("Second column", numeric_cols, 
                                   index=min(1, len(numeric_cols)-1), key=f"col2_{step_id}")
                
                config["columns"] = [col1, col2]
            else:
                # Select multiple columns and find correlations among them
                selected_cols = st.multiselect(
                    "Select columns to analyze for correlations", 
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))],
                    key=f"multi_cols_{step_id}"
                )
                
                if len(selected_cols) >= 2:
                    # Calculate correlation matrix
                    corr_matrix = df[selected_cols].corr().abs()
                    
                    # Get pairs with correlation above threshold
                    threshold = st.slider(
                        "Correlation threshold", 
                        min_value=0.5, 
                        max_value=1.0, 
                        value=0.8, 
                        step=0.05,
                        key=f"corr_threshold_{step_id}"
                    )
                    
                    # Find highly correlated pairs
                    correlated_pairs = []
                    for i in range(len(selected_cols)):
                        for j in range(i+1, len(selected_cols)):
                            if corr_matrix.iloc[i, j] >= threshold:
                                correlated_pairs.append({
                                    "col1": selected_cols[i],
                                    "col2": selected_cols[j],
                                    "correlation": corr_matrix.iloc[i, j]
                                })
                    
                    if correlated_pairs:
                        st.markdown("#### Highly correlated column pairs:")
                        for pair in correlated_pairs:
                            st.markdown(f"- **{pair['col1']}** and **{pair['col2']}**: {pair['correlation']:.2f}")
                        
                        # Select which pair to handle
                        pair_options = [f"{p['col1']} & {p['col2']} ({p['correlation']:.2f})" for p in correlated_pairs]
                        selected_pair = st.selectbox(
                            "Select pair to handle", 
                            pair_options,
                            key=f"pair_select_{step_id}"
                        )
                        
                        # Get the selected pair
                        pair_idx = pair_options.index(selected_pair)
                        selected_pair_data = correlated_pairs[pair_idx]
                        
                        config["columns"] = [selected_pair_data["col1"], selected_pair_data["col2"]]
                    else:
                        st.info(f"No column pairs with correlation >= {threshold} found.")
                        return {}
                else:
                    st.warning("Please select at least 2 columns to analyze correlations.")
                    return {}
        
        elif step_id in ["scale", "handle_outliers", "transform", "impute_missing"]:
            # For these steps, allow multi-column selection
            if step_id == "scale":
                # For scaling, select multiple columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                col_type = "numeric"
                title = "Select columns to scale"
            elif step_id == "handle_outliers":
                # For outlier handling, select multiple numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                col_type = "numeric"
                title = "Select columns to handle outliers"
            elif step_id == "transform":
                # For transformation, select multiple numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                col_type = "numeric"
                title = "Select columns to transform"
            elif step_id == "impute_missing":
                # For imputation, allow selecting multiple columns
                # First check if we want to impute numeric or categorical columns
                impute_type = st.radio(
                    "Column type to impute:",
                    ["Numeric", "Categorical"],
                    key=f"impute_type_{step_id}"
                )
                
                if impute_type == "Numeric":
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    col_type = "numeric"
                else:
                    numeric_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    col_type = "categorical"
                
                title = f"Select {col_type} columns with missing values to impute"
            
            if not numeric_cols:
                st.warning(f"No {col_type} columns available for this operation")
                return {}
            
            # Option to select single or multiple columns
            selection_mode = st.radio(
                "Column selection mode:",
                ["Single column", "Multiple columns"],
                key=f"selection_mode_{step_id}"
            )
            
            if selection_mode == "Single column":
                # Single column selection
                column = st.selectbox("Select column", numeric_cols, key=f"col_{step_id}")
                config["column"] = column
            else:
                # Multiple column selection
                selected_cols = st.multiselect(
                    title, 
                    numeric_cols, 
                    default=numeric_cols[:min(3, len(numeric_cols))], 
                    key=f"cols_{step_id}"
                )
                
                if not selected_cols:
                    st.warning("Please select at least one column")
                    return {}
                
                config["columns"] = selected_cols
                
                # Option to apply the same method to all columns or configure individually
                if len(selected_cols) > 1:
                    config["apply_same_method"] = st.checkbox(
                        "Apply the same method to all selected columns", 
                        value=True,
                        key=f"same_method_{step_id}"
                    )
        
        else:
            # For other steps, select a single column
            if step_id in ["encode", "reduce_cardinality"]:
                # Categorical columns for these operations
                cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                col_type = "categorical"
            elif step_id == "drop_column":
                # Any column type for drop operation
                cols = df.columns.tolist()
                col_type = "any"
                
                # For drop_column, allow selecting multiple columns
                selection_mode = st.radio(
                    "Column selection mode:",
                    ["Single column", "Multiple columns"],
                    key=f"selection_mode_{step_id}"
                )
                
                if selection_mode == "Multiple columns":
                    selected_cols = st.multiselect(
                        "Select columns to drop", 
                        cols, 
                        key=f"cols_{step_id}"
                    )
                    
                    if not selected_cols:
                        st.warning("Please select at least one column to drop")
                        return {}
                    
                    config["columns"] = selected_cols
                    
                    # Skip the rest of the single column selection
                    if "method" in step_info["params"]:
                        method = st.selectbox(
                            "Method", 
                            list(step_info["params"]["method"].keys()), 
                            key=f"method_{step_id}"
                        )
                        config["method"] = method
                    
                    # Add reason for preprocessing step
                    reason = st.text_area(
                        "Reason for this step (optional)", 
                        value=config.get("reason", ""), 
                        key=f"reason_{step_id}"
                    )
                    if reason:
                        config["reason"] = reason
                    
                    return config
            else:
                # Any column type for other operations
                cols = df.columns.tolist()
                col_type = "any"
            
            if not cols:
                st.warning(f"No {col_type} columns available for this operation")
                return {}
            
            column = st.selectbox("Select column", cols, key=f"col_{step_id}")
            config["column"] = column
    
    # Method selection if available
    if step_info["methods"]:
        methods = list(step_info["methods"].keys())
        method_labels = list(step_info["methods"].values())
        
        # Default method based on column type if not already set
        default_idx = 0
        if "method" in config:
            default_idx = methods.index(config["method"]) if config["method"] in methods else 0
        
        method = st.selectbox("Method", methods, 
                             format_func=lambda x: step_info["methods"][x],
                             index=default_idx,
                             key=f"method_{step_id}")
        config["method"] = method
        
        # Additional parameters based on method
        if step_id == "impute_missing" and method == "constant":
            value = st.text_input("Constant value", "0", key=f"value_{step_id}")
            try:
                # Try to convert to appropriate type
                if "column" in config and config["column"] in df.columns:
                    col_type = df[config["column"]].dtype
                    if col_type.kind in 'ifc':  # numeric
                        value = float(value)
                config["value"] = value
            except:
                st.warning("Please enter a valid value")
        
        elif step_id == "reduce_cardinality" and method == "group_rare":
            threshold = st.slider("Threshold (%)", 0.1, 10.0, 1.0, 0.1, key=f"threshold_{step_id}")
            config["threshold"] = threshold / 100  # Convert to proportion
    
    # Reason for step
    reason = st.text_area("Reason for this step", 
                         value=config.get("reason", ""), 
                         key=f"reason_{step_id}")
    config["reason"] = reason
    
    return config

def render_workflow(workflow: List[Dict], df: pd.DataFrame):
    """
    Render the preprocessing workflow as a flowchart.
    
    Args:
        workflow: List of preprocessing steps
        df: DataFrame to preprocess
    """
    if not workflow:
        st.info("No preprocessing steps added yet. Drag steps from the left panel to build your workflow.")
        return
    
    # Display workflow as a flowchart
    st.markdown("### Preprocessing Workflow")
    
    # Add CSS for the workflow visualization
    st.markdown("""
    <style>
    .workflow-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .workflow-step {
        width: 100%;
        max-width: 600px;
        margin-bottom: 15px;
        position: relative;
    }
    
    .workflow-step-content {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
    }
    
    .workflow-step-content:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .workflow-step-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        border-bottom: 1px solid #eee;
        padding-bottom: 8px;
    }
    
    .workflow-step-title {
        font-weight: 600;
        font-size: 16px;
        color: #333;
        display: flex;
        align-items: center;
    }
    
    .workflow-step-icon {
        margin-right: 8px;
        font-size: 20px;
    }
    
    .workflow-step-details {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 10px;
    }
    
    .workflow-step-detail {
        background-color: #f0f0f0;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
        color: #555;
    }
    
    .workflow-step-reason {
        font-style: italic;
        color: #666;
        font-size: 13px;
        margin-top: 5px;
        padding-top: 5px;
        border-top: 1px dashed #eee;
    }
    
    .workflow-connector {
        height: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
    }
    
    .workflow-connector::before {
        content: "";
        position: absolute;
        top: 0;
        bottom: 0;
        width: 2px;
        background-color: #4CAF50;
        z-index: 1;
    }
    
    .workflow-connector-dot {
        width: 10px;
        height: 10px;
        background-color: #4CAF50;
        border-radius: 50%;
        z-index: 2;
    }
    
    .workflow-step-buttons {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        margin-top: 10px;
    }
    
    .workflow-step-button {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .workflow-step-button:hover {
        background-color: #e0e0e0;
    }
    
    .workflow-step-edit {
        color: #2196F3;
    }
    
    .workflow-step-remove {
        color: #F44336;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a visual representation of the workflow
    st.markdown("<div class='workflow-container'>", unsafe_allow_html=True)
    
    for i, step in enumerate(workflow):
        step_id = step["step"]
        step_info = PREPROCESSING_STEPS.get(step_id, {})
        
        # Step container
        st.markdown(f"<div class='workflow-step' id='step_{i}'>", unsafe_allow_html=True)
        
        # Step content
        st.markdown(f"""
        <div class='workflow-step-content'>
            <div class='workflow-step-header'>
                <div class='workflow-step-title'>
                    <span class='workflow-step-icon'>{step_info.get('icon', '🔧')}</span>
                    {step_info.get('name', step_id)}
                </div>
                <div class='workflow-step-index'>Step {i+1}</div>
            </div>
            <div class='workflow-step-details'>
        """, unsafe_allow_html=True)
        
        # Display step details based on step type
        if "column" in step:
            st.markdown(f"<div class='workflow-step-detail'>Column: {step['column']}</div>", unsafe_allow_html=True)
        
        if "columns" in step and isinstance(step["columns"], list):
            columns_str = ", ".join(step["columns"])
            st.markdown(f"<div class='workflow-step-detail'>Columns: {columns_str}</div>", unsafe_allow_html=True)
        
        if "method" in step:
            method_name = step_info.get("methods", {}).get(step["method"], step["method"])
            st.markdown(f"<div class='workflow-step-detail'>Method: {method_name}</div>", unsafe_allow_html=True)
        
        # Close details div
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display reason if available
        if "reason" in step and step["reason"]:
            st.markdown(f"<div class='workflow-step-reason'>\"{step['reason']}\"</div>", unsafe_allow_html=True)
        
        # Buttons for each step
        st.markdown("<div class='workflow-step-buttons'>", unsafe_allow_html=True)
        
        # Edit button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✏️ Edit", key=f"edit_{i}"):
                st.session_state.editing_step = i
                return None
        
        # Remove button
        with col2:
            if st.button("🗑️ Remove", key=f"remove_{i}"):
                return i  # Return index to remove
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Close step content div
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add connector between steps (except for the last step)
        if i < len(workflow) - 1:
            st.markdown("""
            <div class='workflow-connector'>
                <div class='workflow-connector-dot'></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Close step container div
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return None  # No step to remove

def apply_workflow(df: pd.DataFrame, workflow: List[Dict]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply a preprocessing workflow to a DataFrame.
    
    Args:
        df: DataFrame to preprocess
        workflow: List of preprocessing steps
        
    Returns:
        Tuple of (processed DataFrame, list of messages)
    """
    messages = []
    processed_df = df.copy()
    
    for step in workflow:
        step_id = step.get("step")
        
        if step_id not in PREPROCESSING_STEPS:
            messages.append(f"Unknown preprocessing step: {step_id}")
            continue
        
        try:
            # Handle multi-column configurations
            if "columns" in step and isinstance(step["columns"], list) and len(step["columns"]) > 0:
                # Check if we should apply the same method to all columns
                if step.get("apply_same_method", False):
                    # Apply the same method to all columns
                    method = step.get("method")
                    for column in step["columns"]:
                        # Create a single-column configuration
                        single_col_config = step.copy()
                        single_col_config["column"] = column
                        if "columns" in single_col_config:
                            del single_col_config["columns"]
                        if "apply_same_method" in single_col_config:
                            del single_col_config["apply_same_method"]
                        
                        # Apply the step to this column
                        processed_df, step_messages = apply_preprocessing_step(processed_df, single_col_config)
                        messages.extend(step_messages)
                else:
                    # Apply the step as is (for steps that natively support multiple columns)
                    processed_df, step_messages = apply_preprocessing_step(processed_df, step)
                    messages.extend(step_messages)
            else:
                # Apply single-column step
                processed_df, step_messages = apply_preprocessing_step(processed_df, step)
                messages.extend(step_messages)
        
        except Exception as e:
            messages.append(f"Error applying {step_id}: {str(e)}")
    
    return processed_df, messages 