import pandas as pd
import numpy as np
import io
import os
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns
from sklearn import datasets

def load_dataset(file_path: Optional[str] = None, file_buffer: Optional[io.BytesIO] = None, 
                 dataset_name: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Load a dataset from a file path, buffer, or predefined dataset.
    
    Args:
        file_path: Path to the dataset file
        file_buffer: File buffer containing the dataset
        dataset_name: Name of predefined dataset
        
    Returns:
        Tuple of (DataFrame, dataset_name)
    """
    if file_path:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        dataset_name = os.path.basename(file_path)
        
    elif file_buffer:
        df = pd.read_csv(file_buffer)
        dataset_name = "uploaded_dataset"
        
    elif dataset_name:
        if dataset_name == "iris":
            data = datasets.load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif dataset_name == "boston":
            data = datasets.fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif dataset_name == "diabetes":
            data = datasets.load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif dataset_name == "wine":
            data = datasets.load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif dataset_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif dataset_name == "titanic":
            df = sns.load_dataset("titanic")
        elif dataset_name == "tips":
            df = sns.load_dataset("tips")
        elif dataset_name == "planets":
            df = sns.load_dataset("planets")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    else:
        raise ValueError("Must provide either file_path, file_buffer, or dataset_name")
    
    # Limit to 1000 rows as specified in requirements
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)
        
    return df, dataset_name

def get_dataset_metadata(df: pd.DataFrame) -> Dict:
    """
    Extract metadata from a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing metadata
    """
    metadata = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": list(df.select_dtypes(include=['number']).columns),
        "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
        "datetime_columns": list(df.select_dtypes(include=['datetime']).columns),
        "summary_stats": df.describe().to_dict(),
    }
    
    # Add column-specific metadata
    metadata["column_metadata"] = {}
    for col in df.columns:
        col_meta = {
            "dtype": str(df[col].dtype),
            "missing_count": df[col].isnull().sum(),
            "missing_percentage": (df[col].isnull().sum() / len(df)) * 100,
        }
        
        if df[col].dtype.kind in 'ifc':  # numeric
            col_meta.update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                "skew": float(df[col].skew()) if not pd.isna(df[col].skew()) else None,
                "kurtosis": float(df[col].kurtosis()) if not pd.isna(df[col].kurtosis()) else None,
                "is_integer": all(df[col].dropna().apply(lambda x: float(x).is_integer())),
                "zeros_count": (df[col] == 0).sum(),
                "negative_count": (df[col] < 0).sum(),
            })
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':  # categorical
            value_counts = df[col].value_counts()
            col_meta.update({
                "unique_count": df[col].nunique(),
                "top_values": value_counts.head(5).to_dict(),
                "is_binary": df[col].nunique() == 2,
            })
        
        metadata["column_metadata"][col] = col_meta
    
    return metadata

def detect_data_issues(df: pd.DataFrame) -> Dict:
    """
    Detect common issues in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of detected issues
    """
    issues = {
        "missing_values": {},
        "outliers": {},
        "inconsistent_types": {},
        "high_cardinality": {},
        "imbalanced_categories": {},
        "zero_variance": [],
        "high_correlation": [],
        "potential_id_columns": [],
    }
    
    # Missing values
    missing_vals = df.isnull().sum()
    missing_percent = (missing_vals / len(df)) * 100
    issues["missing_values"] = {col: {"count": int(count), "percent": float(percent)} 
                               for col, (count, percent) in 
                               zip(missing_vals.index, zip(missing_vals, missing_percent)) 
                               if count > 0}
    
    # Outliers (using IQR method for numeric columns)
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            issues["outliers"][col] = {
                "count": len(outliers),
                "percent": (len(outliers) / len(df)) * 100,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
    
    # Inconsistent types (e.g., mixed numeric and string in object columns)
    for col in df.select_dtypes(include=['object']).columns:
        # Check if column contains mixed numeric and non-numeric values
        numeric_count = sum(pd.to_numeric(df[col], errors='coerce').notna())
        if 0 < numeric_count < len(df[col].dropna()):
            issues["inconsistent_types"][col] = {
                "numeric_count": numeric_count,
                "non_numeric_count": len(df[col].dropna()) - numeric_count,
                "percent_numeric": (numeric_count / len(df[col].dropna())) * 100
            }
    
    # High cardinality categorical columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.5 and df[col].nunique() > 10:
            issues["high_cardinality"][col] = {
                "unique_count": df[col].nunique(),
                "unique_ratio": unique_ratio
            }
            
            # Potential ID columns
            if unique_ratio > 0.9:
                issues["potential_id_columns"].append(col)
    
    # Imbalanced categories
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() < 10:  # Only check columns with reasonable number of categories
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.max() > 0.8:  # If dominant category > 80%
                issues["imbalanced_categories"][col] = {
                    "dominant_category": value_counts.idxmax(),
                    "dominant_percent": float(value_counts.max() * 100)
                }
    
    # Zero variance columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            issues["zero_variance"].append(col)
    
    # High correlation between numeric features
    if len(df.select_dtypes(include=['number']).columns) > 1:
        corr_matrix = df.select_dtypes(include=['number']).corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(i, j, corr_matrix.loc[i, j]) 
                          for i in upper_tri.index 
                          for j in upper_tri.columns 
                          if upper_tri.loc[i, j] > 0.9]
        
        if high_corr_pairs:
            issues["high_correlation"] = [
                {"col1": col1, "col2": col2, "correlation": float(corr)}
                for col1, col2, corr in high_corr_pairs
            ]
    
    return issues

def suggest_preprocessing_steps(df: pd.DataFrame, issues: Dict) -> List[Dict]:
    """
    Suggest preprocessing steps based on dataset and detected issues.
    
    Args:
        df: Input DataFrame
        issues: Dictionary of detected issues
        
    Returns:
        List of suggested preprocessing steps
    """
    suggestions = []
    
    # Missing values handling
    if issues["missing_values"]:
        for col, info in issues["missing_values"].items():
            if info["percent"] < 5:  # Low missing percentage
                if df[col].dtype.kind in 'ifc':  # numeric
                    suggestions.append({
                        "step": "impute_missing",
                        "column": col,
                        "method": "median",
                        "reason": f"Fill {info['percent']:.1f}% missing values with median (low missing rate)"
                    })
                else:  # categorical
                    suggestions.append({
                        "step": "impute_missing",
                        "column": col,
                        "method": "mode",
                        "reason": f"Fill {info['percent']:.1f}% missing values with mode (low missing rate)"
                    })
            elif info["percent"] < 30:  # Moderate missing percentage
                if df[col].dtype.kind in 'ifc':  # numeric
                    # Check if it's a time series (has date/time in column name)
                    if any(term in col.lower() for term in ['date', 'time', 'day', 'month', 'year']):
                        suggestions.append({
                            "step": "impute_missing",
                            "column": col,
                            "method": "interpolate",
                            "reason": f"Fill {info['percent']:.1f}% missing values with interpolation (time series data)"
                        })
                    else:
                        suggestions.append({
                            "step": "impute_missing",
                            "column": col,
                            "method": "knn",
                            "reason": f"Fill {info['percent']:.1f}% missing values with KNN imputation"
                        })
                else:  # categorical
                    suggestions.append({
                        "step": "impute_missing",
                        "column": col,
                        "method": "new_category",
                        "reason": f"Create new category for {info['percent']:.1f}% missing values"
                    })
            else:  # High missing percentage
                suggestions.append({
                    "step": "drop_column",
                    "column": col,
                    "reason": f"Drop column with {info['percent']:.1f}% missing values"
                })
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        duplicate_percent = (duplicates / len(df)) * 100
        suggestions.append({
            "step": "remove_duplicates",
            "method": "all_columns",
            "reason": f"Remove {duplicates} duplicate rows ({duplicate_percent:.1f}% of data)"
        })
    
    # Outlier handling
    if issues["outliers"]:
        for col, info in issues["outliers"].items():
            if info["percent"] < 1:  # Very few outliers
                suggestions.append({
                    "step": "handle_outliers",
                    "column": col,
                    "method": "remove",
                    "reason": f"Remove {info['count']} outliers ({info['percent']:.1f}%)"
                })
            elif info["percent"] < 5:  # Some outliers
                suggestions.append({
                    "step": "handle_outliers",
                    "column": col,
                    "method": "winsorize",
                    "reason": f"Winsorize {info['count']} outliers ({info['percent']:.1f}%)"
                })
            else:  # Many outliers - might be a legitimate distribution
                suggestions.append({
                    "step": "transform",
                    "column": col,
                    "method": "log",
                    "reason": f"Apply log transform to handle skewed distribution with {info['percent']:.1f}% outliers"
                })
    
    # Inconsistent types
    if issues["inconsistent_types"]:
        for col, info in issues["inconsistent_types"].items():
            if info["percent_numeric"] > 80:  # Mostly numeric
                suggestions.append({
                    "step": "convert_type",
                    "column": col,
                    "method": "to_numeric",
                    "reason": f"Convert to numeric ({info['percent_numeric']:.1f}% are numeric values)"
                })
            else:
                suggestions.append({
                    "step": "convert_type",
                    "column": col,
                    "method": "to_categorical",
                    "reason": "Convert to categorical (mixed types)"
                })
    
    # High cardinality
    if issues["high_cardinality"]:
        for col, info in issues["high_cardinality"].items():
            if col not in issues["potential_id_columns"]:
                suggestions.append({
                    "step": "reduce_cardinality",
                    "column": col,
                    "method": "group_rare",
                    "reason": f"Group rare categories ({info['unique_count']} unique values)"
                })
    
    # Potential ID columns
    for col in issues["potential_id_columns"]:
        suggestions.append({
            "step": "drop_column",
            "column": col,
            "reason": "Potential ID column with high cardinality"
        })
    
    # Zero variance
    for col in issues["zero_variance"]:
        suggestions.append({
            "step": "drop_column",
            "column": col,
            "reason": "Zero variance column (constant value)"
        })
    
    # High correlation
    if issues["high_correlation"]:
        for corr_info in issues["high_correlation"]:
            suggestions.append({
                "step": "handle_correlation",
                "columns": [corr_info["col1"], corr_info["col2"]],
                "method": "drop_one",
                "reason": f"High correlation ({corr_info['correlation']:.2f}) between features"
            })
    
    # Encoding categorical variables
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col not in [s["column"] for s in suggestions if s.get("step") == "drop_column"]:
            # Check if it might be text data (longer strings)
            if df[col].astype(str).str.len().mean() > 20:
                suggestions.append({
                    "step": "text_processing",
                    "column": col,
                    "method": "tfidf",
                    "reason": f"Column '{col}' appears to contain text data - apply TF-IDF vectorization"
                })
            elif df[col].nunique() == 2:
                suggestions.append({
                    "step": "encode",
                    "column": col,
                    "method": "label",
                    "reason": "Binary categorical variable - use label encoding"
                })
            elif df[col].nunique() <= 10:
                suggestions.append({
                    "step": "encode",
                    "column": col,
                    "method": "onehot",
                    "reason": f"Categorical with {df[col].nunique()} categories - use one-hot encoding"
                })
            elif df[col].nunique() <= 30:
                suggestions.append({
                    "step": "encode",
                    "column": col,
                    "method": "frequency",
                    "reason": f"Medium cardinality categorical ({df[col].nunique()} categories) - use frequency encoding"
                })
            else:
                suggestions.append({
                    "step": "encode",
                    "column": col,
                    "method": "target",
                    "reason": f"High cardinality categorical ({df[col].nunique()} categories) - use target encoding"
                })
    
    # Scaling numeric features
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        # Check if ranges are very different
        ranges = {col: df[col].max() - df[col].min() for col in numeric_cols if not pd.isna(df[col].max()) and not pd.isna(df[col].min())}
        if ranges and max(ranges.values()) / min(ranges.values()) > 10:
            suggestions.append({
                "step": "scale",
                "columns": list(numeric_cols),
                "method": "standard",
                "reason": "Features have very different scales - use standardization"
            })
    
    # Check for datetime columns
    datetime_cols = []
    for col in df.columns:
        # Check if column is already datetime
        if pd.api.types.is_datetime64_dtype(df[col]):
            datetime_cols.append(col)
        # Check if column name suggests datetime
        elif any(term in col.lower() for term in ['date', 'time', 'day', 'month', 'year']):
            try:
                pd.to_datetime(df[col], errors='raise')
                datetime_cols.append(col)
            except:
                pass
    
    # Suggest time features for datetime columns
    for col in datetime_cols:
        suggestions.append({
            "step": "time_features",
            "column": col,
            "method": "date_parts",
            "reason": f"Extract date components from datetime column '{col}'"
        })
        
        # Also suggest cyclical encoding for better ML performance
        suggestions.append({
            "step": "time_features",
            "column": col,
            "method": "cyclical",
            "reason": f"Create cyclical features from datetime column '{col}' for better ML performance"
        })
    
    # Check for potential target columns (binary or multi-class with few classes)
    potential_target_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 10:
                potential_target_cols.append((col, n_unique))
    
    # If we have potential target columns, suggest feature selection and handling imbalance
    if potential_target_cols:
        # Sort by number of classes (prefer binary targets)
        potential_target_cols.sort(key=lambda x: x[1])
        target_col, n_classes = potential_target_cols[0]
        
        # Check for class imbalance
        value_counts = df[target_col].value_counts(normalize=True)
        if value_counts.max() > 0.7:  # Imbalanced if dominant class > 70%
            suggestions.append({
                "step": "handle_imbalance",
                "column": target_col,
                "method": "smote",
                "reason": f"Address class imbalance in potential target column '{target_col}'"
            })
        
        # Suggest feature selection
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 5:  # Only if we have enough numeric features
            suggestions.append({
                "step": "feature_selection",
                "columns": numeric_cols,
                "target_column": target_col,
                "method": "random_forest",
                "reason": f"Select most important features for predicting '{target_col}'"
            })
    
    # If we have many numeric features, suggest dimensionality reduction
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) >= 10:
        suggestions.append({
            "step": "dimensionality_reduction",
            "columns": list(numeric_cols),
            "method": "pca",
            "n_components": min(5, len(numeric_cols)),
            "reason": f"Reduce dimensionality of {len(numeric_cols)} numeric features using PCA"
        })
    
    # If we have multiple numeric features, suggest feature interactions
    if len(numeric_cols) >= 2:
        # Find pairs of columns that might have meaningful interactions
        # For simplicity, just suggest the first few pairs
        pairs = []
        for i, col1 in enumerate(numeric_cols[:5]):  # Limit to first 5 columns
            for col2 in numeric_cols[i+1:min(i+3, len(numeric_cols))]:  # Limit to 2 pairs per column
                pairs.append([col1, col2])
        
        if pairs:
            suggestions.append({
                "step": "feature_interaction",
                "columns": pairs[0],  # Just suggest the first pair
                "method": "multiplication",
                "reason": f"Create interaction feature between '{pairs[0][0]}' and '{pairs[0][1]}'"
            })
    
    # Suggest binning for numeric columns with many unique values
    for col in numeric_cols:
        if df[col].nunique() > 20:  # Only suggest for columns with many unique values
            suggestions.append({
                "step": "binning",
                "column": col,
                "method": "equal_width",
                "n_bins": 5,
                "reason": f"Convert continuous variable '{col}' into categorical bins"
            })
    
    return suggestions

def apply_preprocessing_step(df: pd.DataFrame, step_config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply a preprocessing step to a DataFrame.
    
    Args:
        df: DataFrame to preprocess
        step_config: Configuration for the preprocessing step
        
    Returns:
        Tuple of (processed DataFrame, list of messages)
    """
    step_id = step_config.get("step")
    processed_df = df.copy()
    messages = []
    
    try:
        if step_id == "impute_missing":
            column = step_config.get("column")
            method = step_config.get("method", "median")
            
            if column not in processed_df.columns:
                return processed_df, [f"Column '{column}' not found"]
            
            # Count missing values before imputation
            missing_before = processed_df[column].isna().sum()
            
            if method == "mean":
                if not pd.api.types.is_numeric_dtype(processed_df[column]):
                    return processed_df, [f"Mean imputation requires numeric data"]
                
                value = processed_df[column].mean()
                processed_df[column] = processed_df[column].fillna(value)
                messages.append(f"Imputed {missing_before} missing values in '{column}' with mean ({value:.2f})")
            
            elif method == "median":
                if not pd.api.types.is_numeric_dtype(processed_df[column]):
                    return processed_df, [f"Median imputation requires numeric data"]
                
                value = processed_df[column].median()
                processed_df[column] = processed_df[column].fillna(value)
                messages.append(f"Imputed {missing_before} missing values in '{column}' with median ({value:.2f})")
            
            elif method == "mode":
                value = processed_df[column].mode()[0]
                processed_df[column] = processed_df[column].fillna(value)
                messages.append(f"Imputed {missing_before} missing values in '{column}' with mode ({value})")
            
            elif method == "constant":
                value = step_config.get("value", 0)
                processed_df[column] = processed_df[column].fillna(value)
                messages.append(f"Imputed {missing_before} missing values in '{column}' with constant ({value})")
            
            elif method == "knn":
                if not pd.api.types.is_numeric_dtype(processed_df[column]):
                    return processed_df, [f"KNN imputation requires numeric data"]
                
                # Simple KNN imputation (placeholder)
                # In a real implementation, you would use a proper KNN imputer
                value = processed_df[column].mean()
                processed_df[column] = processed_df[column].fillna(value)
                messages.append(f"Imputed {missing_before} missing values in '{column}' with KNN (simplified)")
            
            elif method == "new_category":
                processed_df[column] = processed_df[column].fillna("Missing")
                messages.append(f"Imputed {missing_before} missing values in '{column}' with new category 'Missing'")
            
            elif method == "forward_fill":
                # Forward fill (propagate last valid observation forward)
                processed_df[column] = processed_df[column].ffill()
                filled_count = missing_before - processed_df[column].isna().sum()
                messages.append(f"Forward filled {filled_count} missing values in '{column}'")
                
                # Check if any missing values remain (e.g., at the beginning of the series)
                remaining_missing = processed_df[column].isna().sum()
                if remaining_missing > 0:
                    messages.append(f"Note: {remaining_missing} missing values at the beginning of '{column}' could not be filled")
            
            elif method == "backward_fill":
                # Backward fill (propagate next valid observation backward)
                processed_df[column] = processed_df[column].bfill()
                filled_count = missing_before - processed_df[column].isna().sum()
                messages.append(f"Backward filled {filled_count} missing values in '{column}'")
                
                # Check if any missing values remain (e.g., at the end of the series)
                remaining_missing = processed_df[column].isna().sum()
                if remaining_missing > 0:
                    messages.append(f"Note: {remaining_missing} missing values at the end of '{column}' could not be filled")
            
            elif method == "interpolate":
                if not pd.api.types.is_numeric_dtype(processed_df[column]):
                    return processed_df, [f"Interpolation requires numeric data"]
                
                # Linear interpolation
                processed_df[column] = processed_df[column].interpolate(method='linear')
                filled_count = missing_before - processed_df[column].isna().sum()
                messages.append(f"Interpolated {filled_count} missing values in '{column}'")
                
                # Check if any missing values remain (e.g., at the edges)
                remaining_missing = processed_df[column].isna().sum()
                if remaining_missing > 0:
                    # Fill remaining missing values with forward/backward fill
                    processed_df[column] = processed_df[column].ffill().bfill()
                    messages.append(f"Filled {remaining_missing} edge missing values in '{column}' with forward/backward fill")
        
        elif step_id == "remove_duplicates":
            method = step_config.get("method", "all_columns")
            
            # Count duplicates before removal
            if method == "all_columns":
                # Consider all columns
                duplicates_count = processed_df.duplicated().sum()
                if duplicates_count > 0:
                    processed_df = processed_df.drop_duplicates()
                    messages.append(f"Removed {duplicates_count} duplicate rows (considering all columns)")
                else:
                    messages.append("No duplicate rows found (considering all columns)")
            
            elif method == "subset":
                # Consider subset of columns
                if "columns" in step_config and isinstance(step_config["columns"], list):
                    columns = step_config["columns"]
                    # Check if all specified columns exist
                    missing_cols = [col for col in columns if col not in processed_df.columns]
                    if missing_cols:
                        return processed_df, [f"Columns not found: {', '.join(missing_cols)}"]
                    
                    duplicates_count = processed_df.duplicated(subset=columns).sum()
                    if duplicates_count > 0:
                        processed_df = processed_df.drop_duplicates(subset=columns)
                        messages.append(f"Removed {duplicates_count} duplicate rows (considering columns: {', '.join(columns)})")
                    else:
                        messages.append(f"No duplicate rows found (considering columns: {', '.join(columns)})")
                else:
                    messages.append("No columns specified for duplicate detection")
            
            elif method in ["keep_first", "keep_last"]:
                keep_param = "first" if method == "keep_first" else "last"
                
                # Check if subset is specified
                if "columns" in step_config and isinstance(step_config["columns"], list):
                    columns = step_config["columns"]
                    # Check if all specified columns exist
                    missing_cols = [col for col in columns if col not in processed_df.columns]
                    if missing_cols:
                        return processed_df, [f"Columns not found: {', '.join(missing_cols)}"]
                    
                    duplicates_count = processed_df.duplicated(subset=columns, keep=keep_param).sum()
                    if duplicates_count > 0:
                        processed_df = processed_df.drop_duplicates(subset=columns, keep=keep_param)
                        messages.append(f"Removed {duplicates_count} duplicate rows, keeping {keep_param} occurrence (considering columns: {', '.join(columns)})")
                    else:
                        messages.append(f"No duplicate rows found (considering columns: {', '.join(columns)})")
                else:
                    # Consider all columns
                    duplicates_count = processed_df.duplicated(keep=keep_param).sum()
                    if duplicates_count > 0:
                        processed_df = processed_df.drop_duplicates(keep=keep_param)
                        messages.append(f"Removed {duplicates_count} duplicate rows, keeping {keep_param} occurrence (considering all columns)")
                    else:
                        messages.append("No duplicate rows found (considering all columns)")
        
        elif step_id == "binning":
            column = step_config.get("column")
            method = step_config.get("method", "equal_width")
            
            if column not in processed_df.columns:
                return processed_df, [f"Column '{column}' not found"]
            
            if not pd.api.types.is_numeric_dtype(processed_df[column]):
                return processed_df, [f"Binning requires numeric data"]
            
            # Get number of bins (default to 5)
            n_bins = step_config.get("n_bins", 5)
            
            if method == "equal_width":
                # Equal-width binning
                bin_edges = np.linspace(
                    processed_df[column].min(),
                    processed_df[column].max(),
                    n_bins + 1
                )
                
                # Create bin labels
                bin_labels = [f"Bin {i+1}" for i in range(n_bins)]
                
                # Apply binning
                processed_df[f"{column}_binned"] = pd.cut(
                    processed_df[column],
                    bins=bin_edges,
                    labels=bin_labels,
                    include_lowest=True
                )
                
                messages.append(f"Created equal-width binning of '{column}' with {n_bins} bins")
                
                # Add bin edges to message
                bin_ranges = [f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}" for i in range(n_bins)]
                bin_info = ", ".join([f"{label}: {range_}" for label, range_ in zip(bin_labels, bin_ranges)])
                messages.append(f"Bin ranges: {bin_info}")
            
            elif method == "equal_frequency":
                # Equal-frequency (quantile) binning
                bin_edges = pd.qcut(
                    processed_df[column],
                    q=n_bins,
                    retbins=True,
                    duplicates='drop'
                )[1]
                
                # If we have fewer bins due to duplicates, adjust n_bins
                actual_n_bins = len(bin_edges) - 1
                
                # Create bin labels
                bin_labels = [f"Bin {i+1}" for i in range(actual_n_bins)]
                
                # Apply binning
                processed_df[f"{column}_binned"] = pd.qcut(
                    processed_df[column],
                    q=n_bins,
                    labels=bin_labels,
                    duplicates='drop'
                )
                
                messages.append(f"Created equal-frequency binning of '{column}' with {actual_n_bins} bins")
                
                # Add bin edges to message
                bin_ranges = [f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}" for i in range(actual_n_bins)]
                bin_info = ", ".join([f"{label}: {range_}" for label, range_ in zip(bin_labels, bin_ranges)])
                messages.append(f"Bin ranges: {bin_info}")
            
            elif method == "kmeans":
                # K-means binning
                from sklearn.cluster import KMeans
                
                # Reshape data for KMeans
                X = processed_df[column].values.reshape(-1, 1)
                
                # Apply KMeans
                kmeans = KMeans(n_clusters=n_bins, random_state=42)
                clusters = kmeans.fit_predict(X)
                
                # Get cluster centers
                centers = kmeans.cluster_centers_.flatten()
                
                # Sort clusters by centers
                cluster_order = np.argsort(centers)
                cluster_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(cluster_order)}
                
                # Apply mapping to get ordered clusters
                ordered_clusters = np.array([cluster_mapping[c] for c in clusters])
                
                # Create bin labels
                bin_labels = [f"Bin {i+1}" for i in range(n_bins)]
                
                # Assign bins
                processed_df[f"{column}_binned"] = pd.Categorical(
                    [bin_labels[c] for c in ordered_clusters],
                    categories=bin_labels,
                    ordered=True
                )
                
                messages.append(f"Created K-means binning of '{column}' with {n_bins} bins")
                
                # Add cluster centers to message
                sorted_centers = centers[cluster_order]
                centers_info = ", ".join([f"{label}: center at {center:.2f}" for label, center in zip(bin_labels, sorted_centers)])
                messages.append(f"Bin centers: {centers_info}")
            
            elif method == "custom":
                # Custom bins
                if "bin_edges" in step_config and isinstance(step_config["bin_edges"], list):
                    bin_edges = step_config["bin_edges"]
                    
                    # Ensure bin edges are in ascending order
                    bin_edges = sorted(bin_edges)
                    
                    # Create bin labels
                    n_bins = len(bin_edges) - 1
                    bin_labels = [f"Bin {i+1}" for i in range(n_bins)]
                    
                    # Apply binning
                    processed_df[f"{column}_binned"] = pd.cut(
                        processed_df[column],
                        bins=bin_edges,
                        labels=bin_labels,
                        include_lowest=True
                    )
                    
                    messages.append(f"Created custom binning of '{column}' with {n_bins} bins")
                    
                    # Add bin edges to message
                    bin_ranges = [f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}" for i in range(n_bins)]
                    bin_info = ", ".join([f"{label}: {range_}" for label, range_ in zip(bin_labels, bin_ranges)])
                    messages.append(f"Bin ranges: {bin_info}")
                else:
                    messages.append("No bin edges specified for custom binning")
        
        elif step_id == "drop_column":
            if "column" in step_config:
                column = step_config.get("column")
                if column not in processed_df.columns:
                    return processed_df, [f"Column '{column}' not found"]
                
                processed_df = processed_df.drop(columns=[column])
                messages.append(f"Dropped column '{column}'")
            elif "columns" in step_config and isinstance(step_config["columns"], list):
                columns = step_config.get("columns")
                # Filter to only include columns that exist
                valid_columns = [col for col in columns if col in processed_df.columns]
                if not valid_columns:
                    return processed_df, [f"None of the specified columns were found"]
                
                processed_df = processed_df.drop(columns=valid_columns)
                messages.append(f"Dropped columns: {', '.join(valid_columns)}")
        
        elif step_id == "handle_outliers":
            column = step_config.get("column")
            method = step_config.get("method", "winsorize")
            
            if column not in processed_df.columns:
                return processed_df, [f"Column '{column}' not found"]
            
            if not pd.api.types.is_numeric_dtype(processed_df[column]):
                return processed_df, [f"Outlier handling requires numeric data"]
            
            # Calculate IQR
            Q1 = processed_df[column].quantile(0.25)
            Q3 = processed_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((processed_df[column] < lower_bound) | (processed_df[column] > upper_bound)).sum()
            
            if method == "remove":
                processed_df = processed_df[(processed_df[column] >= lower_bound) & (processed_df[column] <= upper_bound)]
                messages.append(f"Removed {outliers} outliers from '{column}'")
            
            elif method == "winsorize":
                processed_df[column] = processed_df[column].clip(lower=lower_bound, upper=upper_bound)
                messages.append(f"Winsorized {outliers} outliers in '{column}'")
            
            elif method == "transform":
                # Log transform as an example
                if (processed_df[column] <= 0).any():
                    min_val = processed_df[column].min()
                    shift = abs(min_val) + 1 if min_val <= 0 else 0
                    processed_df[column] = np.log(processed_df[column] + shift)
                    messages.append(f"Applied log transform to '{column}' with shift {shift}")
                else:
                    processed_df[column] = np.log(processed_df[column])
                    messages.append(f"Applied log transform to '{column}'")
            
            elif method == "zscore":
                # Z-score method
                from scipy import stats
                
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(processed_df[column]))
                
                # Define threshold (default is 3)
                threshold = step_config.get("threshold", 3)
                
                # Identify outliers
                outliers = (z_scores > threshold).sum()
                
                if step_config.get("action", "remove") == "remove":
                    # Remove outliers
                    processed_df = processed_df[z_scores <= threshold]
                    messages.append(f"Removed {outliers} outliers from '{column}' using Z-score method (threshold: {threshold})")
                else:
                    # Replace outliers with threshold values
                    mean = processed_df[column].mean()
                    std = processed_df[column].std()
                    
                    # Replace high outliers
                    high_mask = processed_df[column] > mean + threshold * std
                    processed_df.loc[high_mask, column] = mean + threshold * std
                    
                    # Replace low outliers
                    low_mask = processed_df[column] < mean - threshold * std
                    processed_df.loc[low_mask, column] = mean - threshold * std
                    
                    messages.append(f"Capped {outliers} outliers in '{column}' using Z-score method (threshold: {threshold})")
            
            elif method == "iqr":
                # IQR method (already calculated above)
                if step_config.get("action", "remove") == "remove":
                    # Remove outliers
                    processed_df = processed_df[(processed_df[column] >= lower_bound) & (processed_df[column] <= upper_bound)]
                    messages.append(f"Removed {outliers} outliers from '{column}' using IQR method")
                else:
                    # Replace outliers with boundary values
                    processed_df[column] = processed_df[column].clip(lower=lower_bound, upper=upper_bound)
                    messages.append(f"Capped {outliers} outliers in '{column}' using IQR method")
        
        elif step_id == "transform":
            column = step_config.get("column")
            method = step_config.get("method", "log")
            
            if column not in processed_df.columns:
                return processed_df, [f"Column '{column}' not found"]
            
            if not pd.api.types.is_numeric_dtype(processed_df[column]):
                return processed_df, [f"Transformation requires numeric data"]
            
            if method == "log":
                # Check if we need to shift
                if (processed_df[column] <= 0).any():
                    min_val = processed_df[column].min()
                    shift = abs(min_val) + 1
                    processed_df[column] = np.log(processed_df[column] + shift)
                    messages.append(f"Applied log transform to '{column}' with shift {shift}")
                else:
                    processed_df[column] = np.log(processed_df[column])
                    messages.append(f"Applied log transform to '{column}'")
            
            elif method == "sqrt":
                # Check if we need to shift
                if (processed_df[column] < 0).any():
                    min_val = processed_df[column].min()
                    shift = abs(min_val)
                    processed_df[column] = np.sqrt(processed_df[column] + shift)
                    messages.append(f"Applied square root transform to '{column}' with shift {shift}")
                else:
                    processed_df[column] = np.sqrt(processed_df[column])
                    messages.append(f"Applied square root transform to '{column}'")
            
            elif method == "box-cox":
                # Box-Cox requires positive values
                if (processed_df[column] <= 0).any():
                    min_val = processed_df[column].min()
                    shift = abs(min_val) + 1
                    # Apply Box-Cox transform
                    processed_df[column], _ = stats.boxcox(processed_df[column] + shift)
                    messages.append(f"Applied Box-Cox transform to '{column}' with shift {shift}")
                else:
                    # Apply Box-Cox transform
                    processed_df[column], _ = stats.boxcox(processed_df[column])
                    messages.append(f"Applied Box-Cox transform to '{column}'")
        
        elif step_id == "convert_type":
            column = step_config.get("column")
            method = step_config.get("method", "to_numeric")
            
            if column not in processed_df.columns:
                return processed_df, [f"Column '{column}' not found"]
            
            if method == "to_numeric":
                processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce')
                messages.append(f"Converted '{column}' to numeric type")
            
            elif method == "to_categorical":
                processed_df[column] = processed_df[column].astype('category')
                messages.append(f"Converted '{column}' to categorical type")
            
            elif method == "to_datetime":
                processed_df[column] = pd.to_datetime(processed_df[column], errors='coerce')
                messages.append(f"Converted '{column}' to datetime type")
        
        elif step_id == "reduce_cardinality":
            column = step_config.get("column")
            method = step_config.get("method", "group_rare")
            threshold = step_config.get("threshold", 0.01)
            
            if column not in processed_df.columns:
                return processed_df, [f"Column '{column}' not found"]
            
            if method == "group_rare":
                # Calculate value counts
                value_counts = processed_df[column].value_counts(normalize=True)
                
                # Identify rare categories
                rare_categories = value_counts[value_counts < threshold].index.tolist()
                
                if rare_categories:
                    # Replace rare categories with 'Other'
                    processed_df[column] = processed_df[column].replace(rare_categories, 'Other')
                    messages.append(f"Grouped {len(rare_categories)} rare categories in '{column}' as 'Other'")
                else:
                    messages.append(f"No rare categories found in '{column}' below threshold {threshold}")
        
        elif step_id == "encode":
            column = step_config.get("column")
            method = step_config.get("method", "onehot")
            
            if column not in processed_df.columns:
                return processed_df, [f"Column '{column}' not found"]
            
            if method == "label":
                # Convert to string first to handle any data type
                processed_df[column] = processed_df[column].astype(str)
                
                # Create a mapping of categories to integers
                categories = processed_df[column].unique()
                mapping = {category: i for i, category in enumerate(categories)}
                
                # Apply the mapping
                processed_df[column] = processed_df[column].map(mapping)
                messages.append(f"Applied label encoding to '{column}' ({len(categories)} categories)")
            
            elif method == "onehot":
                # Get dummies
                dummies = pd.get_dummies(processed_df[column], prefix=column, drop_first=False)
                
                # Drop the original column and join the dummies
                processed_df = pd.concat([processed_df.drop(columns=[column]), dummies], axis=1)
                messages.append(f"Applied one-hot encoding to '{column}' ({dummies.shape[1]} new columns)")
            
            elif method == "target":
                # Target encoding (requires target variable)
                if "target_column" in step_config and step_config["target_column"] in processed_df.columns:
                    target_column = step_config["target_column"]
                    
                    # Calculate mean target value for each category
                    target_means = processed_df.groupby(column)[target_column].mean()
                    
                    # Apply encoding
                    processed_df[f"{column}_target_encoded"] = processed_df[column].map(target_means)
                    
                    # Handle missing values (new categories)
                    global_mean = processed_df[target_column].mean()
                    processed_df[f"{column}_target_encoded"].fillna(global_mean, inplace=True)
                    
                    messages.append(f"Applied target encoding to '{column}' using target '{target_column}'")
                    
                    # Option to drop original column
                    if step_config.get("drop_original", False):
                        processed_df = processed_df.drop(columns=[column])
                        messages.append(f"Dropped original column '{column}' after target encoding")
                else:
                    messages.append("Target encoding requires a target column")
            
            elif method == "frequency":
                # Frequency encoding
                freq_encoding = processed_df[column].value_counts(normalize=True)
                
                # Apply encoding
                processed_df[f"{column}_freq_encoded"] = processed_df[column].map(freq_encoding)
                
                messages.append(f"Applied frequency encoding to '{column}'")
                
                # Option to drop original column
                if step_config.get("drop_original", False):
                    processed_df = processed_df.drop(columns=[column])
                    messages.append(f"Dropped original column '{column}' after frequency encoding")
            
            elif method == "binary":
                try:
                    from category_encoders import BinaryEncoder
                    
                    # Create and fit binary encoder
                    encoder = BinaryEncoder(cols=[column])
                    binary_encoded = encoder.fit_transform(processed_df[column])
                    
                    # Rename columns to include original column name
                    binary_encoded.columns = [f"{column}_bin_{i}" for i in range(binary_encoded.shape[1])]
                    
                    # Concatenate with original DataFrame
                    processed_df = pd.concat([processed_df, binary_encoded], axis=1)
                    
                    messages.append(f"Applied binary encoding to '{column}' ({binary_encoded.shape[1]} new columns)")
                    
                    # Option to drop original column
                    if step_config.get("drop_original", True):
                        processed_df = processed_df.drop(columns=[column])
                        messages.append(f"Dropped original column '{column}' after binary encoding")
                except ImportError:
                    messages.append("Binary encoding requires category_encoders. Please install it with 'pip install category_encoders'")
            
            elif method == "ordinal":
                # Check if order is provided
                if "order" in step_config and isinstance(step_config["order"], list):
                    order = step_config["order"]
                    
                    # Create mapping
                    mapping = {category: i for i, category in enumerate(order)}
                    
                    # Apply mapping
                    processed_df[f"{column}_ordinal"] = processed_df[column].map(mapping)
                    
                    # Handle missing categories
                    missing_categories = processed_df[processed_df[f"{column}_ordinal"].isna()][column].unique()
                    if len(missing_categories) > 0:
                        messages.append(f"Warning: {len(missing_categories)} categories not found in provided order")
                        
                        # Fill missing with -1 or next value
                        next_value = len(order)
                        for cat in missing_categories:
                            mapping[cat] = next_value
                            next_value += 1
                        
                        # Reapply mapping
                        processed_df[f"{column}_ordinal"] = processed_df[column].map(mapping)
                    
                    messages.append(f"Applied ordinal encoding to '{column}' with custom order")
                else:
                    # Use alphabetical order
                    categories = sorted(processed_df[column].unique())
                    mapping = {category: i for i, category in enumerate(categories)}
                    
                    # Apply mapping
                    processed_df[f"{column}_ordinal"] = processed_df[column].map(mapping)
                    
                    messages.append(f"Applied ordinal encoding to '{column}' with alphabetical order")
                
                # Option to drop original column
                if step_config.get("drop_original", False):
                    processed_df = processed_df.drop(columns=[column])
                    messages.append(f"Dropped original column '{column}' after ordinal encoding")
            
            elif method == "count":
                # Count encoding
                count_encoding = processed_df[column].value_counts()
                
                # Apply encoding
                processed_df[f"{column}_count_encoded"] = processed_df[column].map(count_encoding)
                
                messages.append(f"Applied count encoding to '{column}'")
                
                # Option to drop original column
                if step_config.get("drop_original", False):
                    processed_df = processed_df.drop(columns=[column])
                    messages.append(f"Dropped original column '{column}' after count encoding")
        
        elif step_id == "handle_imbalance":
            column = step_config.get("column")  # Target column
            method = step_config.get("method", "smote")
            
            if column not in processed_df.columns:
                return processed_df, [f"Column '{column}' not found"]
            
            # Get feature columns (all except target)
            feature_columns = [col for col in processed_df.columns if col != column]
            
            # Check if there are any non-numeric features
            non_numeric = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(processed_df[col])]
            if non_numeric:
                messages.append(f"Warning: Non-numeric features detected. These will be excluded: {', '.join(non_numeric)}")
                feature_columns = [col for col in feature_columns if col not in non_numeric]
            
            if not feature_columns:
                return processed_df, ["No numeric feature columns available for imbalance handling"]
            
            # Get class distribution before
            class_counts_before = processed_df[column].value_counts()
            min_class_before = class_counts_before.min()
            max_class_before = class_counts_before.max()
            imbalance_ratio_before = max_class_before / min_class_before
            
            if method == "smote":
                try:
                    from imblearn.over_sampling import SMOTE
                    
                    # Create SMOTE instance
                    smote = SMOTE(random_state=42)
                    
                    # Apply SMOTE
                    X_resampled, y_resampled = smote.fit_resample(processed_df[feature_columns], processed_df[column])
                    
                    # Create new DataFrame
                    processed_df = pd.DataFrame(X_resampled, columns=feature_columns)
                    processed_df[column] = y_resampled
                    
                    # Get class distribution after
                    class_counts_after = processed_df[column].value_counts()
                    
                    messages.append(f"Applied SMOTE oversampling to balance '{column}'")
                    messages.append(f"Class distribution before: {dict(class_counts_before)}")
                    messages.append(f"Class distribution after: {dict(class_counts_after)}")
                except ImportError:
                    messages.append("SMOTE requires imbalanced-learn. Please install it with 'pip install imbalanced-learn'")
            
            elif method == "adasyn":
                try:
                    from imblearn.over_sampling import ADASYN
                    
                    # Create ADASYN instance
                    adasyn = ADASYN(random_state=42)
                    
                    # Apply ADASYN
                    X_resampled, y_resampled = adasyn.fit_resample(processed_df[feature_columns], processed_df[column])
                    
                    # Create new DataFrame
                    processed_df = pd.DataFrame(X_resampled, columns=feature_columns)
                    processed_df[column] = y_resampled
                    
                    # Get class distribution after
                    class_counts_after = processed_df[column].value_counts()
                    
                    messages.append(f"Applied ADASYN oversampling to balance '{column}'")
                    messages.append(f"Class distribution before: {dict(class_counts_before)}")
                    messages.append(f"Class distribution after: {dict(class_counts_after)}")
                except ImportError:
                    messages.append("ADASYN requires imbalanced-learn. Please install it with 'pip install imbalanced-learn'")
            
            elif method == "random_undersampling":
                try:
                    from imblearn.under_sampling import RandomUnderSampler
                    
                    # Create RandomUnderSampler instance
                    undersampler = RandomUnderSampler(random_state=42)
                    
                    # Apply random undersampling
                    X_resampled, y_resampled = undersampler.fit_resample(processed_df[feature_columns], processed_df[column])
                    
                    # Create new DataFrame
                    processed_df = pd.DataFrame(X_resampled, columns=feature_columns)
                    processed_df[column] = y_resampled
                    
                    # Get class distribution after
                    class_counts_after = processed_df[column].value_counts()
                    
                    messages.append(f"Applied random undersampling to balance '{column}'")
                    messages.append(f"Class distribution before: {dict(class_counts_before)}")
                    messages.append(f"Class distribution after: {dict(class_counts_after)}")
                except ImportError:
                    messages.append("Random undersampling requires imbalanced-learn. Please install it with 'pip install imbalanced-learn'")
            
            elif method == "class_weights":
                # Calculate class weights
                class_weights = {
                    cls: max_class_before / count 
                    for cls, count in class_counts_before.items()
                }
                
                # Store class weights in a new column
                processed_df[f"{column}_weight"] = processed_df[column].map(class_weights)
                
                messages.append(f"Added class weights for '{column}' (imbalance ratio: {imbalance_ratio_before:.2f})")
                messages.append(f"Class weights: {class_weights}")
        
        elif step_id == "text_processing":
            column = step_config.get("column")
            method = step_config.get("method", "tokenize")
            
            if column not in processed_df.columns:
                return processed_df, [f"Column '{column}' not found"]
            
            # Check if column is text/object type
            if not pd.api.types.is_string_dtype(processed_df[column]):
                try:
                    processed_df[column] = processed_df[column].astype(str)
                    messages.append(f"Converted '{column}' to string type")
                except:
                    return processed_df, [f"Could not convert '{column}' to string type"]
            
            if method == "tokenize":
                # Simple tokenization
                processed_df[f"{column}_tokens"] = processed_df[column].apply(lambda x: str(x).split())
                
                # Count tokens
                processed_df[f"{column}_token_count"] = processed_df[f"{column}_tokens"].apply(len)
                
                messages.append(f"Tokenized '{column}' and added token count")
            
            elif method == "stopwords":
                try:
                    import nltk
                    from nltk.corpus import stopwords
                    
                    # Download stopwords if not already downloaded
                    try:
                        nltk.data.find('corpora/stopwords')
                    except LookupError:
                        nltk.download('stopwords', quiet=True)
                    
                    # Get language (default to English)
                    language = step_config.get("language", "english")
                    
                    # Get stopwords for the language
                    stop_words = set(stopwords.words(language))
                    
                    # Tokenize and remove stopwords
                    processed_df[f"{column}_filtered"] = processed_df[column].apply(
                        lambda x: ' '.join([word for word in str(x).lower().split() if word not in stop_words])
                    )
                    
                    messages.append(f"Removed stopwords from '{column}' (language: {language})")
                except ImportError:
                    messages.append("Stopword removal requires NLTK. Please install it with 'pip install nltk'")
            
            elif method == "stemming":
                try:
                    import nltk
                    from nltk.stem import PorterStemmer, SnowballStemmer
                    
                    # Get stemmer type (default to Porter)
                    stemmer_type = step_config.get("stemmer", "porter")
                    
                    # Create stemmer
                    if stemmer_type == "porter":
                        stemmer = PorterStemmer()
                    elif stemmer_type == "snowball":
                        # Get language (default to English)
                        language = step_config.get("language", "english")
                        stemmer = SnowballStemmer(language)
                    else:
                        return processed_df, [f"Unknown stemmer type: {stemmer_type}"]
                    
                    # Apply stemming
                    processed_df[f"{column}_stemmed"] = processed_df[column].apply(
                        lambda x: ' '.join([stemmer.stem(word) for word in str(x).lower().split()])
                    )
                    
                    messages.append(f"Applied {stemmer_type} stemming to '{column}'")
                except ImportError:
                    messages.append("Stemming requires NLTK. Please install it with 'pip install nltk'")
            
            elif method == "lemmatization":
                try:
                    import nltk
                    from nltk.stem import WordNetLemmatizer
                    
                    # Download WordNet if not already downloaded
                    try:
                        nltk.data.find('corpora/wordnet')
                    except LookupError:
                        nltk.download('wordnet', quiet=True)
                        nltk.download('punkt', quiet=True)
                    
                    # Create lemmatizer
                    lemmatizer = WordNetLemmatizer()
                    
                    # Apply lemmatization
                    processed_df[f"{column}_lemmatized"] = processed_df[column].apply(
                        lambda x: ' '.join([lemmatizer.lemmatize(word) for word in str(x).lower().split()])
                    )
                    
                    messages.append(f"Applied lemmatization to '{column}'")
                except ImportError:
                    messages.append("Lemmatization requires NLTK. Please install it with 'pip install nltk'")
            
            elif method == "tfidf":
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    
                    # Get max features (default to 100)
                    max_features = step_config.get("max_features", 100)
                    
                    # Create and fit TF-IDF vectorizer
                    vectorizer = TfidfVectorizer(max_features=max_features)
                    tfidf_matrix = vectorizer.fit_transform(processed_df[column].fillna(''))
                    
                    # Get feature names
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Convert to DataFrame
                    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{column}_tfidf_{word}" for word in feature_names])
                    
                    # Concatenate with original DataFrame
                    processed_df = pd.concat([processed_df, tfidf_df], axis=1)
                    
                    messages.append(f"Created {len(feature_names)} TF-IDF features from '{column}'")
                except ImportError:
                    messages.append("TF-IDF vectorization requires scikit-learn. Please install it with 'pip install scikit-learn'")
            
            elif method == "word_embeddings":
                messages.append("Word embeddings require additional setup and are not implemented in this version")
        
        else:
            messages.append(f"Unknown preprocessing step: {step_id}")
    
    except Exception as e:
        messages.append(f"Error in {step_id}: {str(e)}")
    
    return processed_df, messages 