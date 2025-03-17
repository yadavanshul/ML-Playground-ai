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