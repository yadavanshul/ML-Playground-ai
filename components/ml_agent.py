import os
import json
import time
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MLMiniAgent:
    """
    ML Mini Agent that provides recommendations for machine learning models
    based on dataset characteristics, EDA insights, and preprocessing steps.
    """
    
    def __init__(self):
        """Initialize the ML Mini Agent."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4"
        self.reasoning_log = []
        self.last_response = None
    
    def log_reasoning(self, message: str):
        """Log reasoning steps for transparency."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.reasoning_log.append({
            "timestamp": timestamp,
            "message": message
        })
    
    def get_model_recommendations(self, 
                                 df: pd.DataFrame, 
                                 target_column: str,
                                 problem_type: str,
                                 eda_insights: Dict[str, Any],
                                 preprocessing_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get model recommendations based on dataset characteristics, EDA insights,
        and preprocessing steps.
        
        Args:
            df: The dataset (after preprocessing)
            target_column: The target column for prediction
            problem_type: 'classification' or 'regression'
            eda_insights: Insights from the EDA phase
            preprocessing_steps: List of preprocessing steps applied
            
        Returns:
            Dictionary with model recommendations and explanations
        """
        try:
            # Extract dataset characteristics
            n_rows, n_cols = df.shape
            n_numeric = len(df.select_dtypes(include=['number']).columns)
            n_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
            missing_values = df.isnull().sum().sum()
            
            # Create a summary of preprocessing steps
            preprocessing_summary = self._summarize_preprocessing_steps(preprocessing_steps)
            
            # Create a summary of EDA insights
            eda_summary = self._summarize_eda_insights(eda_insights)
            
            # Log the action
            self.log_reasoning(f"Generating model recommendations for {problem_type} problem with target '{target_column}'")
            
            # Instead of calling OpenAI API, use a mock implementation
            # Generate mock recommendations based on problem type
            if problem_type.lower() == "classification":
                recommendations = self._get_mock_classification_recommendations(df, target_column)
            else:
                recommendations = self._get_mock_regression_recommendations(df, target_column)
            
            self.last_response = recommendations
            self.log_reasoning(f"Generated recommendations for {len(recommendations.get('recommended_models', []))} models")
            
            return recommendations
                
        except Exception as e:
            self.log_reasoning(f"Error generating model recommendations: {str(e)}")
            return {
                "error": str(e)
            }
    
    def _get_mock_classification_recommendations(self, df, target_column):
        """Generate mock recommendations for classification problems."""
        # Check if target is binary or multi-class
        n_classes = df[target_column].nunique()
        is_binary = n_classes == 2
        is_imbalanced = df[target_column].value_counts(normalize=True).max() > 0.7
        
        recommendations = {
            "recommended_models": [
                {
                    "model_id": "random_forest",
                    "name": "Random Forest",
                    "explanation": "Random Forest is a versatile ensemble method that performs well on a wide range of classification problems.",
                    "strengths": [
                        "Handles both numerical and categorical features well",
                        "Robust to outliers and non-linear data",
                        "Provides feature importance metrics"
                    ],
                    "limitations": [
                        "Can be computationally intensive for very large datasets",
                        "May overfit on noisy data without proper tuning"
                    ]
                },
                {
                    "model_id": "gradient_boosting",
                    "name": "Gradient Boosting",
                    "explanation": "Gradient Boosting often achieves state-of-the-art results on classification tasks through iterative optimization.",
                    "strengths": [
                        "Typically provides high accuracy",
                        "Handles complex relationships in data",
                        "Less prone to overfitting than decision trees"
                    ],
                    "limitations": [
                        "Requires careful tuning of hyperparameters",
                        "Training can be time-consuming"
                    ]
                }
            ],
            "additional_preprocessing": [],
            "potential_challenges": [],
            "general_advice": "For this classification problem, ensemble methods like Random Forest and Gradient Boosting are likely to perform well."
        }
        
        # Add logistic regression for binary classification
        if is_binary:
            recommendations["recommended_models"].append({
                "model_id": "logistic_regression",
                "name": "Logistic Regression",
                "explanation": "Logistic Regression is a good baseline for binary classification problems, especially when interpretability is important.",
                "strengths": [
                    "Highly interpretable results",
                    "Works well for linearly separable data",
                    "Fast training and prediction"
                ],
                "limitations": [
                    "May underperform on complex, non-linear relationships",
                    "Sensitive to outliers"
                ]
            })
        
        # Add SVM for smaller datasets
        if df.shape[0] < 10000:
            recommendations["recommended_models"].append({
                "model_id": "svm",
                "name": "Support Vector Machine",
                "explanation": "SVM can be effective for classification tasks with clear margins of separation.",
                "strengths": [
                    "Effective in high-dimensional spaces",
                    "Versatile through different kernel functions",
                    "Works well when classes are separable"
                ],
                "limitations": [
                    "Doesn't scale well to larger datasets",
                    "Sensitive to choice of kernel and regularization parameters"
                ]
            })
        
        # Add advice for imbalanced data
        if is_imbalanced:
            recommendations["additional_preprocessing"].append(
                "Consider using class weights or resampling techniques to address class imbalance"
            )
            recommendations["potential_challenges"].append(
                "Class imbalance may lead to biased models that favor the majority class"
            )
        
        return recommendations
    
    def _get_mock_regression_recommendations(self, df, target_column):
        """Generate mock recommendations for regression problems."""
        # Check dataset characteristics
        n_features = len(df.columns) - 1  # Excluding target
        has_categorical = len(df.select_dtypes(include=['object', 'category']).columns) > 0
        
        recommendations = {
            "recommended_models": [
                {
                    "model_id": "random_forest",
                    "name": "Random Forest Regressor",
                    "explanation": "Random Forest is a robust ensemble method that performs well on a wide range of regression problems.",
                    "strengths": [
                        "Handles both numerical and categorical features well",
                        "Robust to outliers and non-linear data",
                        "Provides feature importance metrics"
                    ],
                    "limitations": [
                        "Can be computationally intensive for very large datasets",
                        "May struggle with extrapolation beyond the range of training data"
                    ]
                },
                {
                    "model_id": "gradient_boosting",
                    "name": "Gradient Boosting Regressor",
                    "explanation": "Gradient Boosting often achieves state-of-the-art results on regression tasks through iterative optimization.",
                    "strengths": [
                        "Typically provides high accuracy",
                        "Handles complex relationships in data",
                        "Less prone to overfitting than decision trees"
                    ],
                    "limitations": [
                        "Requires careful tuning of hyperparameters",
                        "Training can be time-consuming"
                    ]
                }
            ],
            "additional_preprocessing": [],
            "potential_challenges": [],
            "general_advice": "For this regression problem, ensemble methods like Random Forest and Gradient Boosting are likely to perform well."
        }
        
        # Add linear regression for simpler problems
        if n_features < 20 and not has_categorical:
            recommendations["recommended_models"].append({
                "model_id": "linear_regression",
                "name": "Linear Regression",
                "explanation": "Linear Regression is a good baseline for regression problems, especially when interpretability is important.",
                "strengths": [
                    "Highly interpretable results",
                    "Fast training and prediction",
                    "Works well for linear relationships"
                ],
                "limitations": [
                    "May underperform on complex, non-linear relationships",
                    "Sensitive to outliers"
                ]
            })
        
        # Add SVR for smaller datasets
        if df.shape[0] < 10000:
            recommendations["recommended_models"].append({
                "model_id": "svr",
                "name": "Support Vector Regressor",
                "explanation": "SVR can be effective for regression tasks with complex patterns.",
                "strengths": [
                    "Effective in high-dimensional spaces",
                    "Versatile through different kernel functions",
                    "Robust to outliers with proper parameters"
                ],
                "limitations": [
                    "Doesn't scale well to larger datasets",
                    "Sensitive to choice of kernel and regularization parameters"
                ]
            })
        
        # Add advice for feature engineering
        if n_features > 10:
            recommendations["additional_preprocessing"].append(
                "Consider feature selection or dimensionality reduction to improve model performance"
            )
        
        return recommendations
    
    def _summarize_preprocessing_steps(self, preprocessing_steps: List[Dict[str, Any]]) -> str:
        """Summarize preprocessing steps into a readable format."""
        if not preprocessing_steps:
            return "No preprocessing steps applied."
        
        summary = []
        for i, step in enumerate(preprocessing_steps):
            step_type = step.get("step", "unknown")
            column = step.get("column", "multiple columns")
            method = step.get("method", "")
            
            if step_type == "impute_missing":
                summary.append(f"{i+1}. Missing value imputation on '{column}' using {method}")
            elif step_type == "drop_column":
                summary.append(f"{i+1}. Dropped column '{column}'")
            elif step_type == "drop_duplicates":
                summary.append(f"{i+1}. Removed duplicate rows")
            elif step_type == "handle_outliers":
                summary.append(f"{i+1}. Outlier handling on '{column}' using {method}")
            elif step_type == "scale_features":
                summary.append(f"{i+1}. Feature scaling on '{column}' using {method}")
            elif step_type == "encode_categorical":
                summary.append(f"{i+1}. Encoded '{column}' using {method}")
            elif step_type == "feature_engineering":
                summary.append(f"{i+1}. Feature engineering: {method} on '{column}'")
            elif step_type == "dimensionality_reduction":
                summary.append(f"{i+1}. Dimensionality reduction using {method}")
            elif step_type == "handle_imbalance":
                summary.append(f"{i+1}. Handled class imbalance using {method}")
            else:
                summary.append(f"{i+1}. {step_type} on '{column}'")
        
        return "\n".join(summary)
    
    def _summarize_eda_insights(self, eda_insights: Dict[str, Any]) -> str:
        """Summarize EDA insights into a readable format."""
        if not eda_insights:
            return "No EDA insights available."
        
        summary = []
        
        # Extract summary information
        if "summary" in eda_insights:
            summary_info = eda_insights["summary"]
            if "rows" in summary_info and "columns" in summary_info:
                summary.append(f"Dataset has {summary_info['rows']} rows and {summary_info['columns']} columns.")
            
            if "column_types" in summary_info:
                col_types = summary_info["column_types"]
                summary.append(f"Column types: {', '.join([f'{k}: {v}' for k, v in col_types.items()])}")
        
        # Extract data issues
        if "issues" in eda_insights:
            issues = eda_insights["issues"]
            
            if "missing_values" in issues and issues["missing_values"]:
                missing = issues["missing_values"]
                summary.append(f"Missing values detected in {len(missing)} columns.")
            
            if "outliers" in issues and issues["outliers"]:
                outliers = issues["outliers"]
                summary.append(f"Outliers detected in {len(outliers)} columns: {', '.join(outliers)}")
            
            if "high_correlation" in issues and issues["high_correlation"]:
                high_corr = issues["high_correlation"]
                summary.append(f"High correlation detected between {len(high_corr)} pairs of features.")
        
        # Extract insights
        if "insights" in eda_insights:
            insights = eda_insights["insights"]
            for insight in insights:
                summary.append(f"Insight: {insight}")
        
        return "\n".join(summary)
    
    def get_model_evaluation_insights(self, 
                                     trained_models: Dict[str, Any],
                                     best_model_id: str,
                                     problem_type: str) -> Dict[str, Any]:
        """
        Generate insights and recommendations based on model evaluation results.
        
        Args:
            trained_models: Dictionary of trained models with their evaluation metrics
            best_model_id: ID of the best performing model
            problem_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with insights and recommendations
        """
        try:
            # Extract best model information
            best_model = trained_models.get(best_model_id, {})
            best_model_name = best_model.get("name", "Unknown")
            
            # Log the action
            self.log_reasoning(f"Generating model evaluation insights for {best_model_name}")
            
            # Instead of calling OpenAI API, use a mock implementation
            insights = self._get_mock_model_insights(trained_models, best_model_id, problem_type)
            
            self.log_reasoning(f"Generated model evaluation insights for {best_model_name}")
            
            return insights
                
        except Exception as e:
            self.log_reasoning(f"Error generating model evaluation insights: {str(e)}")
            return {
                "error": str(e)
            }
    
    def _get_mock_model_insights(self, trained_models, best_model_id, problem_type):
        """Generate mock insights for model evaluation."""
        best_model = trained_models.get(best_model_id, {})
        best_model_name = best_model.get("name", "Unknown")
        
        # Get metrics for best model
        best_metrics = best_model.get("metrics", {})
        
        # Get feature importance if available
        feature_importance = {}
        if "feature_importance" in best_model and "feature_importance" in best_model["feature_importance"]:
            feature_importance = best_model["feature_importance"]["feature_importance"]
        
        # Generate insights based on problem type
        if problem_type.lower() == "classification":
            accuracy = best_metrics.get("accuracy", 0)
            f1 = best_metrics.get("f1", 0)
            
            performance_level = "excellent" if accuracy > 0.9 else "good" if accuracy > 0.8 else "moderate" if accuracy > 0.7 else "poor"
            
            insights = {
                "performance_analysis": f"The {best_model_name} achieved {performance_level} performance with an accuracy of {accuracy:.2f} and F1 score of {f1:.2f}. This indicates that the model is able to correctly classify instances with reasonable reliability.",
                
                "model_comparison": "Ensemble methods like Random Forest and Gradient Boosting typically outperform simpler models on classification tasks due to their ability to capture complex patterns in the data.",
                
                "feature_insights": "The most important features for prediction are shown in the feature importance plot. These features have the strongest influence on the model's predictions and should be carefully considered when interpreting results.",
                
                "improvement_recommendations": [
                    "Consider hyperparameter tuning to further improve model performance",
                    "Experiment with feature engineering to create more informative features",
                    "Try ensemble methods that combine multiple models for better predictions",
                    "Evaluate the model on different metrics depending on your specific business needs"
                ],
                
                "next_steps": [
                    "Deploy the model to a production environment",
                    "Set up monitoring to track model performance over time",
                    "Collect feedback to continuously improve the model",
                    "Document the model's strengths and limitations for stakeholders"
                ]
            }
        else:  # regression
            rmse = best_metrics.get("rmse", 0)
            r2 = best_metrics.get("r2", 0)
            
            performance_level = "excellent" if r2 > 0.9 else "good" if r2 > 0.7 else "moderate" if r2 > 0.5 else "poor"
            
            insights = {
                "performance_analysis": f"The {best_model_name} achieved {performance_level} performance with an RMSE of {rmse:.2f} and R² of {r2:.2f}. This indicates how well the model is able to predict the target variable.",
                
                "model_comparison": "Ensemble methods like Random Forest and Gradient Boosting typically outperform simpler models on regression tasks due to their ability to capture non-linear relationships in the data.",
                
                "feature_insights": "The most important features for prediction are shown in the feature importance plot. These features have the strongest influence on the model's predictions and should be carefully considered when interpreting results.",
                
                "improvement_recommendations": [
                    "Consider hyperparameter tuning to further improve model performance",
                    "Experiment with feature engineering to create more informative features",
                    "Try ensemble methods that combine multiple models for better predictions",
                    "Consider transforming the target variable if the distribution is skewed"
                ],
                
                "next_steps": [
                    "Deploy the model to a production environment",
                    "Set up monitoring to track model performance over time",
                    "Collect feedback to continuously improve the model",
                    "Document the model's strengths and limitations for stakeholders"
                ]
            }
        
        return insights 