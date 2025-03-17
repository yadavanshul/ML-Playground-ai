import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    explained_variance_score, max_error, median_absolute_error
)
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define available models
CLASSIFICATION_MODELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost"
}

REGRESSION_MODELS = {
    "linear_regression": "Linear Regression",
    "decision_tree": "Decision Tree Regressor",
    "random_forest": "Random Forest Regressor",
    "svr": "Support Vector Regressor",
    "knn": "K-Nearest Neighbors Regressor",
    "gradient_boosting": "Gradient Boosting Regressor",
    "xgboost": "XGBoost Regressor"
}

def get_available_models(problem_type: str) -> Dict[str, str]:
    """
    Get available models for the given problem type.
    
    Args:
        problem_type: 'classification' or 'regression'
        
    Returns:
        Dictionary of model_id -> model_name
    """
    if problem_type.lower() == "classification":
        return CLASSIFICATION_MODELS
    else:
        return REGRESSION_MODELS

def prepare_data_for_ml(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for machine learning by splitting into train and test sets.
    
    Args:
        df: DataFrame containing the data
        target_column: Name of the target column
        feature_columns: List of feature column names
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified sampling (for classification)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Extract features and target
    X = df[feature_columns].values
    y = df[target_column].values
    
    # Split data
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=df[target_column]
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    return X_train, X_test, y_train, y_test

def get_model_instance(model_id: str, problem_type: str, **kwargs) -> Any:
    """
    Get a model instance based on model_id and problem_type.
    
    Args:
        model_id: ID of the model to instantiate
        problem_type: 'classification' or 'regression'
        **kwargs: Additional parameters for the model
        
    Returns:
        Model instance
    """
    # This is a simulation, so we'll just return a dictionary with model info
    # In a real implementation, this would instantiate actual ML models
    
    if problem_type.lower() == "classification":
        if model_id == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**kwargs)
        elif model_id == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(**kwargs)
        elif model_id == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**kwargs)
        elif model_id == "svm":
            from sklearn.svm import SVC
            return SVC(**kwargs)
        elif model_id == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(**kwargs)
        elif model_id == "naive_bayes":
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB(**kwargs)
        elif model_id == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**kwargs)
        elif model_id == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(**kwargs)
            except ImportError:
                # Fallback if XGBoost is not installed
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(**kwargs)
    else:  # regression
        if model_id == "linear_regression":
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**kwargs)
        elif model_id == "decision_tree":
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor(**kwargs)
        elif model_id == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**kwargs)
        elif model_id == "svr":
            from sklearn.svm import SVR
            return SVR(**kwargs)
        elif model_id == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(**kwargs)
        elif model_id == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**kwargs)
        elif model_id == "xgboost":
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(**kwargs)
            except ImportError:
                # Fallback if XGBoost is not installed
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(**kwargs)
    
    # Default fallback
    return {
        "model_id": model_id,
        "problem_type": problem_type,
        "params": kwargs
    }

def train_and_evaluate_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    problem_type: str
) -> Dict[str, Any]:
    """
    Train and evaluate a model.
    
    Args:
        model: Model instance to train
        X_train: Training features
        X_test: Testing features
        y_train: Training targets
        y_test: Testing targets
        problem_type: 'classification' or 'regression'
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {}
    if problem_type.lower() == "classification":
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        
        # For multi-class classification, we need to specify average
        try:
            metrics["precision"] = precision_score(y_test, y_pred, average='weighted')
            metrics["recall"] = recall_score(y_test, y_pred, average='weighted')
            metrics["f1"] = f1_score(y_test, y_pred, average='weighted')
        except:
            # Fallback for binary classification or if there's an error
            metrics["precision"] = precision_score(y_test, y_pred, average='binary')
            metrics["recall"] = recall_score(y_test, y_pred, average='binary')
            metrics["f1"] = f1_score(y_test, y_pred, average='binary')
        
        # Add more classification metrics
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
        metrics["matthews_corrcoef"] = matthews_corrcoef(y_test, y_pred)
        
        # Try to get probability predictions for ROC AUC
        try:
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                
                # For binary classification
                if len(np.unique(y_test)) == 2:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    # For multi-class, use one-vs-rest approach
                    metrics["roc_auc"] = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                
                # Store probabilities for plotting
                results_dict = {
                    "metrics": metrics,
                    "predictions": y_pred.tolist(),
                    "actual": y_test.tolist(),
                    "probabilities": y_prob.tolist()
                }
                return results_dict
        except:
            # Skip if we can't calculate ROC AUC
            pass
    else:  # regression
        metrics["mse"] = mean_squared_error(y_test, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_test, y_pred)
        metrics["r2"] = r2_score(y_test, y_pred)
        
        # Add more regression metrics
        metrics["explained_variance"] = explained_variance_score(y_test, y_pred)
        metrics["max_error"] = max_error(y_test, y_pred)
        metrics["median_absolute_error"] = median_absolute_error(y_test, y_pred)
    
    return {
        "metrics": metrics,
        "predictions": y_pred.tolist(),
        "actual": y_test.tolist()
    }

def generate_model_evaluation_plots(
    model_results: Dict[str, Any],
    problem_type: str,
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """
    Generate evaluation plots for a model.
    
    Args:
        model_results: Results from train_and_evaluate_model
        problem_type: 'classification' or 'regression'
        feature_names: Names of features (optional)
        
    Returns:
        Dictionary with plotly figures
    """
    plots = {}
    
    # Extract data
    y_test = np.array(model_results["actual"])
    y_pred = np.array(model_results["predictions"])
    
    if problem_type.lower() == "classification":
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Get class labels
        class_labels = np.unique(y_test)
        
        # Calculate percentages for better interpretation
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create a more informative heatmap
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=["Confusion Matrix"]
        )
        
        # Add heatmap
        heatmap = go.Heatmap(
            z=cm,
            x=class_labels,
            y=class_labels,
            colorscale='Viridis',
            showscale=True,
            text=[[f'Count: {cm[i, j]}<br>Percent: {cm_percent[i, j]:.1f}%' for j in range(len(cm[i]))] for i in range(len(cm))],
            hoverinfo='text'
        )
        
        fig.add_trace(heatmap)
        
        # Update layout for better appearance
        fig.update_layout(
            title={
                'text': 'Confusion Matrix',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Predicted Class',
            yaxis_title='Actual Class',
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left', 'autorange': 'reversed'},
            height=400,
            width=None,  # Remove fixed width to make it responsive
            plot_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=80, b=40),
            autosize=True  # Make the plot responsive
        )
        
        # Add text annotations
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                fig.add_annotation(
                    x=class_labels[j], 
                    y=class_labels[i],
                    text=f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)",
                    showarrow=False,
                    font=dict(
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        size=12
                    ),
                    align='center'
                )
                
        plots["confusion_matrix"] = fig
        
        # ROC curve (for binary classification)
        if len(np.unique(y_test)) == 2:
            from sklearn.metrics import roc_curve, auc
            try:
                # Get probability predictions if possible
                if "probabilities" in model_results:
                    y_score = np.array(model_results["probabilities"])[:, 1]
                else:
                    y_score = y_pred
                
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'ROC curve (area = {roc_auc:.2f})',
                    mode='lines'
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name='Random',
                    mode='lines',
                    line=dict(dash='dash')
                ))
                fig.update_layout(
                    title='Receiver Operating Characteristic (ROC) Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                plots["roc_curve"] = fig
                
                # Precision-Recall curve
                from sklearn.metrics import precision_recall_curve, average_precision_score
                precision, recall, _ = precision_recall_curve(y_test, y_score)
                avg_precision = average_precision_score(y_test, y_score)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    name=f'PR curve (AP = {avg_precision:.2f})',
                    mode='lines'
                ))
                fig.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision'
                )
                plots["precision_recall_curve"] = fig
            except:
                # Skip if we can't generate ROC curve
                pass
        
        # For multi-class classification, add class distribution
        if len(np.unique(y_test)) > 2:
            # Class distribution
            class_counts = np.bincount(y_test.astype(int))
            class_labels = np.unique(y_test)
            
            # Create a DataFrame for better visualization
            class_df = pd.DataFrame({
                'Class': class_labels,
                'Count': class_counts,
                'Percentage': (class_counts / len(y_test) * 100).round(1)
            })
            
            # Create a more visually appealing bar chart
            fig = px.bar(
                class_df,
                x='Class', 
                y='Count',
                text=class_df['Percentage'].apply(lambda x: f'{x}%'),
                color='Count',
                color_continuous_scale='Viridis',
                title="Test Set Class Distribution",
                labels={"Class": "Class", "Count": "Frequency"}
            )
            
            # Improve layout
            fig.update_layout(
                plot_bgcolor='white',
                font=dict(size=12),
                margin=dict(l=40, r=40, t=50, b=40),
                coloraxis_showscale=False,
                height=400
            )
            
            # Improve bar appearance
            fig.update_traces(
                textposition='outside',
                textfont=dict(size=12),
                marker=dict(line=dict(width=1, color='#000000'))
            )
            
            plots["class_distribution"] = fig
            
            # Per-class metrics
            try:
                from sklearn.metrics import classification_report
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # Extract per-class metrics
                classes = []
                precision = []
                recall = []
                f1 = []
                
                for class_name, metrics in report.items():
                    if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                        classes.append(class_name)
                        precision.append(metrics['precision'])
                        recall.append(metrics['recall'])
                        f1.append(metrics['f1-score'])
                
                # Create grouped bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(x=classes, y=precision, name='Precision'))
                fig.add_trace(go.Bar(x=classes, y=recall, name='Recall'))
                fig.add_trace(go.Bar(x=classes, y=f1, name='F1 Score'))
                
                fig.update_layout(
                    title='Per-Class Metrics',
                    xaxis_title='Class',
                    yaxis_title='Score',
                    barmode='group'
                )
                plots["per_class_metrics"] = fig
            except:
                # Skip if we can't generate per-class metrics
                pass
    else:  # regression
        # Actual vs Predicted
        fig = px.scatter(
            x=y_test, y=y_pred,
            labels={"x": "Actual", "y": "Predicted"},
            title="Actual vs Predicted Values"
        )
        
        # Add perfect prediction line
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash')
        ))
        
        plots["actual_vs_predicted"] = fig
        
        # Residuals plot
        residuals = y_test - y_pred
        fig = px.scatter(
            x=y_pred, y=residuals,
            labels={"x": "Predicted", "y": "Residuals"},
            title="Residuals Plot"
        )
        fig.add_hline(y=0, line_dash="dash")
        plots["residuals"] = fig
        
        # Residuals distribution
        fig = px.histogram(
            residuals,
            title="Residuals Distribution",
            labels={"value": "Residual", "count": "Frequency"}
        )
        plots["residuals_distribution"] = fig
        
        # Q-Q plot for residuals
        from scipy import stats
        qq = stats.probplot(residuals, dist="norm")
        x = np.array([qq[0][0][0], qq[0][0][-1]])
        y = np.array([qq[0][1][0], qq[0][1][-1]])
        slope, intercept = np.polyfit(qq[0][0], qq[0][1], 1)
        y_line = slope * x + intercept
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=qq[0][0], y=qq[0][1],
            mode='markers',
            name='Residuals'
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_line,
            mode='lines',
            name='Normal Line'
        ))
        fig.update_layout(
            title='Q-Q Plot (Residuals)',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles'
        )
        plots["qq_plot"] = fig
    
    # Metrics summary
    metrics = model_results["metrics"]
    metric_names = list(metrics.keys())
    metric_values = [metrics[name] for name in metric_names]
    
    fig = px.bar(
        x=metric_names, y=metric_values,
        labels={"x": "Metric", "y": "Value"},
        title="Model Performance Metrics"
    )
    plots["metrics_summary"] = fig
    
    return plots

def get_feature_importance(
    model,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Get feature importance from a model if available.
    
    Args:
        model: Trained model
        feature_names: Names of features
        
    Returns:
        Dictionary with feature importance information
    """
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create sorted lists
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        # Create plot
        fig = px.bar(
            x=sorted_features, y=sorted_importances,
            labels={"x": "Feature", "y": "Importance"},
            title="Feature Importance"
        )
        
        return {
            "feature_importance": dict(zip(sorted_features, sorted_importances.tolist())),
            "plot": fig
        }
    
    # Check if model has coef_ attribute (linear models)
    elif hasattr(model, 'coef_'):
        coefs = model.coef_
        
        # Handle multi-class case
        if coefs.ndim > 1:
            # Use absolute mean across classes
            importances = np.abs(coefs).mean(axis=0)
        else:
            importances = np.abs(coefs)
        
        indices = np.argsort(importances)[::-1]
        
        # Create sorted lists
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        # Create plot
        fig = px.bar(
            x=sorted_features, y=sorted_importances,
            labels={"x": "Feature", "y": "Coefficient Magnitude"},
            title="Feature Coefficients"
        )
        
        return {
            "feature_importance": dict(zip(sorted_features, sorted_importances.tolist())),
            "plot": fig
        }
    
    # No feature importance available
    return {
        "feature_importance": {},
        "plot": None
    }

def perform_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    problem_type: str = "classification"
) -> Dict[str, Any]:
    """
    Perform cross-validation for a model.
    
    Args:
        model: Model instance
        X: Features
        y: Targets
        cv: Number of cross-validation folds
        problem_type: 'classification' or 'regression'
        
    Returns:
        Dictionary with cross-validation results
    """
    # Choose scoring metric based on problem type
    if problem_type.lower() == "classification":
        scoring = "accuracy"
    else:
        scoring = "neg_mean_squared_error"
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    # Process results
    if problem_type.lower() == "regression" and scoring == "neg_mean_squared_error":
        # Convert negative MSE to positive RMSE
        cv_scores = np.sqrt(-cv_scores)
        metric_name = "RMSE"
    else:
        metric_name = scoring.capitalize()
    
    # Create plot
    fig = px.box(
        y=cv_scores,
        labels={"y": metric_name},
        title=f"Cross-Validation Results ({cv} Folds)"
    )
    
    return {
        "cv_scores": cv_scores.tolist(),
        "mean_score": cv_scores.mean(),
        "std_score": cv_scores.std(),
        "plot": fig
    }

def get_model_hyperparameters(model_id: str) -> Dict[str, List[Any]]:
    """
    Get hyperparameters for a model that can be tuned.
    
    Args:
        model_id: ID of the model
        
    Returns:
        Dictionary with hyperparameter names and possible values
    """
    # Define hyperparameters for each model type
    hyperparams = {
        "logistic_regression": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        },
        "decision_tree": {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "svm": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto", 0.1, 1]
        },
        "knn": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
        },
        "naive_bayes": {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
        },
        "gradient_boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 0.9, 1.0]
        },
        "xgboost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
        },
        "linear_regression": {
            "fit_intercept": [True, False],
            "normalize": [True, False]
        },
        "svr": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto", 0.1, 1],
            "epsilon": [0.1, 0.2, 0.5]
        }
    }
    
    # Return hyperparameters for the specified model
    return hyperparams.get(model_id, {})

def perform_hyperparameter_tuning(
    model_id: str,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int = 3,
    problem_type: str = "classification"
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for a model.
    
    Args:
        model_id: ID of the model
        X: Features
        y: Targets
        param_grid: Grid of parameters to search
        cv: Number of cross-validation folds
        problem_type: 'classification' or 'regression'
        
    Returns:
        Dictionary with tuning results
    """
    # Get base model
    base_model = get_model_instance(model_id, problem_type)
    
    # Choose scoring metric based on problem type
    if problem_type.lower() == "classification":
        scoring = "accuracy"
    else:
        scoring = "neg_mean_squared_error"
    
    # Perform grid search
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
    )
    grid_search.fit(X, y)
    
    # Get results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Process results for regression
    if problem_type.lower() == "regression" and scoring == "neg_mean_squared_error":
        best_score = np.sqrt(-best_score)
        metric_name = "RMSE"
    else:
        metric_name = scoring.capitalize()
    
    # Get all results
    cv_results = grid_search.cv_results_
    
    # Create a dataframe of results
    results_df = pd.DataFrame({
        "params": [str(p) for p in cv_results["params"]],
        "mean_score": cv_results["mean_test_score"],
        "std_score": cv_results["std_test_score"]
    })
    
    # Sort by score
    if problem_type.lower() == "regression" and scoring == "neg_mean_squared_error":
        # For RMSE, lower is better
        results_df["mean_score"] = np.sqrt(-results_df["mean_score"])
        results_df = results_df.sort_values("mean_score")
    else:
        # For other metrics, higher is better
        results_df = results_df.sort_values("mean_score", ascending=False)
    
    # Create plot of top results
    top_results = results_df.head(10)
    fig = px.bar(
        top_results,
        x="params", y="mean_score",
        error_y="std_score",
        labels={"params": "Parameters", "mean_score": metric_name},
        title=f"Top Hyperparameter Combinations ({metric_name})"
    )
    
    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": results_df.to_dict("records"),
        "plot": fig,
        "best_model": grid_search.best_estimator_
    } 