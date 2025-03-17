import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import time
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
import uuid
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Import custom modules
from ai_eda_pipeline.utils.data_utils import load_dataset, get_dataset_metadata, detect_data_issues, suggest_preprocessing_steps, apply_preprocessing_step
from ai_eda_pipeline.utils.visualization_utils import get_available_plots, generate_plot
from ai_eda_pipeline.components.ai_agents import MainAIAgent
from ai_eda_pipeline.components.preprocessing_workflow import get_preprocessing_steps_by_category, get_step_config_ui, render_workflow, apply_workflow, PREPROCESSING_STEPS
from ai_eda_pipeline.components.preprocessing_agent import PreprocessingMiniAgent
from ai_eda_pipeline.components.ml_agent import MLMiniAgent
from ai_eda_pipeline.components.ml_pipeline import (
    prepare_data_for_ml, get_model_instance, train_and_evaluate_model,
    generate_model_evaluation_plots, get_feature_importance,
    perform_cross_validation, get_model_hyperparameters,
    perform_hyperparameter_tuning, get_available_models,
    CLASSIFICATION_MODELS, REGRESSION_MODELS
)

# Set page configuration
st.set_page_config(
    page_title="Machine Learning Playground, Using AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.df = None
    st.session_state.dataset_name = None
    st.session_state.analysis = None
    st.session_state.available_plots = {}
    st.session_state.dashboard_plots = []
    st.session_state.insights = {}
    st.session_state.preprocessing_steps = []
    st.session_state.ai_agent = MainAIAgent()
    st.session_state.preprocessing_agent = PreprocessingMiniAgent()
    st.session_state.ml_agent = MLMiniAgent()
    st.session_state.reasoning_log = []
    st.session_state.plot_configs = {}
    
    # Preprocessing workflow state
    st.session_state.preprocessing_workflow = []
    st.session_state.processed_df = None
    st.session_state.editing_step = None
    st.session_state.active_tab = "eda"  # Default tab: 'eda' or 'preprocessing' or 'ml'
    st.session_state.preprocessing_messages = []
    st.session_state.preprocessing_suggestions = []
    
    # Preview state
    st.session_state.preview_plot = None
    
    # ML Pipeline state
    st.session_state.target_column = None
    st.session_state.problem_type = None
    st.session_state.selected_features = []
    st.session_state.train_test_split = {"test_size": 0.2, "random_state": 42}
    st.session_state.active_ml_step = 0  # 0: Data Preparation, 1: Model Selection, 2: Training, 3: Insights
    st.session_state.selected_models = []
    st.session_state.trained_models = {}
    st.session_state.best_model = None
    st.session_state.feature_importance = None
    st.session_state.ml_results = {}
    st.session_state.ml_plots = {}
    st.session_state.ml_messages = []
    st.session_state.model_recommendations = None
    st.session_state.model_insights = None
    
    # Phase tracking
    st.session_state.phase_status = {
        "eda": {"completed": False, "progress": 0},
        "preprocessing": {"completed": False, "progress": 0},
        "ml": {"completed": False, "progress": 0}
    }

# Add custom CSS
st.markdown("""
<style>
/* General styling */
.stApp {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Dashboard styling */
.dashboard-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.dashboard-plot {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1.2rem;
    background-color: white;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    position: relative;
    transition: all 0.3s ease;
    overflow: hidden;
}

.dashboard-plot:hover {
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    transform: translateY(-3px);
}

.plot-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #f0f0f0;
}

.plot-title {
    font-weight: 600;
    font-size: 1.2rem;
    color: #2E7D32;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.plot-title::before {
    content: "";
    display: inline-block;
    width: 12px;
    height: 12px;
    background-color: #4CAF50;
    border-radius: 50%;
}

.plot-actions {
    display: flex;
    gap: 0.5rem;
}

.plot-actions button {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 1.2rem;
    padding: 0.2rem;
    border-radius: 4px;
    transition: background-color 0.2s;
}

.plot-actions button:hover {
    background-color: #f0f0f0;
}

/* AI Recommendations styling */
.ai-recommendation {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
    background-color: white;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.ai-recommendation::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(to bottom, #4CAF50, #2196F3);
}

.ai-recommendation:hover {
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    transform: translateY(-3px);
}

.ai-recommendation-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    color: #2E7D32;
}

.ai-recommendation-reason {
    color: #555;
    margin-bottom: 1.2rem;
    font-size: 0.95rem;
    line-height: 1.5;
    padding-left: 0.5rem;
    border-left: 2px solid #f0f0f0;
}

.viz-badge {
    font-size: 0.7rem;
    padding: 0.3rem 0.6rem;
    border-radius: 20px;
    color: white;
    background-color: #4CAF50;
    display: inline-block;
    margin-left: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.histogram { background-color: #4CAF50; }
.boxplot { background-color: #2196F3; }
.scatter { background-color: #9C27B0; }
.bar { background-color: #FF9800; }
.pie { background-color: #E91E63; }
.correlation_heatmap { background-color: #F44336; }
.line { background-color: #00BCD4; }
.pairplot { background-color: #795548; }

/* Available plots styling */
.plot-box-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}

.plot-box {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 0.8rem;
    background-color: white;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    width: 100px;
    height: 100px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    position: relative;
}

.plot-box:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    background-color: #f9f9f9;
}

.plot-box-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.plot-box-label {
    font-size: 0.8rem;
    color: #333;
}

.plot-box-tooltip {
    position: absolute;
    bottom: -40px;
    left: 50%;
    transform: translateX(-50%);
    background-color: rgba(0,0,0,0.8);
    color: white;
    padding: 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    opacity: 0;
    transition: opacity 0.2s;
    pointer-events: none;
    white-space: nowrap;
    z-index: 1000;
}

.plot-box:hover .plot-box-tooltip {
    opacity: 1;
}

/* Preview button styling */
.preview-button {
    position: relative;
    display: inline-block;
    width: 100%;
}

.preview-content {
    display: none;
    position: absolute;
    z-index: 1000;
    background-color: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    padding: 1rem;
    width: 300px;
    right: 0;
    top: 100%;
}

.preview-button:hover .preview-content {
    display: block;
}

/* Hide the actual plotly chart but keep it for the hover content */
[data-testid="stPlotlyChart"] {
    display: none;
}

.preview-button:hover [data-testid="stPlotlyChart"] {
    display: block;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}

/* Insight container styling */
.insight-container {
    background-color: #f8f9fa;
    border-left: 4px solid #4CAF50;
    padding: 1rem;
    margin-top: 1rem;
    border-radius: 0 4px 4px 0;
}

/* Preview container styling */
.preview-container {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    margin-top: 10px;
    background-color: white;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .plot-box {
        width: 80px;
        height: 80px;
    }
    
    .plot-box-icon {
        font-size: 1.5rem;
    }
    
    .plot-box-label {
        font-size: 0.7rem;
    }
}

/* Reasoning log styling */
.reasoning-log {
    padding: 8px 12px;
    margin-bottom: 8px;
    border-radius: 6px;
    font-size: 0.9rem;
    line-height: 1.4;
    border-left: 3px solid #ccc;
}

.reasoning-log-thinking {
    background-color: #f0f7ff;
    border-left-color: #2196F3;
}

.reasoning-log-insight {
    background-color: #f0fff4;
    border-left-color: #4CAF50;
}

.reasoning-log-recommendation {
    background-color: #fff8e1;
    border-left-color: #FFC107;
}

.reasoning-log-action {
    background-color: #f5f5f5;
    border-left-color: #9E9E9E;
}

.reasoning-log-timestamp {
    font-weight: 600;
    color: #555;
    margin-right: 6px;
}

/* Custom tab navigation styling */
div[data-testid="stHorizontalBlock"] {
    background: #f8f9fa;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

/* Style for the tab buttons */
div[data-testid="stHorizontalBlock"] button {
    border-radius: 8px;
    border: none;
    padding: 10px 15px;
    font-weight: 600;
    transition: all 0.3s ease;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

/* Active tab button style */
div[data-testid="stHorizontalBlock"] button[data-active="true"] {
    background-color: #4CAF50;
    color: white;
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    transform: translateY(-2px);
}

/* Inactive tab button style */
div[data-testid="stHorizontalBlock"] button:not([data-active="true"]) {
    background-color: #f0f0f0;
    color: #333;
}

/* Hover effect for inactive tab buttons */
div[data-testid="stHorizontalBlock"] button:not([data-active="true"]):hover {
    background-color: #e0e0e0;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Modern color scheme */
:root {
    --primary: #2E7D32;
    --primary-light: #4CAF50;
    --primary-dark: #1B5E20;
    --accent: #FF9800;
    --accent-light: #FFB74D;
    --text-primary: #212121;
    --text-secondary: #757575;
    --background: #FFFFFF;
    --card-bg: #F5F7F9;
    --border: #E0E0E0;
}

/* General styling */
.main .block-container {
    padding-top: 2rem;
}

/* Card styling */
.stCard {
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    background-color: var(--card-bg);
    border-left: 4px solid var(--primary);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stCard:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.1);
}

/* Button styling */
.stButton > button {
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 600;
    color: var(--primary);
    background-color: var(--card-bg);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: var(--card-bg);
}

/* Recommendation cards */
.recommendation-card {
    border-left: 4px solid var(--primary);
    padding: 1rem;
    margin-bottom: 1rem;
    background-color: var(--card-bg);
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.recommendation-title {
    font-weight: 600;
    color: var(--primary);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.recommendation-content {
    margin-left: 1rem;
}

/* Data issues section */
.data-issue-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
}

.stTabs [data-baseweb="tab"] {
    height: 3rem;
    white-space: pre-wrap;
    border-radius: 6px 6px 0 0;
    padding: 0 1rem;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary-light) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Add custom CSS for a more modern and interesting UI
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary: #2E7D32;
        --primary-light: #4CAF50;
        --primary-dark: #1B5E20;
        --accent: #FF9800;
        --accent-light: #FFB74D;
        --text-primary: #212121;
        --text-secondary: #757575;
        --background: #FFFFFF;
        --card-bg: #F5F7F9;
        --border: #E0E0E0;
    }
    
    /* General styling */
    .main .block-container {
        padding-top: 1rem;
        max-width: 100%;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Make the sidebar narrower */
    [data-testid="stSidebar"] {
        min-width: 250px !important;
        max-width: 300px !important;
    }
    
    /* Make the main content area wider */
    .main {
        width: calc(100% - 300px) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-dark);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Make plots stand out with a card-like appearance and ensure they're wide */
    [data-testid="stPlotlyChart"] {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border);
        width: 100% !important;
    }
    
    /* Ensure plots are responsive */
    [data-testid="stPlotlyChart"] > div {
        width: 100% !important;
    }
    
    /* Improve dataframe appearance */
    [data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border);
        width: 100% !important;
    }
    
    /* Card styling */
    .stCard {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: var(--card-bg);
        border-left: 4px solid var(--primary);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        width: 100% !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary);
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: var(--primary-dark);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    
    /* Tabs styling - make them more visible */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f2f6;
        padding: 0.25rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-light) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid var(--border);
    }
    
    /* Metrics styling - make them more compact and fit better */
    [data-testid="stMetric"] {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 0.5rem;
        border: 1px solid var(--border);
        margin-bottom: 0.5rem;
    }
    [data-testid="stMetric"] > div:first-child {
        color: var(--primary);
        font-size: 0.8rem;
    }
    [data-testid="stMetric"] > div:nth-child(2) {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Make columns fit better */
    [data-testid="column"] {
        padding: 0.25rem !important;
    }
    
    /* Improve selectbox and multiselect */
    [data-testid="stSelectbox"], [data-testid="stMultiSelect"] {
        background-color: white;
        border-radius: 6px;
        border: 1px solid var(--border);
        padding: 0.25rem;
        margin-bottom: 0.5rem;
    }
    
    /* Improve expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--primary);
        background-color: #f0f2f6;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    
    /* Custom header styling */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        color: var(--primary);
        border-bottom: 2px solid var(--primary-light);
        padding-bottom: 0.25rem;
    }
    
    /* Success/Info/Warning/Error message styling */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
    }
    
    /* Make the ML Pipeline section wider */
    .element-container:has([data-testid="stVerticalBlock"]) {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Ensure plots in the ML Pipeline section are wide */
    .element-container:has([data-testid="stVerticalBlock"]) [data-testid="stPlotlyChart"] {
        width: 100% !important;
    }
    
    /* Fix for confusion matrix width */
    .js-plotly-plot, .plot-container {
        width: 100% !important;
    }
    
    /* Ensure all three sections fit well */
    .row-widget.stRadio > div {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
    }
    
    /* Make the blue info boxes more visible */
    .stInfo {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to add to reasoning log
def add_to_log(message, is_thinking=False, is_insight=False, is_recommendation=False):
    """Add a message to the reasoning log."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry_type = "thinking" if is_thinking else "insight" if is_insight else "recommendation" if is_recommendation else "action"
    
    st.session_state.reasoning_log.append({
        "timestamp": timestamp,
        "message": message,
        "type": entry_type
    })

# Function to add preprocessing message
def add_preprocessing_message(message, is_error=False):
    """Add a message to the preprocessing messages log."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    st.session_state.preprocessing_messages.append({
        "timestamp": timestamp,
        "message": message,
        "is_error": is_error
    })

def add_ml_message(message, is_error=False, is_recommendation=False, is_insight=False):
    """Add a message to the ML messages log."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    message_type = "error" if is_error else "recommendation" if is_recommendation else "insight" if is_insight else "info"
    
    st.session_state.ml_messages.append({
        "timestamp": timestamp,
        "message": message,
        "type": message_type
    })
    
    # Also add to the main log for consistency
    if is_error:
        add_to_log(f"ML Pipeline Error: {message}", is_thinking=False)
    elif is_recommendation:
        add_to_log(f"ML Recommendation: {message}", is_recommendation=True)
    elif is_insight:
        add_to_log(f"ML Insight: {message}", is_insight=True)
    else:
        add_to_log(f"ML Pipeline: {message}", is_thinking=True)

# Function to add a preprocessing step to workflow
def add_preprocessing_step(step_config):
    if not step_config:
        return
    
    if "editing_step" in st.session_state and st.session_state.editing_step is not None:
        # Update existing step
        idx = st.session_state.editing_step
        if idx < len(st.session_state.preprocessing_workflow):
            st.session_state.preprocessing_workflow[idx] = step_config
            add_preprocessing_message(f"Updated {step_config.get('step')} step in workflow")
        st.session_state.editing_step = None
    else:
        # Add new step
        st.session_state.preprocessing_workflow.append(step_config)
        add_preprocessing_message(f"Added {step_config.get('step')} step to workflow")

# Function to apply preprocessing workflow
def apply_preprocessing_workflow():
    if not st.session_state.preprocessing_workflow:
        add_preprocessing_message("No preprocessing steps to apply", is_error=True)
        return False
    
    try:
        # Apply workflow to original dataframe
        processed_df, messages = apply_workflow(st.session_state.df, st.session_state.preprocessing_workflow)
        
        # Store processed dataframe
        st.session_state.processed_df = processed_df
        
        # Add messages to log
        for msg in messages:
            add_preprocessing_message(msg)
        
        # Add a success message
        add_preprocessing_message("Preprocessing workflow applied successfully!")
        
        return True
    except Exception as e:
        add_preprocessing_message(f"Error applying preprocessing workflow: {str(e)}", is_error=True)
        return False

# Function to evaluate the preprocessing workflow
def evaluate_preprocessing_workflow():
    """Evaluate the current preprocessing workflow using the AI agent."""
    if not st.session_state.preprocessing_workflow:
        add_preprocessing_message("No preprocessing steps to evaluate", is_error=True)
        return None, None
    
    try:
        # Get evaluation from the agent
        score, feedback = st.session_state.preprocessing_agent.evaluate_preprocessing_pipeline(
            st.session_state.preprocessing_workflow
        )
        
        # Add message to log
        add_preprocessing_message(f"Evaluated preprocessing workflow (Score: {int(score * 10)}/10)")
        
        return score, feedback
    except Exception as e:
        add_preprocessing_message(f"Error evaluating preprocessing workflow: {str(e)}", is_error=True)
        return None, None

# Function to load and analyze dataset
def load_and_analyze_dataset(file_buffer=None, dataset_name=None):
    try:
        # Load dataset
        add_to_log("Starting dataset loading process", is_thinking=True)
        df, name = load_dataset(file_buffer=file_buffer, dataset_name=dataset_name)
        
        # Store in session state
        st.session_state.df = df
        st.session_state.dataset_name = name
        
        # Analyze dataset using AI agent
        add_to_log(f"Analyzing dataset: {name}")
        add_to_log(f"Examining data structure: {df.shape[0]} rows, {df.shape[1]} columns", is_thinking=True)
        add_to_log(f"Checking for missing values and data types", is_thinking=True)
        
        # Log column information
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        add_to_log(f"Found {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns", is_thinking=True)
        
        # Analyze dataset
        analysis = st.session_state.ai_agent.analyze_dataset(df, name)
        st.session_state.analysis = analysis
        
        # Log analysis results
        if "issues" in analysis:
            issues = analysis["issues"]
            if issues["missing_values"]:
                add_to_log(f"Detected missing values in {len(issues['missing_values'])} columns", is_thinking=True)
            if issues["outliers"]:
                add_to_log(f"Detected outliers in {len(issues['outliers'])} columns", is_thinking=True)
            if issues["high_correlation"]:
                add_to_log(f"Found {len(issues['high_correlation'])} highly correlated column pairs", is_thinking=True)
        
        # Get available plots
        add_to_log("Determining suitable visualization types for this dataset", is_thinking=True)
        st.session_state.available_plots = get_available_plots(df)
        
        # Get AI-recommended visualizations
        add_to_log("Generating AI-recommended visualizations based on data patterns", is_thinking=True)
        st.session_state.recommended_visualizations = st.session_state.ai_agent.recommend_visualizations(df)
        
        # Log recommendations
        for i, viz in enumerate(st.session_state.recommended_visualizations[:3]):  # Log first 3 recommendations
            add_to_log(f"Recommending {viz['type']} visualization: {viz['reason'][:100]}...", is_recommendation=True)
        
        # Initialize plot configs
        st.session_state.plot_configs = {}
        
        # Clear dashboard plots and insights
        st.session_state.dashboard_plots = []
        st.session_state.insights = {}
        
        add_to_log(f"Dataset loaded and analyzed successfully: {name} ({df.shape[0]} rows, {df.shape[1]} columns)")
        return True
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        add_to_log(f"Error loading dataset: {str(e)}")
        return False

# Function to generate a plot and add it to dashboard
def add_plot_to_dashboard(plot_type, config):
    if len(st.session_state.dashboard_plots) >= 6:
        st.warning("Maximum of 6 plots allowed on dashboard. Remove a plot to add a new one.")
        add_to_log("Failed to add plot: Maximum limit reached")
        return False
    
    add_to_log(f"Preparing to add {plot_type} plot to dashboard", is_thinking=True)
    
    # Log configuration details
    config_details = []
    for key, value in config.items():
        if isinstance(value, list) and len(value) > 3:
            value_str = f"{', '.join(str(v) for v in value[:3])}... ({len(value)} items)"
        else:
            value_str = str(value)
        config_details.append(f"{key}: {value_str}")
    
    add_to_log(f"Plot configuration: {'; '.join(config_details)}", is_thinking=True)
    
    # Store the plot configuration directly in the dashboard_plots list
    st.session_state.dashboard_plots.append({
        "type": plot_type,
        "config": config
    })
    
    add_to_log(f"Added {plot_type} plot to dashboard")
    return True

# Function to remove plot from dashboard
def remove_plot_from_dashboard(index):
    if 0 <= index < len(st.session_state.dashboard_plots):
        removed_plot = st.session_state.dashboard_plots.pop(index)
        add_to_log(f"Removed {removed_plot['type']} plot from dashboard")
        return True
    
    return False

# Function to get AI insight for a plot
def get_ai_insight(plot_id):
    if plot_id not in st.session_state.plot_configs:
        return "Error: Plot not found"
    
    plot_config = st.session_state.plot_configs[plot_id]
    add_to_log(f"Generating {plot_config['type']} plot for insight analysis", is_thinking=True)
    plot_data = generate_plot(st.session_state.df, plot_config["type"], plot_config["config"])
    
    add_to_log(f"Requesting AI insight for {plot_config['type']} plot")
    
    # Add thinking steps for different plot types
    if plot_config['type'] == 'histogram':
        add_to_log("Analyzing distribution shape, skewness, and potential outliers", is_thinking=True)
    elif plot_config['type'] == 'scatter':
        add_to_log("Examining correlation patterns, clusters, and potential relationships", is_thinking=True)
    elif plot_config['type'] == 'boxplot':
        add_to_log("Identifying quartiles, median values, and outlier presence", is_thinking=True)
    elif plot_config['type'] == 'correlation_heatmap':
        add_to_log("Detecting strong positive and negative correlations between variables", is_thinking=True)
    else:
        add_to_log(f"Analyzing patterns and trends in the {plot_config['type']} visualization", is_thinking=True)
    
    # Get insight from EDA agent
    insight = st.session_state.ai_agent.get_eda_agent().generate_insight(
        st.session_state.df,
        plot_data,
        st.session_state.dataset_name
    )
    
    # Store insight
    st.session_state.insights[plot_id] = insight
    
    # Log a summary of the insight
    insight_summary = insight.split('.')[0] + '.' if '.' in insight else insight
    add_to_log(f"Key insight: {insight_summary}", is_insight=True)
    
    add_to_log(f"Generated AI insight for {plot_config['type']} plot")
    return insight

def get_plot_config(plot_type, df):
    """
    Generate configuration for a plot type based on the dataframe.
    
    Args:
        plot_type (str): The type of plot to configure
        df (pd.DataFrame): The dataframe to analyze
        
    Returns:
        dict: Configuration for the plot
    """
    # Get column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Default configurations based on plot type
    if plot_type == "histogram":
        # Choose the first numeric column
        if numeric_cols:
            return {"column": numeric_cols[0]}
        return {"column": df.columns[0]}  # Fallback
        
    elif plot_type == "boxplot":
        # Choose the first numeric column
        if numeric_cols:
            return {"column": numeric_cols[0]}
        return {"column": df.columns[0]}  # Fallback
        
    elif plot_type == "scatter":
        # Choose the first two numeric columns
        if len(numeric_cols) >= 2:
            return {"x_column": numeric_cols[0], "y_column": numeric_cols[1]}
        elif len(numeric_cols) == 1:
            return {"x_column": numeric_cols[0], "y_column": numeric_cols[0]}
        return {"x_column": df.columns[0], "y_column": df.columns[0]}  # Fallback
        
    elif plot_type == "bar":
        # Choose the first categorical column and first numeric column
        if categorical_cols and numeric_cols:
            return {"x_column": categorical_cols[0], "y_column": numeric_cols[0]}
        elif categorical_cols:
            return {"column": categorical_cols[0]}
        return {"column": df.columns[0]}  # Fallback
        
    elif plot_type == "pie":
        # Choose the first categorical column
        if categorical_cols:
            return {"column": categorical_cols[0]}
        return {"column": df.columns[0]}  # Fallback
        
    elif plot_type == "correlation_heatmap":
        # Use all numeric columns
        if numeric_cols:
            return {"columns": numeric_cols, "method": "pearson"}
        return {"columns": df.columns.tolist(), "method": "pearson"}  # Fallback
        
    elif plot_type == "line":
        # Choose the first two numeric columns
        if len(numeric_cols) >= 2:
            return {"x_column": numeric_cols[0], "y_column": numeric_cols[1]}
        elif len(numeric_cols) == 1:
            return {"x_column": numeric_cols[0], "y_column": numeric_cols[0]}
        return {"x_column": df.columns[0], "y_column": df.columns[0]}  # Fallback
        
    elif plot_type == "pairplot":
        # Choose up to 4 numeric columns
        if numeric_cols:
            selected_cols = numeric_cols[:min(4, len(numeric_cols))]
            hue = categorical_cols[0] if categorical_cols and len(df[categorical_cols[0]].unique()) <= 5 else None
            return {"columns": selected_cols, "hue": hue}
        return {"columns": df.columns.tolist()[:min(4, len(df.columns))], "hue": None}  # Fallback
    
    # Default empty config
    return {}

# Function to render the ML Pipeline tab
def render_ml_pipeline_tab():
    st.markdown("<h2 style='font-size: 1.6rem; font-weight: 600; margin-top: 0.5rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Supervised Machine Learning Pipeline</h2>", unsafe_allow_html=True)
    
    # Check if we have data to work with
    if st.session_state.df is None:
        st.info("Please load a dataset first to use the ML Pipeline.")
        return
    
    # Use processed data if available, otherwise use original data
    df_to_use = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
    
    # Display phase progress in a more compact layout
    progress_cols = st.columns(3)
    with progress_cols[0]:
        eda_status = "✅" if st.session_state.phase_status["eda"]["completed"] else "🔄"
        st.markdown(f"**EDA Phase: {eda_status}**")
    with progress_cols[1]:
        preproc_status = "✅" if st.session_state.phase_status["preprocessing"]["completed"] else "🔄"
        st.markdown(f"**Preprocessing Phase: {preproc_status}**")
    with progress_cols[2]:
        ml_status = "✅" if st.session_state.phase_status["ml"]["completed"] else "🔄"
        st.markdown(f"**ML Phase: {ml_status}**")
    
    # Target Selection (moved to the top level)
    st.markdown("<div style='background-color: #E8F5E9; padding: 1rem; border-radius: 10px; border-left: 4px solid #4CAF50; margin-bottom: 1rem;'><h3 style='margin-top: 0; font-size: 1.2rem; color: #2E7D32;'>Target Selection</h3><p style='margin-bottom: 0;'>Select the target variable you want to predict. This defines your supervised learning problem.</p></div>", unsafe_allow_html=True)
    
    # Create a more compact layout for target selection
    target_cols = st.columns([2, 1])
    
    with target_cols[0]:
        numeric_cols = df_to_use.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df_to_use.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df_to_use.columns.tolist()
        
        # Select target column
        target_col = st.selectbox(
            "Select Target Column (what you want to predict):",
            all_cols,
            index=0 if st.session_state.target_column is None else all_cols.index(st.session_state.target_column) if st.session_state.target_column in all_cols else 0,
            key="target_column_select"
        )
        
        # Store the selected target column
        st.session_state.target_column = target_col
    
    with target_cols[1]:
        # Determine problem type
        if target_col in categorical_cols or df_to_use[target_col].nunique() < 10:
            problem_type = "classification"
            unique_values = df_to_use[target_col].nunique()
            if unique_values == 2:
                st.success(f"Binary Classification ({unique_values} categories)")
            else:
                st.success(f"Multi-class Classification ({unique_values} categories)")
            add_ml_message(f"Detected classification problem for target '{target_col}' with {unique_values} classes")
        else:
            problem_type = "regression"
            st.success(f"Regression (numeric prediction)")
            add_ml_message(f"Detected regression problem for target '{target_col}'")
        
        # Store problem type
        st.session_state.problem_type = problem_type
    
    # Display target distribution
    st.markdown("### Target Variable Distribution")
    
    if problem_type == "classification":
        # For classification, create a more informative bar chart
        value_counts = df_to_use[target_col].value_counts().reset_index()
        value_counts.columns = ['Category', 'Count']
        
        # Calculate percentages
        total = value_counts['Count'].sum()
        value_counts['Percentage'] = (value_counts['Count'] / total * 100).round(1)
        
        # Create a more visually appealing bar chart
        fig = px.bar(
            value_counts, 
            x='Category', 
            y='Count',
            text=value_counts['Percentage'].apply(lambda x: f'{x}%'),
            color='Count',
            color_continuous_scale='Viridis',
            title=f"Distribution of {target_col} (Classification Target)",
            labels={'Category': target_col, 'Count': 'Frequency'}
        )
        
        # Improve layout
        fig.update_layout(
            plot_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=50, b=40),
            coloraxis_showscale=False,
            height=350,
            autosize=True
        )
        
        # Improve bar appearance
        fig.update_traces(
            textposition='outside',
            textfont=dict(size=12),
            marker=dict(line=dict(width=1, color='#000000'))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add descriptive statistics in a more compact format
        with st.expander("Class Distribution Statistics"):
            st.dataframe(value_counts, hide_index=True)
        
    else:  # Regression
        # For regression, create a more informative histogram
        fig = px.histogram(
            df_to_use, 
            x=target_col,
            nbins=30,
            marginal="box",
            title=f"Distribution of {target_col} (Regression Target)",
            color_discrete_sequence=['#4CAF50'],
            opacity=0.7
        )
        
        # Add a KDE curve
        from scipy import stats
        kde_x = np.linspace(df_to_use[target_col].min(), df_to_use[target_col].max(), 1000)
        kde = stats.gaussian_kde(df_to_use[target_col].dropna())
        kde_y = kde(kde_x)
        
        # Scale the KDE curve to match histogram height
        hist_values, _ = np.histogram(df_to_use[target_col].dropna(), bins=30)
        scale_factor = max(hist_values) / max(kde_y)
        kde_y = kde_y * scale_factor
        
        fig.add_trace(
            go.Scatter(
                x=kde_x, 
                y=kde_y,
                mode='lines',
                name='Density',
                line=dict(color='#FF9800', width=3)
            )
        )
        
        # Improve layout
        fig.update_layout(
            plot_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=50, b=40),
            height=350,
            autosize=True,
            xaxis_title=target_col,
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add descriptive statistics in a more compact format
        with st.expander("Numerical Statistics"):
            stats = df_to_use[target_col].describe().reset_index()
            stats.columns = ['Statistic', 'Value']
            st.dataframe(stats, hide_index=True)
    
    # Remove target column from potential features
    potential_features = [col for col in all_cols if col != target_col]
    
    # Update selected_features in session state to ensure it only contains valid features
    if "selected_features" in st.session_state:
        # Filter out any features that are no longer valid (including the target column)
        st.session_state.selected_features = [f for f in st.session_state.selected_features if f in potential_features]
    else:
        # Initialize with all potential features
        st.session_state.selected_features = potential_features.copy()
    
    # ML Pipeline Steps
    ml_steps = st.tabs(["1️⃣ Data Preparation", "2️⃣ Model Selection", "3️⃣ Training & Evaluation", "4️⃣ Model Insights"])
    
    # Step 1: Data Preparation
    with ml_steps[0]:
        st.markdown("### Data Preparation")
        
        # Show preprocessing summary if available
        if st.session_state.preprocessing_workflow:
            with st.expander("Preprocessing Steps Applied", expanded=False):
                st.markdown("The following preprocessing steps have been applied to the dataset:")
                for i, step in enumerate(st.session_state.preprocessing_workflow):
                    step_type = step.get("step", "unknown")
                    column = step.get("column", "multiple columns")
                    method = step.get("method", "")
                    
                    if step_type == "impute_missing":
                        st.markdown(f"**{i+1}.** Missing value imputation on '{column}' using {method}")
                    elif step_type == "drop_column":
                        st.markdown(f"**{i+1}.** Dropped column '{column}'")
                    elif step_type == "drop_duplicates":
                        st.markdown(f"**{i+1}.** Removed duplicate rows")
                    elif step_type == "handle_outliers":
                        st.markdown(f"**{i+1}.** Outlier handling on '{column}' using {method}")
                    elif step_type == "scale_features":
                        st.markdown(f"**{i+1}.** Feature scaling on '{column}' using {method}")
                    elif step_type == "encode_categorical":
                        st.markdown(f"**{i+1}.** Encoded '{column}' using {method}")
                    elif step_type == "feature_engineering":
                        st.markdown(f"**{i+1}.** Feature engineering: {method} on '{column}'")
                    elif step_type == "dimensionality_reduction":
                        st.markdown(f"**{i+1}.** Dimensionality reduction using {method}")
                    elif step_type == "handle_imbalance":
                        st.markdown(f"**{i+1}.** Handled class imbalance using {method}")
                    else:
                        st.markdown(f"**{i+1}.** {step_type} on '{column}'")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Train-test split options
            st.markdown("#### Train-Test Split Configuration")
            test_size = st.slider(
                "Test Set Size (%):",
                min_value=10,
                max_value=40,
                value=int(st.session_state.train_test_split["test_size"] * 100),
                step=5,
                key="test_size_slider"
            )
            
            # Update train-test split settings
            st.session_state.train_test_split["test_size"] = test_size / 100
            
            # Random seed for reproducibility
            random_state = st.number_input(
                "Random Seed:",
                min_value=1,
                max_value=1000,
                value=st.session_state.train_test_split["random_state"],
                key="random_state_input"
            )
            
            # Update random state
            st.session_state.train_test_split["random_state"] = random_state
            
            # Add stratification option for classification
            if problem_type == "classification":
                stratify = st.checkbox("Use stratified sampling (maintains class distribution)", value=True)
                st.session_state.train_test_split["stratify"] = stratify
        
        with col2:
            # Feature selection
            st.markdown("#### Feature Selection")
            
            # Select features to use - ensure default values are in options
            selected_features = st.multiselect(
                "Select Features to Use:",
                options=potential_features,
                default=st.session_state.selected_features,
                key="feature_select"
            )
            
            # Store selected features
            st.session_state.selected_features = selected_features
            
            # Select all / Deselect all buttons - use horizontal layout without nested columns
            select_all = st.button("Select All Features", key="select_all_btn")
            if select_all:
                st.session_state.selected_features = potential_features.copy()
                st.experimental_rerun()
                
            deselect_all = st.button("Deselect All Features", key="deselect_all_btn")
            if deselect_all:
                st.session_state.selected_features = []
                st.experimental_rerun()
        
        # Show data preview with selected features and target
        if selected_features:
            st.markdown("### Data Preview (Selected Features and Target)")
            preview_df = df_to_use[selected_features + [target_col]].head(5)
            st.dataframe(preview_df, use_container_width=True)
            
            # Get AI recommendations for models
            if st.button("Get AI Model Recommendations", key="get_model_recommendations_btn"):
                with st.spinner("Analyzing data and generating model recommendations..."):
                    # Get model recommendations from ML Mini Agent
                    recommendations = st.session_state.ml_agent.get_model_recommendations(
                        df=df_to_use,
                        target_column=target_col,
                        problem_type=problem_type,
                        eda_insights=st.session_state.analysis if "analysis" in st.session_state else {},
                        preprocessing_steps=st.session_state.preprocessing_workflow
                    )
                    
                    # Store recommendations
                    st.session_state.model_recommendations = recommendations
                    
                    # Add to log
                    add_ml_message("Generated model recommendations based on data analysis", is_insight=True)
                    
                    # Mark EDA phase as completed
                    st.session_state.phase_status["eda"]["completed"] = True
                    st.session_state.phase_status["preprocessing"]["completed"] = True
                    
                    # Rerun to show recommendations
                    st.experimental_rerun()
            
            # Display model recommendations if available
            if "model_recommendations" in st.session_state and st.session_state.model_recommendations:
                recommendations = st.session_state.model_recommendations
                
                if "error" in recommendations:
                    st.error(f"Error generating recommendations: {recommendations['error']}")
                else:
                    with st.expander("AI Model Recommendations", expanded=True):
                        # General advice
                        if "general_advice" in recommendations:
                            st.markdown(f"**General Advice:** {recommendations['general_advice']}")
                        
                        # Recommended models
                        if "recommended_models" in recommendations:
                            st.markdown("#### Recommended Models")
                            for model in recommendations["recommended_models"]:
                                model_id = model.get("model_id", "")
                                model_name = model.get("name", "")
                                explanation = model.get("explanation", "")
                                
                                st.markdown(f"**{model_name}**")
                                st.markdown(f"{explanation}")
                                
                                # Strengths and limitations
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Strengths:**")
                                    for strength in model.get("strengths", []):
                                        st.markdown(f"- {strength}")
                                
                                with col2:
                                    st.markdown("**Limitations:**")
                                    for limitation in model.get("limitations", []):
                                        st.markdown(f"- {limitation}")
                                
                                st.markdown("---")
                        
                        # Additional preprocessing suggestions
                        if "additional_preprocessing" in recommendations and recommendations["additional_preprocessing"]:
                            st.markdown("#### Additional Preprocessing Suggestions")
                            for suggestion in recommendations["additional_preprocessing"]:
                                st.markdown(f"- {suggestion}")
                        
                        # Potential challenges
                        if "potential_challenges" in recommendations and recommendations["potential_challenges"]:
                            st.markdown("#### Potential Challenges")
                            for challenge in recommendations["potential_challenges"]:
                                st.markdown(f"- {challenge}")
            
            # Proceed button
            if st.button("Proceed to Model Selection", key="proceed_to_model_btn"):
                # Store all necessary data in session state
                st.session_state.target_column = target_col
                st.session_state.problem_type = problem_type
                st.session_state.selected_features = selected_features
                st.session_state.train_test_split = {
                    "test_size": test_size / 100,
                    "random_state": random_state,
                    "stratify": st.session_state.train_test_split.get("stratify", True) if problem_type == "classification" else False
                }
                
                # Set active step to model selection
                st.session_state.active_ml_step = 1
                
                # Log the action
                add_ml_message(f"Data preparation completed with {len(selected_features)} features and target '{target_col}'")
                
                # Show success message and rerun
                st.success("Data preparation complete! Proceeding to Model Selection...")
                st.experimental_rerun()
        else:
            st.warning("Please select at least one feature to proceed.")
    
    # Step 2: Model Selection
    with ml_steps[1]:
        # Debug info
        st.markdown(f"<div style='background-color: #E3F2FD; padding: 0.5rem; border-radius: 5px; margin-bottom: 1rem; font-size: 0.8rem;'>Current step: {st.session_state.active_ml_step}, Problem type: {problem_type}</div>", unsafe_allow_html=True)
        
        # Check if we have necessary data
        if not st.session_state.selected_features:
            st.warning("Please complete the Data Preparation step first and select features.")
        else:
            st.markdown("<div style='background-color: #E3F2FD; padding: 1rem; border-radius: 10px; border-left: 4px solid #2196F3; margin-bottom: 1rem;'><h3 style='margin-top: 0; font-size: 1.2rem; color: #0D47A1;'>Model Selection</h3><p style='margin-bottom: 0;'>Select the models you want to train and evaluate. You can choose multiple models to compare their performance.</p></div>", unsafe_allow_html=True)
            
            # Import the get_available_models function from ml_pipeline
            from ai_eda_pipeline.components.ml_pipeline import get_available_models
            
            # Get available models based on problem type
            available_models = get_available_models(problem_type)
            
            # Create two columns for model selection
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("#### Available Models")
                
                # Initialize selected_models in session state if not present
                if "selected_models" not in st.session_state:
                    st.session_state.selected_models = []
                
                # Create checkboxes for model selection
                selected_models = []
                
                # Define complexity levels for each model type
                complexity_levels = {
                    "logistic_regression": "Low",
                    "linear_regression": "Low",
                    "naive_bayes": "Low",
                    "decision_tree": "Medium",
                    "svm": "Medium",
                    "svr": "Medium",
                    "knn": "Medium",
                    "random_forest": "High",
                    "gradient_boosting": "High",
                    "xgboost": "High"
                }
                
                # Define model compatibility based on dataset characteristics
                model_compatibility = {}
                
                # Check dataset characteristics to determine model compatibility
                num_samples = len(df_to_use)
                num_features = len(st.session_state.selected_features)
                has_missing_values = df_to_use[st.session_state.selected_features].isnull().any().any()
                
                # Classification-specific checks
                if problem_type == "classification":
                    # Check class distribution
                    class_counts = df_to_use[target_col].value_counts()
                    min_class_count = class_counts.min()
                    num_classes = len(class_counts)
                    
                    # Logistic Regression
                    if num_classes > 2:
                        model_compatibility["logistic_regression"] = {
                            "suitable": True,
                            "warning": None
                        }
                    else:
                        model_compatibility["logistic_regression"] = {
                            "suitable": True,
                            "warning": None
                        }
                    
                    # SVM
                    if num_samples > 10000:
                        model_compatibility["svm"] = {
                            "suitable": False,
                            "warning": "SVM may be slow for large datasets"
                        }
                    else:
                        model_compatibility["svm"] = {
                            "suitable": True,
                            "warning": None
                        }
                    
                    # Naive Bayes
                    model_compatibility["naive_bayes"] = {
                        "suitable": True,
                        "warning": "Works best with independent features" if num_features > 10 else None
                    }
                    
                    # KNN
                    if num_samples > 10000:
                        model_compatibility["knn"] = {
                            "suitable": False,
                            "warning": "KNN may be slow for large datasets"
                        }
                    else:
                        model_compatibility["knn"] = {
                            "suitable": True,
                            "warning": None
                        }
                    
                    # Decision Tree
                    model_compatibility["decision_tree"] = {
                        "suitable": True,
                        "warning": "Prone to overfitting" if num_samples < 100 else None
                    }
                    
                    # Random Forest
                    model_compatibility["random_forest"] = {
                        "suitable": True,
                        "warning": "May be slow to train" if num_samples > 100000 else None
                    }
                    
                    # Gradient Boosting
                    model_compatibility["gradient_boosting"] = {
                        "suitable": True,
                        "warning": "May be slow to train" if num_samples > 50000 else None
                    }
                    
                    # XGBoost
                    model_compatibility["xgboost"] = {
                        "suitable": True,
                        "warning": "May be slow to train" if num_samples > 100000 else None
                    }
                
                # Regression-specific checks
                else:
                    # Linear Regression
                    model_compatibility["linear_regression"] = {
                        "suitable": True,
                        "warning": "Assumes linear relationship" if num_features > 10 else None
                    }
                    
                    # SVR
                    if num_samples > 10000:
                        model_compatibility["svr"] = {
                            "suitable": False,
                            "warning": "SVR may be slow for large datasets"
                        }
                    else:
                        model_compatibility["svr"] = {
                            "suitable": True,
                            "warning": None
                        }
                    
                    # KNN
                    if num_samples > 10000:
                        model_compatibility["knn"] = {
                            "suitable": False,
                            "warning": "KNN may be slow for large datasets"
                        }
                    else:
                        model_compatibility["knn"] = {
                            "suitable": True,
                            "warning": None
                        }
                    
                    # Decision Tree
                    model_compatibility["decision_tree"] = {
                        "suitable": True,
                        "warning": "Prone to overfitting" if num_samples < 100 else None
                    }
                    
                    # Random Forest
                    model_compatibility["random_forest"] = {
                        "suitable": True,
                        "warning": "May be slow to train" if num_samples > 100000 else None
                    }
                    
                    # Gradient Boosting
                    model_compatibility["gradient_boosting"] = {
                        "suitable": True,
                        "warning": "May be slow to train" if num_samples > 50000 else None
                    }
                    
                    # XGBoost
                    model_compatibility["xgboost"] = {
                        "suitable": True,
                        "warning": "May be slow to train" if num_samples > 100000 else None
                    }
                
                # Display models with compatibility information
                for model_id, model_name in available_models.items():
                    is_selected = model_id in st.session_state.selected_models
                    complexity = complexity_levels.get(model_id, "Medium")
                    
                    # Get compatibility info
                    compatibility = model_compatibility.get(model_id, {"suitable": True, "warning": None})
                    
                    # Add warning icon for unsuitable models
                    prefix = "🔴 " if not compatibility["suitable"] else "🟢 " if not compatibility["warning"] else "🟡 "
                    
                    # Create checkbox with appropriate styling
                    checkbox_label = f"{prefix}{model_name} ({complexity} complexity)"
                    if compatibility["warning"]:
                        checkbox_label += f" - {compatibility['warning']}"
                        
                    if st.checkbox(
                        checkbox_label,
                        value=is_selected,
                        key=f"model_{model_id}",
                        disabled=not compatibility["suitable"]
                    ):
                        selected_models.append(model_id)
                
                # Store selected models
                st.session_state.selected_models = selected_models
            
            with col2:
                # Display AI recommendations if available
                if "model_recommendations" in st.session_state and st.session_state.model_recommendations:
                    st.markdown("#### AI Recommendations")
                    
                    recommendations = st.session_state.model_recommendations
                    if "recommended_models" in recommendations:
                        for model in recommendations["recommended_models"]:
                            model_id = model.get("model_id", "")
                            model_name = model.get("name", "")
                            
                            # Create an expander for each recommended model
                            with st.expander(f"✨ {model_name}", expanded=False):
                                st.markdown(f"{model.get('explanation', '')}")
                                
                                # Quick select button
                                if st.button(f"Select {model_name}", key=f"select_{model_id}"):
                                    if model_id not in st.session_state.selected_models and model_id in available_models:
                                        st.session_state.selected_models.append(model_id)
                                        st.experimental_rerun()
                else:
                    st.info("Complete the Data Preparation step to get AI model recommendations.")
            
            # Display warning if no models selected
            if not selected_models:
                st.warning("Please select at least one model to train.")
            else:
                st.success(f"Selected {len(selected_models)} models for training.")
            
            # Proceed button
            if st.button("Train Selected Models", key="train_models_btn"):
                if not selected_models:
                    st.error("Please select at least one model to train.")
                else:
                    # Set active step to training & evaluation
                    st.session_state.active_ml_step = 2
                    
                    # Log the action
                    add_ml_message(f"Selected {len(selected_models)} models for training: {', '.join([available_models[m] for m in selected_models])}")
                    
                    # Show success message and rerun
                    st.success("Models selected! Proceeding to Training & Evaluation...")
                    st.experimental_rerun()
    
    # Step 3: Training & Evaluation
    with ml_steps[2]:
        # Debug info
        st.markdown(f"<div style='background-color: #E3F2FD; padding: 0.5rem; border-radius: 5px; margin-bottom: 1rem; font-size: 0.8rem;'>Current step: {st.session_state.active_ml_step}, Problem type: {problem_type}</div>", unsafe_allow_html=True)
        
        # Check if we have necessary data and models selected
        if "selected_models" not in st.session_state or not st.session_state.selected_models:
            st.warning("Please complete the Model Selection step first and select at least one model.")
        else:
            st.markdown("<div style='background-color: #E8F5E9; padding: 1rem; border-radius: 10px; border-left: 4px solid #4CAF50; margin-bottom: 1rem;'><h3 style='margin-top: 0; font-size: 1.2rem; color: #2E7D32;'>Training & Evaluation</h3><p style='margin-bottom: 0;'>Train selected models and evaluate their performance on the test set.</p></div>", unsafe_allow_html=True)
            
            # Import the get_available_models function from ml_pipeline
            from ai_eda_pipeline.components.ml_pipeline import get_available_models
            
            # Get available models based on problem type
            available_models = get_available_models(problem_type)
            
            # Display selected models
            st.markdown("#### Selected Models")
            selected_model_names = [available_models[model_id] for model_id in st.session_state.selected_models if model_id in available_models]
            
            # Create a more visually appealing display of selected models
            model_cols = st.columns(min(3, len(selected_model_names)))
            for i, model_name in enumerate(selected_model_names):
                with model_cols[i % len(model_cols)]:
                    st.markdown(f"""
                    <div style='background-color: #F1F8E9; padding: 0.8rem; border-radius: 8px; text-align: center; border: 1px solid #AED581;'>
                        <h4 style='margin: 0; font-size: 1rem;'>{model_name}</h4>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Training options
            with st.expander("Training Options", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Cross-validation options
                    cv_folds = st.slider(
                        "Cross-Validation Folds:",
                        min_value=2,
                        max_value=10,
                        value=5,
                        step=1,
                        key="cv_folds_slider"
                    )
                    
                    # Store CV folds
                    if "training_options" not in st.session_state:
                        st.session_state.training_options = {}
                    st.session_state.training_options["cv_folds"] = cv_folds
                
                with col2:
                    # Hyperparameter tuning option
                    hyperparameter_tuning = st.checkbox(
                        "Enable Hyperparameter Tuning (simulation)",
                        value=False,
                        key="hyperparameter_tuning_checkbox"
                    )
                    
                    # Store hyperparameter tuning option
                    st.session_state.training_options["hyperparameter_tuning"] = hyperparameter_tuning
            
            # Train models button
            train_btn = st.button("Train & Evaluate Models", key="train_evaluate_btn")
            
            # Check if we should train models
            if train_btn or ("trained_models" in st.session_state and st.session_state.trained_models):
                # If button was clicked, train models
                if train_btn:
                    with st.spinner("Training and evaluating models... (simulation)"):
                        try:
                            # Import the prepare_data_for_ml function from ml_pipeline
                            from ai_eda_pipeline.components.ml_pipeline import prepare_data_for_ml
                            
                            # Validate data before training
                            validation_errors = []
                            
                            # Check for missing values
                            missing_values = df_to_use[st.session_state.selected_features].isnull().sum()
                            features_with_missing = missing_values[missing_values > 0]
                            if not features_with_missing.empty:
                                missing_features_str = ", ".join(features_with_missing.index.tolist())
                                validation_errors.append(f"Missing values found in features: {missing_features_str}")
                            
                            # Check for target missing values
                            if df_to_use[target_col].isnull().any():
                                validation_errors.append(f"Missing values found in target column: {target_col}")
                            
                            # Check for inappropriate data types
                            for feature in st.session_state.selected_features:
                                if df_to_use[feature].dtype == 'object' and feature not in df_to_use.select_dtypes(include=['category']).columns:
                                    validation_errors.append(f"Feature '{feature}' has text/categorical data that needs encoding")
                            
                            # For classification, check class distribution
                            if problem_type == "classification":
                                class_counts = df_to_use[target_col].value_counts()
                                if class_counts.min() < 5:
                                    small_classes = class_counts[class_counts < 5].index.tolist()
                                    validation_errors.append(f"Some classes have very few samples: {small_classes}")
                                
                                # Check number of classes
                                if len(class_counts) < 2:
                                    validation_errors.append(f"Target column '{target_col}' has only one class, which is not suitable for classification")
                            
                            # For regression, check target distribution
                            if problem_type == "regression":
                                if df_to_use[target_col].std() == 0:
                                    validation_errors.append(f"Target column '{target_col}' has zero variance, which is not suitable for regression")
                            
                            # Check if we have enough features
                            if len(st.session_state.selected_features) == 0:
                                validation_errors.append("No features selected for training")
                            
                            # Check if we have enough samples
                            if len(df_to_use) < 10:
                                validation_errors.append(f"Dataset has only {len(df_to_use)} samples, which is too few for reliable model training")
                            
                            # If we have validation errors, display them and stop
                            if validation_errors:
                                st.error("Data validation failed. Please fix the following issues:")
                                for error in validation_errors:
                                    st.warning(error)
                                
                                # Provide guidance on how to fix the issues
                                st.info("Recommendations to fix these issues:")
                                st.markdown("""
                                1. **For missing values**: Use the preprocessing tab to impute missing values
                                2. **For categorical features**: Use the preprocessing tab to encode categorical features
                                3. **For class imbalance**: Consider collecting more data for underrepresented classes or using techniques like SMOTE
                                4. **For feature selection**: Select more relevant features for your target variable
                                """)
                                
                                # Log the validation errors
                                for error in validation_errors:
                                    add_ml_message(f"Validation error: {error}", is_error=True)
                                
                                # Stop here
                                return
                            
                            # Prepare data for ML
                            X = df_to_use[st.session_state.selected_features].values
                            y = df_to_use[target_col].values
                            
                            # Use the correct parameter names for prepare_data_for_ml
                            X_train, X_test, y_train, y_test = prepare_data_for_ml(
                                df=df_to_use,
                                target_column=target_col,
                                feature_columns=st.session_state.selected_features,
                                test_size=st.session_state.train_test_split["test_size"],
                                random_state=st.session_state.train_test_split["random_state"],
                                stratify=st.session_state.train_test_split.get("stratify", False)
                            )
                            
                            # Train selected models
                            trained_models, evaluation_results, plots = train_and_evaluate_models(
                                X_train=X_train,
                                X_test=X_test,
                                y_train=y_train,
                                y_test=y_test,
                                selected_models=st.session_state.selected_models,
                                problem_type=problem_type,
                                cv_folds=st.session_state.training_options.get("cv_folds", 5),
                                hyperparameter_tuning=st.session_state.training_options.get("hyperparameter_tuning", False)
                            )
                            
                            # Store results in session state
                            st.session_state.trained_models = trained_models
                            st.session_state.evaluation_results = evaluation_results
                            st.session_state.evaluation_plots = plots
                            
                            # Identify best model
                            best_model_id = identify_best_model(evaluation_results, problem_type)
                            st.session_state.best_model_id = best_model_id
                            
                            # Log the action
                            add_ml_message(f"Trained and evaluated {len(trained_models)} models", is_insight=True)
                            
                            # Mark ML phase as completed
                            st.session_state.phase_status["ml"]["completed"] = True
                            
                            # Set active step to model insights
                            st.session_state.active_ml_step = 3
                            
                            # Rerun to show results
                            st.experimental_rerun()
                        except Exception as e:
                            # Display error message
                            st.error(f"Error during model training: {str(e)}")
                            add_ml_message(f"Error during model training: {str(e)}", is_error=True)
                            
                            # Log the error for debugging
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
                            
                            # Provide guidance on how to fix the issue
                            st.warning("Please try the following steps to resolve the issue:")
                            st.markdown("""
                            1. Make sure you've selected appropriate features for your target variable
                            2. Check if your dataset has missing values that need to be handled
                            3. For classification problems, ensure your target variable has enough samples for each class
                            4. Try selecting different models that might be more suitable for your data
                            """)
                
                # Display evaluation results
                if "evaluation_results" in st.session_state and st.session_state.evaluation_results:
                    results = st.session_state.evaluation_results
                    plots = st.session_state.evaluation_plots
                    best_model_id = st.session_state.best_model_id
                    
                    # Create tabs for different evaluation views
                    eval_tabs = st.tabs(["📊 Metrics", "📈 Plots", "🔍 Feature Importance"])
                    
                    # Metrics tab
                    with eval_tabs[0]:
                        st.markdown("#### Model Performance Metrics")
                        
                        # Create a DataFrame for metrics
                        metrics_df = pd.DataFrame(results)
                        
                        # Highlight the best model
                        if best_model_id:
                            # Fix: available_models returns strings, not dictionaries
                            best_model_name = available_models[best_model_id]
                            st.markdown(f"**Best Model: {best_model_name}** 🏆")
                        
                        # Display metrics table
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Add download button for metrics
                        csv = metrics_df.to_csv(index=False)
                        st.download_button(
                            label="Download Metrics as CSV",
                            data=csv,
                            file_name="model_evaluation_metrics.csv",
                            mime="text/csv",
                            key="download_metrics_btn"
                        )
                    
                    # Plots tab
                    with eval_tabs[1]:
                        if plots:
                            # Create a dropdown to select the plot type
                            plot_type = st.selectbox(
                                "Select Plot Type:",
                                options=list(plots.keys()),
                                key="plot_type_select"
                            )
                            
                            # Display the selected plot
                            if plot_type in plots:
                                st.plotly_chart(plots[plot_type], use_container_width=True)
                            else:
                                st.info(f"No {plot_type} plot available.")
                        else:
                            st.info("No plots available.")
                    
                    # Feature Importance tab
                    with eval_tabs[2]:
                        if "feature_importance" in plots:
                            st.markdown("#### Feature Importance")
                            st.plotly_chart(plots["feature_importance"], use_container_width=True)
                            
                            # Add explanation
                            st.markdown("""
                            **Understanding Feature Importance:**
                            - Higher values indicate features that had a greater impact on the model's predictions
                            - Feature importance helps identify which variables are most influential in your model
                            - Consider focusing on high-importance features for future model iterations
                            """)
                        else:
                            st.info("Feature importance plot not available for the selected models.")
                    
                    # Proceed button
                    if st.button("Proceed to Model Insights", key="proceed_to_insights_btn"):
                        # Set active step to model insights
                        st.session_state.active_ml_step = 3
                        
                        # Generate model insights
                        with st.spinner("Generating model insights... (simulation)"):
                            insights = st.session_state.ml_agent.get_model_evaluation_insights(
                                trained_models=st.session_state.trained_models,
                                best_model_id=st.session_state.best_model_id,
                                problem_type=problem_type
                            )
                            
                            # Store insights
                            st.session_state.model_insights = insights
                            
                            # Log the action
                            add_ml_message("Generated model insights based on evaluation results", is_insight=True)
                        
                        # Show success message and rerun
                        st.success("Evaluation complete! Proceeding to Model Insights...")
                        st.experimental_rerun()
    
    # Step 4: Model Insights
    with ml_steps[3]:
        # Debug info
        st.markdown(f"<div style='background-color: #E3F2FD; padding: 0.5rem; border-radius: 5px; margin-bottom: 1rem; font-size: 0.8rem;'>Current step: {st.session_state.active_ml_step}, Problem type: {problem_type}</div>", unsafe_allow_html=True)
        
        # Check if we have necessary data
        if "model_insights" not in st.session_state or not st.session_state.model_insights:
            st.warning("Please complete the Training & Evaluation step first to get model insights.")
        else:
            st.markdown("<div style='background-color: #E1F5FE; padding: 1rem; border-radius: 10px; border-left: 4px solid #03A9F4; margin-bottom: 1rem;'><h3 style='margin-top: 0; font-size: 1.2rem; color: #01579B;'>Model Insights</h3><p style='margin-bottom: 0;'>AI-generated insights and recommendations based on model performance.</p></div>", unsafe_allow_html=True)
            
            # Import the get_available_models function from ml_pipeline
            from ai_eda_pipeline.components.ml_pipeline import get_available_models
            
            # Get available models based on problem type
            available_models = get_available_models(problem_type)
            
            insights = st.session_state.model_insights
            
            # Create tabs for different insight categories
            insight_tabs = st.tabs(["📊 Performance Analysis", "🔄 Model Comparison", "🔍 Feature Insights", "📈 Improvement Recommendations"])
            
            # Performance Analysis tab
            with insight_tabs[0]:
                st.markdown("#### Performance Analysis")
                
                if "performance_analysis" in insights:
                    # Create a more visually appealing card for performance analysis
                    st.markdown(f"""
                    <div style='background-color: #F3F9FF; padding: 1.2rem; border-radius: 10px; border: 1px solid #BBDEFB; margin-bottom: 1rem;'>
                        <p style='margin-bottom: 0;'>{insights["performance_analysis"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No performance analysis available.")
            
            # Model Comparison tab
            with insight_tabs[1]:
                st.markdown("#### Model Comparison")
                
                if "model_comparison" in insights:
                    # Create a more visually appealing card for model comparison
                    st.markdown(f"""
                    <div style='background-color: #F3F9FF; padding: 1.2rem; border-radius: 10px; border: 1px solid #BBDEFB; margin-bottom: 1rem;'>
                        <p style='margin-bottom: 0;'>{insights["model_comparison"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No model comparison available.")
            
            # Feature Insights tab
            with insight_tabs[2]:
                st.markdown("#### Feature Insights")
                
                if "feature_insights" in insights:
                    # Create a more visually appealing card for feature insights
                    st.markdown(f"""
                    <div style='background-color: #F3F9FF; padding: 1.2rem; border-radius: 10px; border: 1px solid #BBDEFB; margin-bottom: 1rem;'>
                        <p style='margin-bottom: 0;'>{insights["feature_insights"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No feature insights available.")
            
            # Improvement Recommendations tab
            with insight_tabs[3]:
                st.markdown("#### Improvement Recommendations")
                
                if "improvement_recommendations" in insights:
                    # Create a more visually appealing card for improvement recommendations
                    st.markdown(f"""
                    <div style='background-color: #F3F9FF; padding: 1.2rem; border-radius: 10px; border: 1px solid #BBDEFB; margin-bottom: 1rem;'>
                        <h4 style='margin-top: 0; color: #01579B; font-size: 1.1rem;'>Recommendations</h4>
                        <ul style='margin-bottom: 0.5rem;'>
                            {"".join([f"<li>{improvement}</li>" for improvement in insights["improvement_recommendations"]])}
                        </ul>
                        
                        <h4 style='color: #01579B; font-size: 1.1rem;'>Next Steps</h4>
                        <ul style='margin-bottom: 0;'>
                            {"".join([f"<li>{step}</li>" for step in insights["next_steps"]])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No improvement recommendations available.")
            
            # Download model button
            st.markdown("### Download Best Model")
            st.info("In a real application, you would be able to download the trained model for deployment.")
            
            if st.button("Download Best Model (Simulation)", key="download_model_btn"):
                st.success("Model download simulated. In a real application, this would save the model to your computer.")
                
                # Log the action
                add_ml_message("Downloaded best model for deployment")
                
                # Show a toast notification
                st.toast("Model downloaded successfully!", icon="✅")
    
    # Add a note about the simulation
    st.markdown("""
    <div style='background-color: #FFF3E0; padding: 1rem; border-radius: 10px; border-left: 4px solid #FF9800; margin-top: 1rem;'>
        <h3 style='margin-top: 0; font-size: 1.1rem; color: #E65100;'>Simulation Note</h3>
        <p style='margin-bottom: 0;'>This ML pipeline is a simulation for demonstration purposes. In a real application, it would connect to actual machine learning libraries like scikit-learn, TensorFlow, or PyTorch for training and evaluation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display ML messages
    if st.session_state.ml_messages:
        with st.expander("ML Pipeline Log", expanded=False):
            for msg in st.session_state.ml_messages:
                timestamp = msg.get("timestamp", "")
                message = msg.get("message", "")
                is_insight = msg.get("is_insight", False)
                
                if is_insight:
                    st.markdown(f"**{timestamp}** - 💡 *{message}*")
                else:
                    st.markdown(f"**{timestamp}** - {message}")

# Function to export preprocessing workflow to JSON
def export_workflow_to_json():
    """Export the preprocessing workflow to a JSON file."""
    if not st.session_state.preprocessing_workflow:
        add_preprocessing_message("No preprocessing steps to export", is_error=True)
        return None
    
    try:
        # Create a dictionary with workflow and metadata
        export_data = {
            "preprocessing_workflow": st.session_state.preprocessing_workflow,
            "dataset_name": st.session_state.dataset_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {
                "original_shape": list(st.session_state.df.shape),
                "columns": list(st.session_state.df.columns)
            }
        }
        
        # Convert to JSON string
        json_str = json.dumps(export_data, indent=2)
        
        add_preprocessing_message("Exported preprocessing workflow to JSON")
        return json_str
    except Exception as e:
        add_preprocessing_message(f"Error exporting workflow to JSON: {str(e)}", is_error=True)
        return None

# Function to export preprocessing workflow to Python code
def export_workflow_to_python():
    """Export the preprocessing workflow as Python code."""
    if not st.session_state.preprocessing_workflow:
        add_preprocessing_message("No preprocessing steps to export", is_error=True)
        return None
    
    try:
        # Generate Python code
        code_lines = [
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder",
            "from sklearn.impute import KNNImputer",
            "import scipy.stats as stats",
            "",
            "def preprocess_data(df):",
            "    \"\"\"",
            f"    Preprocessing workflow generated by Machine Learning Playground",
            f"    Dataset: {st.session_state.dataset_name}",
            f"    Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "    \"\"\"",
            "    # Make a copy of the dataframe to avoid modifying the original",
            "    processed_df = df.copy()",
            ""
        ]
        
        # Add each preprocessing step
        for i, step in enumerate(st.session_state.preprocessing_workflow):
            step_id = step["step"]
            
            # Add comment for the step
            code_lines.append(f"    # Step {i+1}: {PREPROCESSING_STEPS.get(step_id, {}).get('name', step_id)}")
            if "reason" in step and step["reason"]:
                code_lines.append(f"    # Reason: {step['reason']}")
            
            # Generate code based on step type
            if step_id == "impute_missing":
                col = step.get("column", "")
                method = step.get("method", "median")
                
                if method == "median":
                    code_lines.append(f"    processed_df['{col}'] = processed_df['{col}'].fillna(processed_df['{col}'].median())")
                elif method == "mean":
                    code_lines.append(f"    processed_df['{col}'] = processed_df['{col}'].fillna(processed_df['{col}'].mean())")
                elif method == "mode":
                    code_lines.append(f"    processed_df['{col}'] = processed_df['{col}'].fillna(processed_df['{col}'].mode()[0])")
                elif method == "constant":
                    value = step.get("value", 0)
                    code_lines.append(f"    processed_df['{col}'] = processed_df['{col}'].fillna({value})")
                elif method == "new_category":
                    code_lines.append(f"    processed_df['{col}'] = processed_df['{col}'].fillna('Missing')")
                elif method == "knn":
                    code_lines.append(f"    imputer = KNNImputer(n_neighbors=5)")
                    code_lines.append(f"    processed_df['{col}'] = imputer.fit_transform(processed_df[['{col}']])[:, 0]")
            
            elif step_id == "drop_column":
                col = step.get("column", "")
                code_lines.append(f"    processed_df = processed_df.drop(columns=['{col}'])")
            
            elif step_id == "handle_outliers":
                col = step.get("column", "")
                method = step.get("method", "winsorize")
                
                code_lines.append(f"    Q1 = processed_df['{col}'].quantile(0.25)")
                code_lines.append(f"    Q3 = processed_df['{col}'].quantile(0.75)")
                code_lines.append(f"    IQR = Q3 - Q1")
                code_lines.append(f"    lower_bound = Q1 - 1.5 * IQR")
                code_lines.append(f"    upper_bound = Q3 + 1.5 * IQR")
                
                if method == "remove":
                    code_lines.append(f"    processed_df = processed_df[(processed_df['{col}'] >= lower_bound) & (processed_df['{col}'] <= upper_bound)]")
                elif method == "winsorize":
                    code_lines.append(f"    processed_df['{col}'] = processed_df['{col}'].clip(lower=lower_bound, upper=upper_bound)")
            
            elif step_id == "transform":
                col = step.get("column", "")
                method = step.get("method", "log")
                
                if method == "log":
                    code_lines.append(f"    min_val = processed_df['{col}'].min()")
                    code_lines.append(f"    if min_val <= 0:")
                    code_lines.append(f"        shift = abs(min_val) + 1")
                    code_lines.append(f"        processed_df['{col}'] = np.log(processed_df['{col}'] + shift)")
                    code_lines.append(f"    else:")
                    code_lines.append(f"        processed_df['{col}'] = np.log(processed_df['{col}'])")
                elif method == "sqrt":
                    code_lines.append(f"    min_val = processed_df['{col}'].min()")
                    code_lines.append(f"    if min_val < 0:")
                    code_lines.append(f"        shift = abs(min_val)")
                    code_lines.append(f"        processed_df['{col}'] = np.sqrt(processed_df['{col}'] + shift)")
                    code_lines.append(f"    else:")
                    code_lines.append(f"        processed_df['{col}'] = np.sqrt(processed_df['{col}'])")
                elif method == "box-cox":
                    code_lines.append(f"    min_val = processed_df['{col}'].min()")
                    code_lines.append(f"    if min_val <= 0:")
                    code_lines.append(f"        shift = abs(min_val) + 1")
                    code_lines.append(f"        processed_df['{col}'], _ = stats.boxcox(processed_df['{col}'] + shift)")
                    code_lines.append(f"    else:")
                    code_lines.append(f"        processed_df['{col}'], _ = stats.boxcox(processed_df['{col}'])")
            
            elif step_id == "convert_type":
                col = step.get("column", "")
                method = step.get("method", "")
                
                if method == "to_numeric":
                    code_lines.append(f"    processed_df['{col}'] = pd.to_numeric(processed_df['{col}'], errors='coerce')")
                elif method == "to_categorical":
                    code_lines.append(f"    processed_df['{col}'] = processed_df['{col}'].astype('category')")
                elif method == "to_datetime":
                    code_lines.append(f"    processed_df['{col}'] = pd.to_datetime(processed_df['{col}'], errors='coerce')")
            
            elif step_id == "reduce_cardinality":
                col = step.get("column", "")
                method = step.get("method", "group_rare")
                threshold = step.get("threshold", 0.01)
                
                if method == "group_rare":
                    code_lines.append(f"    value_counts = processed_df['{col}'].value_counts(normalize=True)")
                    code_lines.append(f"    rare_categories = value_counts[value_counts < {threshold}].index")
                    code_lines.append(f"    processed_df['{col}'] = processed_df['{col}'].replace(rare_categories, 'Other')")
            
            elif step_id == "encode":
                col = step.get("column", "")
                method = step.get("method", "onehot")
                
                if method == "label":
                    code_lines.append(f"    le = LabelEncoder()")
                    code_lines.append(f"    processed_df['{col}'] = le.fit_transform(processed_df['{col}'].astype(str))")
                elif method == "onehot":
                    code_lines.append(f"    dummies = pd.get_dummies(processed_df['{col}'], prefix='{col}', drop_first=False)")
                    code_lines.append(f"    processed_df = pd.concat([processed_df.drop(columns=['{col}']), dummies], axis=1)")
                elif method == "target":
                    code_lines.append(f"    if 'target' in processed_df.columns:")
                    code_lines.append(f"        means = processed_df.groupby('{col}')['target'].mean()")
                    code_lines.append(f"        processed_df['{col}_encoded'] = processed_df['{col}'].map(means)")
                    code_lines.append(f"        processed_df = processed_df.drop(columns=['{col}'])")
            
            elif step_id == "scale":
                columns = step.get("columns", [])
                method = step.get("method", "standard")
                
                columns_str = "', '".join(columns)
                
                if method == "standard":
                    code_lines.append(f"    scaler = StandardScaler()")
                    code_lines.append(f"    processed_df[['{columns_str}']] = scaler.fit_transform(processed_df[['{columns_str}']])")
                elif method == "minmax":
                    code_lines.append(f"    scaler = MinMaxScaler()")
                    code_lines.append(f"    processed_df[['{columns_str}']] = scaler.fit_transform(processed_df[['{columns_str}']])")
                elif method == "robust":
                    code_lines.append(f"    scaler = RobustScaler()")
                    code_lines.append(f"    processed_df[['{columns_str}']] = scaler.fit_transform(processed_df[['{columns_str}']])")
            
            elif step_id == "handle_correlation":
                columns = step.get("columns", [])
                method = step.get("method", "drop_one")
                
                if len(columns) == 2:
                    col1, col2 = columns
                    
                    if method == "drop_one":
                        code_lines.append(f"    processed_df = processed_df.drop(columns=['{col2}'])")
                    elif method == "pca":
                        code_lines.append(f"    from sklearn.decomposition import PCA")
                        code_lines.append(f"    pca = PCA(n_components=1)")
                        code_lines.append(f"    processed_df['{col1}_{col2}_pca'] = pca.fit_transform(processed_df[['{col1}', '{col2}']])")
                        code_lines.append(f"    processed_df = processed_df.drop(columns=['{col1}', '{col2}'])")
            
            # Add a blank line between steps
            code_lines.append("")
        
        # Return the processed dataframe
        code_lines.append("    return processed_df")
        
        # Join all lines with newlines
        python_code = "\n".join(code_lines)
        
        add_preprocessing_message("Exported preprocessing workflow to Python code")
        return python_code
    except Exception as e:
        add_preprocessing_message(f"Error exporting workflow to Python: {str(e)}", is_error=True)
        return None

# Function to export preprocessing workflow to PDF report
def export_workflow_to_pdf():
    """Export the preprocessing workflow as a PDF report."""
    if not st.session_state.preprocessing_workflow:
        add_preprocessing_message("No preprocessing steps to export", is_error=True)
        return None
    
    try:
        # Create a BytesIO object to store the PDF
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        )
        
        subheading_style = ParagraphStyle(
            'Subheading',
            parent=styles['Heading3'],
            fontSize=12,
            spaceAfter=8
        )
        
        normal_style = styles['Normal']
        
        # Create the content
        content = []
        
        # Title
        content.append(Paragraph(f"Preprocessing Workflow Report", title_style))
        content.append(Paragraph(f"Dataset: {st.session_state.dataset_name}", normal_style))
        content.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        content.append(Spacer(1, 12))
        
        # Dataset information
        content.append(Paragraph("Dataset Information", heading_style))
        
        # Dataset shape
        shape = st.session_state.df.shape
        content.append(Paragraph(f"Original shape: {shape[0]} rows, {shape[1]} columns", normal_style))
        
        if st.session_state.processed_df is not None:
            processed_shape = st.session_state.processed_df.shape
            content.append(Paragraph(f"Processed shape: {processed_shape[0]} rows, {processed_shape[1]} columns", normal_style))
        
        content.append(Spacer(1, 12))
        
        # Workflow evaluation if available
        if "workflow_evaluation" in st.session_state:
            content.append(Paragraph("Workflow Evaluation", heading_style))
            
            eval_score = st.session_state.workflow_evaluation["score"]
            eval_feedback = st.session_state.workflow_evaluation["feedback"]
            
            content.append(Paragraph(f"Score: {int(eval_score * 10)}/10", normal_style))
            content.append(Paragraph(f"Feedback:", normal_style))
            content.append(Paragraph(eval_feedback, normal_style))
            content.append(Spacer(1, 12))
        
        # Preprocessing steps
        content.append(Paragraph("Preprocessing Steps", heading_style))
        
        for i, step in enumerate(st.session_state.preprocessing_workflow):
            step_id = step["step"]
            step_info = PREPROCESSING_STEPS.get(step_id, {})
            
            # Step title
            content.append(Paragraph(f"Step {i+1}: {step_info.get('name', step_id)}", subheading_style))
            
            # Step details
            details = []
            
            if "column" in step:
                details.append(["Column", step["column"]])
            
            if "columns" in step and isinstance(step["columns"], list):
                details.append(["Columns", ", ".join(step["columns"])])
            
            if "method" in step:
                method_name = step_info.get("methods", {}).get(step["method"], step["method"])
                details.append(["Method", method_name])
            
            if "reason" in step and step["reason"]:
                details.append(["Reason", step["reason"]])
            
            # Create a table for the details
            if details:
                table = Table(details, colWidths=[100, 300])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('BACKGROUND', (1, 0), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                content.append(table)
            
            content.append(Spacer(1, 10))
        
        # Build the PDF
        doc.build(content)
        
        # Get the PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        add_preprocessing_message("Exported preprocessing workflow to PDF report")
        return pdf_data
    except Exception as e:
        add_preprocessing_message(f"Error exporting workflow to PDF: {str(e)}", is_error=True)
        return None

# Main application layout
def main():
    # Create three columns for the layout
    
    # Add custom CSS for a more modern and interesting UI
    st.markdown("""
    <style>
        /* Modern color scheme */
        :root {
            --primary: #2E7D32;
            --primary-light: #4CAF50;
            --primary-dark: #1B5E20;
            --accent: #FF9800;
            --accent-light: #FFB74D;
            --text-primary: #212121;
            --text-secondary: #757575;
            --background: #FFFFFF;
            --card-bg: #F5F7F9;
            --border: #E0E0E0;
        }
        
        /* General styling */
        .main .block-container {
            padding-top: 1rem;
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        /* Make the sidebar narrower */
        [data-testid="stSidebar"] {
            min-width: 250px !important;
            max-width: 300px !important;
        }
        
        /* Make the main content area wider */
        .main {
            width: calc(100% - 300px) !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-dark);
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        /* Make plots stand out with a card-like appearance and ensure they're wide */
        [data-testid="stPlotlyChart"] {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border);
            width: 100% !important;
        }
        
        /* Ensure plots are responsive */
        [data-testid="stPlotlyChart"] > div {
            width: 100% !important;
        }
        
        /* Improve dataframe appearance */
        [data-testid="stDataFrame"] {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border);
            width: 100% !important;
        }
        
        /* Card styling */
        .stCard {
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            padding: 1rem;
            margin-bottom: 1rem;
            background-color: var(--card-bg);
            border-left: 4px solid var(--primary);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            width: 100% !important;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: var(--primary);
            color: white;
            border-radius: 6px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background-color: var(--primary-dark);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-1px);
        }
        
        /* Tabs styling - make them more visible */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #f0f2f6;
            padding: 0.25rem;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--primary-light) !important;
            color: white !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid var(--border);
        }
        
        /* Metrics styling - make them more compact and fit better */
        [data-testid="stMetric"] {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            padding: 0.5rem;
            border: 1px solid var(--border);
            margin-bottom: 0.5rem;
        }
        [data-testid="stMetric"] > div:first-child {
            color: var(--primary);
            font-size: 0.8rem;
        }
        [data-testid="stMetric"] > div:nth-child(2) {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        /* Make columns fit better */
        [data-testid="column"] {
            padding: 0.25rem !important;
        }
        
        /* Improve selectbox and multiselect */
        [data-testid="stSelectbox"], [data-testid="stMultiSelect"] {
            background-color: white;
            border-radius: 6px;
            border: 1px solid var(--border);
            padding: 0.25rem;
            margin-bottom: 0.5rem;
        }
        
        /* Improve expander */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: var(--primary);
            background-color: #f0f2f6;
            border-radius: 6px;
            padding: 0.5rem 1rem;
        }
        
        /* Custom header styling */
        .sidebar-header {
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            color: var(--primary);
            border-bottom: 2px solid var(--primary-light);
            padding-bottom: 0.25rem;
        }
        
        /* Success/Info/Warning/Error message styling */
        .stSuccess, .stInfo, .stWarning, .stError {
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
        }
        
        /* Make the ML Pipeline section wider */
        .element-container:has([data-testid="stVerticalBlock"]) {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Ensure plots in the ML Pipeline section are wide */
        .element-container:has([data-testid="stVerticalBlock"]) [data-testid="stPlotlyChart"] {
            width: 100% !important;
        }
        
        /* Fix for confusion matrix width */
        .js-plotly-plot, .plot-container {
            width: 100% !important;
        }
        
        /* Ensure all three sections fit well */
        .row-widget.stRadio > div {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
        }
        
        /* Make the blue info boxes more visible */
        .stInfo {
            background-color: #E3F2FD;
            border-left: 4px solid #2196F3;
            padding: 1rem;
            margin: 1rem 0;
            width: 100% !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Rest of the main function
    left_col, main_col, right_col = st.columns([1, 2, 1])
    
    # Left sidebar for dataset selection and available plots
    with left_col:
        st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Dataset Selection</div>", unsafe_allow_html=True)
        
        # Dataset selection options
        dataset_option = st.radio(
            "Choose dataset source:",
            ["Upload your own", "Use predefined dataset"]
        )
        
        if dataset_option == "Upload your own":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                if st.button("Load Dataset"):
                    with st.spinner("Loading and analyzing dataset..."):
                        success = load_and_analyze_dataset(file_buffer=uploaded_file)
                        if success:
                            st.success("Dataset loaded successfully!")
        
        else:  # Use predefined dataset
            predefined_datasets = [
                "iris", "wine", "breast_cancer", "diabetes", 
                "titanic", "tips", "planets"
            ]
            selected_dataset = st.selectbox("Select dataset:", predefined_datasets)
            
            if st.button("Load Dataset"):
                with st.spinner("Loading and analyzing dataset..."):
                    success = load_and_analyze_dataset(dataset_name=selected_dataset)
                    if success:
                        st.success("Dataset loaded successfully!")
        
        # Display sidebar content based on active tab
        if st.session_state.df is not None:
            if st.session_state.active_tab == "eda":
                # EDA sidebar content - available plots
                st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Available Plots</div>", unsafe_allow_html=True)
                
                # Create a container with max height and scrolling
                plot_container = st.container()
                
                with plot_container:
                    # Add CSS for scrollable container and fancy plot boxes
                    st.markdown("""
                    <style>
                    .scrollable-container {
                        max-height: 500px;
                        overflow-y: auto;
                        padding-right: 10px;
                    }
                    
                    .plot-category {
                        margin-bottom: 15px;
                        border-bottom: 1px solid #eee;
                        padding-bottom: 5px;
                    }
                    
                    .plot-category-title {
                        font-size: 16px;
                        font-weight: 600;
                        color: #333;
                        margin-bottom: 10px;
                        display: flex;
                        align-items: center;
                    }
                    
                    .fancy-plot-box {
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 12px;
                        margin-bottom: 10px;
                        text-align: center;
                        background: linear-gradient(to bottom, #ffffff, #f9f9f9);
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                        transition: all 0.3s ease;
                    }
                    
                    .fancy-plot-box:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        border-color: #4CAF50;
                    }
                    
                    .plot-icon {
                        font-size: 24px;
                        margin-bottom: 8px;
                        color: #4CAF50;
                    }
                    
                    .plot-name {
                        font-weight: 500;
                        margin-bottom: 5px;
                        color: #333;
                    }
                    
                    .plot-description {
                        font-size: 12px;
                        color: #666;
                        margin-bottom: 10px;
                        height: 36px;
                        overflow: hidden;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<div class='scrollable-container'>", unsafe_allow_html=True)
                    
                    # Group plots by category with better descriptions
                    plot_categories = [
                        {
                            "name": "Distribution", 
                            "icon": "📈",
                            "plots": [
                                {"type": "histogram", "desc": "Shows frequency distribution of a numeric variable"},
                                {"type": "boxplot", "desc": "Displays median, quartiles and outliers"}
                            ]
                        },
                        {
                            "name": "Relationship", 
                            "icon": "🔗",
                            "plots": [
                                {"type": "scatter", "desc": "Shows relationship between two numeric variables"},
                                {"type": "correlation_heatmap", "desc": "Visualizes correlation matrix between variables"},
                                {"type": "pairplot", "desc": "Creates a matrix of scatter plots for multiple variables"}
                            ]
                        },
                        {
                            "name": "Categorical", 
                            "icon": "📊",
                            "plots": [
                                {"type": "bar", "desc": "Compares values across different categories"},
                                {"type": "pie", "desc": "Shows proportion of categories in a whole"}
                            ]
                        },
                        {
                            "name": "Time Series", 
                            "icon": "⏱️",
                            "plots": [
                                {"type": "line", "desc": "Shows trends over a continuous variable"}
                            ]
                        }
                    ]
                    
                    for category in plot_categories:
                        st.markdown(f"<div class='plot-category'><div class='plot-category-title'>{category['name']}</div>", unsafe_allow_html=True)
                        
                        # Create a grid of 2 columns for the plots
                        plot_cols = st.columns(2)
                        col_idx = 0
                        
                        for plot_info in category['plots']:
                            plot_type = plot_info["type"]
                            plot_desc = plot_info["desc"]
                            
                            with plot_cols[col_idx % 2]:
                                st.markdown(f"""
                                <div class="fancy-plot-box">
                                    <div class="plot-icon">
                                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                            <path d="M3 3V21H21" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                            <path d="M7 14L11 10L15 14L19 10" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        </svg>
                                    </div>
                                    <div class="plot-name">{plot_type.capitalize()}</div>
                                    <div class="plot-description">{plot_desc}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if st.button(f"Add to Dashboard", key=f"add_{plot_type}"):
                                    # Get configuration for the plot type
                                    config = get_plot_config(plot_type, st.session_state.df)
                                    
                                    # Add to dashboard
                                    add_plot_to_dashboard(plot_type, config)
                                    add_to_log(f"Added {plot_type} plot to dashboard")
                                    st.experimental_rerun()
                            
                            col_idx += 1
                        
                        st.markdown("</div>", unsafe_allow_html=True)  # Close plot-category div
                    
                    st.markdown("</div>", unsafe_allow_html=True)  # Close the scrollable container
            
            elif st.session_state.active_tab == "preprocessing":
                # Preprocessing sidebar content - available steps
                st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Preprocessing Steps</div>", unsafe_allow_html=True)
                
                # Add CSS for preprocessing steps
                st.markdown("""
                <style>
                .preprocessing-category {
                    margin-bottom: 15px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 5px;
                }
                
                .preprocessing-category-title {
                    font-size: 16px;
                    font-weight: 600;
                    color: #333;
                    margin-bottom: 10px;
                    display: flex;
                    align-items: center;
                }
                
                .preprocessing-step-box {
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 10px;
                    background: linear-gradient(to bottom, #ffffff, #f9f9f9);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    transition: all 0.3s ease;
                    cursor: grab;
                }
                
                .preprocessing-step-box:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    border-color: #4CAF50;
                }
                
                .step-icon {
                    font-size: 20px;
                    margin-right: 8px;
                    color: #4CAF50;
                }
                
                .step-name {
                    font-weight: 500;
                    margin-bottom: 5px;
                    color: #333;
                }
                
                .step-description {
                    font-size: 12px;
                    color: #666;
                    margin-bottom: 5px;
                }
                
                .ai-suggested {
                    border-left: 3px solid #2196F3;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # AI-suggested preprocessing steps
                with st.expander("🤖 AI-Suggested Steps", expanded=True):
                    # Remove the button to get AI suggestions from the left sidebar
                    # This will only be available in the right sidebar
                    
                    # Display AI suggestions if available
                    if "preprocessing_suggestions" in st.session_state and st.session_state.preprocessing_suggestions:
                        for i, suggestion in enumerate(st.session_state.preprocessing_suggestions):
                            step_type = suggestion.get("step", "")
                            step_info = PREPROCESSING_STEPS.get(step_type, {})
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"""
                                <div class="preprocessing-step-box ai-suggested">
                                    <div class="step-name">{step_info.get('icon', '🔧')} {step_info.get('name', step_type)}</div>
                                    <div class="step-description">{suggestion.get('reason', '')}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                if st.button("Add", key=f"add_suggestion_{i}"):
                                    st.session_state.configuring_step = step_type
                                    # Pre-fill configuration with suggestion
                                    st.session_state.prefill_config = suggestion
                                    st.experimental_rerun()
                    else:
                        st.info("AI suggestions will appear here. Click 'Get AI Preprocessing Suggestions' in the right sidebar to generate them.")
                
                # Manual preprocessing steps selection
                with st.expander("🛠️ Manual Steps", expanded=True):
                    # Get preprocessing steps by category
                    steps_by_category = get_preprocessing_steps_by_category()
                    
                    for category, steps in steps_by_category.items():
                        st.markdown(f"<div class='preprocessing-category-title'>{category}</div>", unsafe_allow_html=True)
                        
                        for step in steps:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"""
                                <div class="preprocessing-step-box">
                                    <div class="step-name">{step.get('icon', '🔧')} {step.get('name', '')}</div>
                                    <div class="step-description">{step.get('description', '')}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                if st.button("Add", key=f"add_step_{step['id']}"):
                                    st.session_state.configuring_step = step['id']
                                    st.experimental_rerun()
            
            elif st.session_state.active_tab == "ml":
                # ML Pipeline sidebar content
                st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>ML Pipeline Options</div>", unsafe_allow_html=True)
                
                # Add ML Pipeline options
                st.markdown("""
                <style>
                .ml-option-box {
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 15px;
                    background: linear-gradient(to bottom, #ffffff, #f9f9f9);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    transition: all 0.3s ease;
                }
                
                .ml-option-box:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    border-color: #4CAF50;
                }
                
                .ml-option-title {
                    font-weight: 600;
                    margin-bottom: 8px;
                    color: #2E7D32;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .ml-option-description {
                    font-size: 13px;
                    color: #555;
                    margin-bottom: 10px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display ML Pipeline options
                st.markdown("""
                <div class="ml-option-box">
                    <div class="ml-option-title">🎯 Model Selection</div>
                    <div class="ml-option-description">Choose from various ML algorithms suitable for your data and problem type.</div>
                </div>
                
                <div class="ml-option-box">
                    <div class="ml-option-title">⚙️ Hyperparameter Tuning</div>
                    <div class="ml-option-description">Optimize model parameters for better performance.</div>
                </div>
                
                <div class="ml-option-box">
                    <div class="ml-option-title">📊 Feature Engineering</div>
                    <div class="ml-option-description">Create new features or transform existing ones to improve model accuracy.</div>
                </div>
                
                <div class="ml-option-box">
                    <div class="ml-option-title">🔍 Model Evaluation</div>
                    <div class="ml-option-description">Assess model performance using various metrics and validation techniques.</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("ML Pipeline options will be fully implemented in Phase 3.")
            
        # Display reasoning log at the bottom
        st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>AI Reasoning Log</div>", unsafe_allow_html=True)
        
        # Add CSS for reasoning log styling
        st.markdown("""
        <style>
        .reasoning-log {
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 6px;
            font-size: 0.9rem;
            line-height: 1.4;
            border-left: 3px solid #ccc;
        }
        
        .reasoning-log-thinking {
            background-color: #f0f7ff;
            border-left-color: #2196F3;
        }
        
        .reasoning-log-insight {
            background-color: #f0fff4;
            border-left-color: #4CAF50;
        }
        
        .reasoning-log-recommendation {
            background-color: #fff8e1;
            border-left-color: #FFC107;
        }
        
        .reasoning-log-action {
            background-color: #f5f5f5;
            border-left-color: #9E9E9E;
        }
        
        .reasoning-log-timestamp {
            font-weight: 600;
            color: #555;
            margin-right: 6px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        log_container = st.container()
        with log_container:
            for log_entry in st.session_state.reasoning_log:
                # Access the dictionary fields directly
                timestamp = log_entry["timestamp"]
                message = log_entry["message"]
                entry_type = log_entry["type"]
                
                log_class = "reasoning-log-action"  # Default class
                
                if entry_type == "thinking":
                    log_class = "reasoning-log-thinking"
                elif entry_type == "insight":
                    log_class = "reasoning-log-insight"
                elif entry_type == "recommendation":
                    log_class = "reasoning-log-recommendation"
                
                st.markdown(f"""
                <div class="reasoning-log {log_class}">
                    <span class="reasoning-log-timestamp">[{timestamp}]</span>
                    {message}
                </div>
                """, unsafe_allow_html=True)

    # Main column for dashboard
    with main_col:
        st.title("Machine Learning Playground, Using AI")
        
        if st.session_state.df is not None:
            st.markdown(f"### Dataset: {st.session_state.dataset_name}")
            st.markdown(f"Rows: {st.session_state.df.shape[0]} | Columns: {st.session_state.df.shape[1]}")
            
            # Create tabs for EDA and Preprocessing
            tab_names = ["📊 Exploratory Data Analysis", "🔄 Preprocessing Pipeline", "🧠 ML Pipeline"]
            
            # Check if tab index is stored in session state
            if "active_tab_index" not in st.session_state:
                st.session_state.active_tab_index = 0
            
            # Create tab selection buttons at the top
            st.markdown("""
            <style>
            .tab-button-container {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                border-bottom: 1px solid #e0e0e0;
                padding-bottom: 10px;
            }
            .tab-button {
                background-color: #f0f0f0;
                border: none;
                padding: 10px 20px;
                border-radius: 5px 5px 0 0;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            .tab-button:hover {
                background-color: #e0e0e0;
            }
            .tab-button.active {
                background-color: #4CAF50;
                color: white;
            }
            
            /* Custom tab navigation styling */
            div[data-testid="stHorizontalBlock"] {
                background: #f8f9fa;
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            
            /* Style for the tab buttons */
            div[data-testid="stHorizontalBlock"] button {
                border-radius: 8px;
                border: none;
                padding: 10px 15px;
                font-weight: 600;
                transition: all 0.3s ease;
                width: 100%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }
            
            /* Active tab button style */
            div[data-testid="stHorizontalBlock"] button[data-active="true"] {
                background-color: #4CAF50;
                color: white;
                box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
                transform: translateY(-2px);
            }
            
            /* Inactive tab button style */
            div[data-testid="stHorizontalBlock"] button:not([data-active="true"]) {
                background-color: #f0f0f0;
                color: #333;
            }
            
            /* Hover effect for inactive tab buttons */
            div[data-testid="stHorizontalBlock"] button:not([data-active="true"]):hover {
                background-color: #e0e0e0;
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create custom tab navigation
            st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h3 style="color: #2E7D32; margin-bottom: 10px;">Select a Tab</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                active_class = "active" if st.session_state.active_tab_index == 0 else ""
                if st.button("📊 Exploratory Data Analysis", key="select_eda_tab", help="Switch to Exploratory Data Analysis tab", use_container_width=True):
                    st.session_state.active_tab_index = 0
                    st.session_state.active_tab = "eda"
                    st.experimental_rerun()
            
            with col2:
                active_class = "active" if st.session_state.active_tab_index == 1 else ""
                if st.button("🔄 Preprocessing Pipeline", key="select_preprocessing_tab", help="Switch to Preprocessing Pipeline tab", use_container_width=True):
                    st.session_state.active_tab_index = 1
                    st.session_state.active_tab = "preprocessing"
                    st.experimental_rerun()
            
            with col3:
                active_class = "active" if st.session_state.active_tab_index == 2 else ""
                if st.button("🧠 Machine Learning Pipeline", key="select_ml_tab", help="Switch to Machine Learning Pipeline tab", use_container_width=True):
                    st.session_state.active_tab_index = 2
                    st.session_state.active_tab = "ml"
                    st.experimental_rerun()
            
            # Add a visual indicator for the active tab
            st.markdown(f"""
            <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
                <div style="width: 33%; height: 4px; background-color: {st.session_state.active_tab_index == 0 and '#4CAF50' or 'transparent'};"></div>
                <div style="width: 33%; height: 4px; background-color: {st.session_state.active_tab_index == 1 and '#4CAF50' or 'transparent'};"></div>
                <div style="width: 33%; height: 4px; background-color: {st.session_state.active_tab_index == 2 and '#4CAF50' or 'transparent'};"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create tabs with the current active tab
            tabs = st.tabs(tab_names)
            
            # Set active tab based on session state
            active_tab_index = st.session_state.active_tab_index
            if active_tab_index == 0:
                st.session_state.active_tab = "eda"
            elif active_tab_index == 1:
                st.session_state.active_tab = "preprocessing"
            elif active_tab_index == 2:
                st.session_state.active_tab = "ml"
            
            # EDA Tab
            with tabs[0]:
                st.markdown("<h2 style='font-size: 1.8rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1.5rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Your Dashboard</h2>", unsafe_allow_html=True)
                
                # Check if dashboard is empty
                if not st.session_state.dashboard_plots:
                    st.markdown("""
                    <div style="text-align: center; padding: 3rem; background: linear-gradient(to bottom right, #f8f9fa, #e9ecef); border-radius: 12px; margin: 2rem 0; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                        <svg width="100" height="100" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 1.5rem; opacity: 0.6;">
                            <path d="M3 3V21H21" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M7 14L11 10L15 14L19 10" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <circle cx="11" cy="10" r="1" fill="#4CAF50"/>
                            <circle cx="15" cy="14" r="1" fill="#4CAF50"/>
                            <circle cx="19" cy="10" r="1" fill="#4CAF50"/>
                            <circle cx="7" cy="14" r="1" fill="#4CAF50"/>
                        </svg>
                        <h3 style="color: #2E7D32; font-weight: 500; margin-bottom: 1rem;">Your dashboard is empty</h3>
                        <p style="color: #555; max-width: 400px; margin: 0 auto; line-height: 1.5;">Add plots from the Available Plots section or use AI recommendations to visualize your data and gain insights.</p>
                        <div style="margin-top: 1.5rem;">
                            <span style="display: inline-block; padding: 0.5rem 1rem; background-color: #4CAF50; color: white; border-radius: 20px; font-size: 0.9rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">Start exploring your data</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Create a container for the dashboard plots
                    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                    
                    # Display each plot in the dashboard
                    for i, plot_data in enumerate(st.session_state.dashboard_plots):
                        # Generate a unique key for this plot
                        plot_key = f"dashboard_plot_{i}"
                        
                        # Create a container for this plot
                        st.markdown(f'<div class="dashboard-plot" id="{plot_key}">', unsafe_allow_html=True)
                        
                        # Check if plot_data is a string (plot_id) or a dictionary
                        if isinstance(plot_data, str):
                            # It's a plot_id from the old format
                            if plot_data in st.session_state.plot_configs:
                                plot_config = st.session_state.plot_configs[plot_data]
                                plot_type = plot_config["type"]
                                plot_config_data = plot_config["config"]
                            else:
                                st.error(f"Plot configuration not found for ID: {plot_data}")
                                continue
                        else:
                            # It's already a dictionary with type and config
                            plot_type = plot_data["type"]
                            plot_config_data = plot_data["config"]
                        
                        # Plot header with title and actions
                        st.markdown(f"""
                        <div class="plot-header">
                            <div class="plot-title">{plot_type.capitalize()}</div>
                            <div class="plot-actions">
                                <button onclick="document.getElementById('customize_{i}').style.display = document.getElementById('customize_{i}').style.display === 'none' ? 'block' : 'none';">
                                    ⚙️
                                </button>
                                <button onclick="document.getElementById('insight_{i}').style.display = document.getElementById('insight_{i}').style.display === 'none' ? 'block' : 'none';">
                                    💡
                                </button>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add a direct remove button using Streamlit
                        if st.button("❌ Remove Plot", key=f"remove_plot_{i}"):
                            remove_plot_from_dashboard(i)
                            st.experimental_rerun()
                        
                        # Generate the plot
                        plot_result = generate_plot(st.session_state.df, plot_type, plot_config_data)
                        
                        # Display the plot
                        if "error" in plot_result:
                            st.error(plot_result["error"])
                        else:
                            st.plotly_chart(plot_result["figure"], use_container_width=True)
                        
                        # Customization panel (hidden by default)
                        st.markdown(f"""
                        <div id="customize_{i}" style="display: none; padding: 1rem; background-color: #f8f9fa; border-radius: 8px; margin-top: 1rem;">
                            <h4>Customize Plot</h4>
                            <p>Customization options will appear here.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Insight panel (hidden by default)
                        st.markdown(f"""
                        <div id="insight_{i}" style="display: none; padding: 1rem; background-color: #f8f9fa; border-radius: 8px; margin-top: 1rem;">
                            <h4>AI Insight</h4>
                            <p>Click the button below to generate an AI insight for this plot.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Generate insight button
                        if st.button("Generate Insight", key=f"insight_btn_{i}"):
                            with st.spinner("Generating insight..."):
                                # Get the plot data
                                if isinstance(plot_data, str):
                                    # It's a plot_id from the old format
                                    if plot_data in st.session_state.plot_configs:
                                        plot_config = st.session_state.plot_configs[plot_data]
                                        plot_type = plot_config["type"]
                                        plot_config_data = plot_config["config"]
                                    else:
                                        st.error(f"Plot configuration not found for ID: {plot_data}")
                                        continue
                                else:
                                    # It's already a dictionary with type and config
                                    plot_type = plot_data["type"]
                                    plot_config_data = plot_data["config"]
                                
                                # Add thinking logs
                                add_to_log(f"User requested insight for {plot_type} plot")
                                add_to_log(f"Analyzing {plot_type} visualization data patterns", is_thinking=True)
                                
                                # Add specific thinking based on plot type
                                if plot_type == 'histogram':
                                    add_to_log("Examining distribution characteristics: central tendency, spread, and shape", is_thinking=True)
                                    add_to_log("Checking for normality, skewness, and potential outliers", is_thinking=True)
                                elif plot_type == 'scatter':
                                    add_to_log("Analyzing relationship between variables: direction, strength, and form", is_thinking=True)
                                    add_to_log("Looking for clusters, trends, and potential outliers", is_thinking=True)
                                elif plot_type == 'boxplot':
                                    add_to_log("Examining distribution statistics: median, quartiles, and range", is_thinking=True)
                                    add_to_log("Identifying potential outliers and comparing distributions", is_thinking=True)
                                elif plot_type == 'correlation_heatmap':
                                    add_to_log("Identifying strong positive and negative correlations", is_thinking=True)
                                    add_to_log("Looking for patterns and clusters of related variables", is_thinking=True)
                                elif plot_type == 'bar':
                                    add_to_log("Comparing values across categories and identifying key patterns", is_thinking=True)
                                elif plot_type == 'pie':
                                    add_to_log("Analyzing proportional relationships and dominant categories", is_thinking=True)
                                else:
                                    add_to_log(f"Analyzing patterns and trends in the {plot_type} visualization", is_thinking=True)
                                
                                # Generate the plot
                                plot_result = generate_plot(st.session_state.df, plot_type, plot_config_data)
                                
                                # Generate insight
                                add_to_log("Formulating data-driven insights based on visualization patterns", is_thinking=True)
                                insight = st.session_state.ai_agent.get_eda_agent().generate_insight(
                                    st.session_state.df,
                                    plot_result,
                                    st.session_state.dataset_name
                                )
                                
                                # Log the insight
                                insight_summary = insight.split('.')[0] + '.' if '.' in insight else insight
                                add_to_log(f"Key insight: {insight_summary}", is_insight=True)
                                
                                st.markdown(f"""
                                <div class="insight-container">
                                    <h4>AI Insight</h4>
                                    {insight}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Close the plot container
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Close the dashboard container
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Preprocessing Tab
            with tabs[1]:
                st.markdown("<h2 style='font-size: 1.8rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1.5rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Preprocessing Workflow</h2>", unsafe_allow_html=True)
                
                # Step configuration area
                if "configuring_step" in st.session_state and st.session_state.configuring_step:
                    step_id = st.session_state.configuring_step
                    
                    # Get step configuration UI
                    prefill_config = None
                    if "prefill_config" in st.session_state:
                        prefill_config = st.session_state.prefill_config
                        # Clear prefill after using it
                        del st.session_state.prefill_config
                    
                    step_config = get_step_config_ui(step_id, st.session_state.df, prefill_config)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("Add to Workflow", key="confirm_step"):
                            add_preprocessing_step(step_config)
                            st.session_state.configuring_step = None
                            st.experimental_rerun()
                    
                    with col2:
                        if st.button("Cancel", key="cancel_step"):
                            st.session_state.configuring_step = None
                            st.experimental_rerun()
                
                # Display current workflow
                else:
                    # Workflow visualization
                    st.markdown("<div class='workflow-container'>", unsafe_allow_html=True)
                    
                    if not st.session_state.preprocessing_workflow:
                        st.info("No preprocessing steps added yet. Select steps from the left sidebar to build your workflow.")
                    else:
                        # Render workflow and get index to remove if any
                        remove_idx = render_workflow(st.session_state.preprocessing_workflow, st.session_state.df)
                        
                        # Handle step removal
                        if remove_idx is not None:
                            st.session_state.preprocessing_workflow.pop(remove_idx)
                            add_preprocessing_message(f"Removed step from workflow")
                            st.experimental_rerun()
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Apply workflow button
                    if st.session_state.preprocessing_workflow:
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            if st.button("Apply Preprocessing", key="apply_workflow"):
                                with st.spinner("Applying preprocessing workflow..."):
                                    success = apply_preprocessing_workflow()
                                    if success:
                                        st.success("Preprocessing workflow applied successfully!")
                        
                        with col2:
                            if st.button("Evaluate Workflow", key="evaluate_workflow"):
                                with st.spinner("Evaluating preprocessing workflow..."):
                                    score, feedback = evaluate_preprocessing_workflow()
                                    if score is not None:
                                        st.session_state.workflow_evaluation = {
                                            "score": score,
                                            "feedback": feedback
                                        }
                                        st.success(f"Workflow evaluated! Score: {int(score * 10)}/10")
                                        st.experimental_rerun()
                        
                        with col3:
                            if st.button("Clear Workflow", key="clear_workflow"):
                                st.session_state.preprocessing_workflow = []
                                st.session_state.processed_df = None
                                if "workflow_evaluation" in st.session_state:
                                    del st.session_state.workflow_evaluation
                                add_preprocessing_message("Cleared preprocessing workflow")
                                st.experimental_rerun()
                        
                        # Display workflow evaluation if available
                        if "workflow_evaluation" in st.session_state:
                            eval_score = st.session_state.workflow_evaluation["score"]
                            eval_feedback = st.session_state.workflow_evaluation["feedback"]
                            
                            st.markdown("<div class='sidebar-header'>Workflow Evaluation</div>", unsafe_allow_html=True)
                            
                            # Display score with a progress bar
                            score_percentage = int(eval_score * 100)
                            st.progress(eval_score)
                            st.markdown(f"**Score: {int(eval_score * 10)}/10** ({score_percentage}%)")
                            
                            # Display feedback
                            st.markdown("### Feedback")
                            st.markdown(eval_feedback)
                        
                        # Export options
                        if st.session_state.preprocessing_workflow:
                            st.markdown("<div class='sidebar-header'>Export Workflow</div>", unsafe_allow_html=True)
                            
                            export_col1, export_col2, export_col3 = st.columns(3)
                            
                            with export_col1:
                                if st.button("Export JSON"):
                                    json_data = export_workflow_to_json()
                                    if json_data:
                                        st.download_button(
                                            label="Download JSON",
                                            data=json_data,
                                            file_name=f"preprocessing_workflow_{st.session_state.dataset_name}.json",
                                            mime="application/json"
                                        )
                            
                            with export_col2:
                                if st.button("Export Python"):
                                    python_code = export_workflow_to_python()
                                    if python_code:
                                        st.download_button(
                                            label="Download Python",
                                            data=python_code,
                                            file_name=f"preprocessing_workflow_{st.session_state.dataset_name}.py",
                                            mime="text/plain"
                                        )
                            
                            with export_col3:
                                if st.button("Export PDF"):
                                    pdf_data = export_workflow_to_pdf()
                                    if pdf_data:
                                        st.download_button(
                                            label="Download PDF",
                                            data=pdf_data,
                                            file_name=f"preprocessing_workflow_{st.session_state.dataset_name}.pdf",
                                            mime="application/pdf"
                                        )
                        
                        # Display processed data if available
                        if st.session_state.processed_df is not None:
                            st.markdown("<div class='sidebar-header'>Processed Dataset</div>", unsafe_allow_html=True)
                            
                            # Display processed data info
                            original_shape = st.session_state.df.shape
                            processed_shape = st.session_state.processed_df.shape
                            
                            st.markdown(f"Original: {original_shape[0]} rows, {original_shape[1]} columns")
                            st.markdown(f"Processed: {processed_shape[0]} rows, {processed_shape[1]} columns")
                            
                            # Display sample of processed data
                            st.dataframe(st.session_state.processed_df.head(10))
                            
                            # Option to download processed data
                            csv = st.session_state.processed_df.to_csv(index=False)
                            st.download_button(
                                label="Download Processed Data",
                                data=csv,
                                file_name=f"processed_{st.session_state.dataset_name}.csv",
                                mime="text/csv"
                            )
                            
                            # Compare with original data
                            with st.expander("Compare with Original Data"):
                                # Show column differences
                                original_cols = set(st.session_state.df.columns)
                                processed_cols = set(st.session_state.processed_df.columns)
                                
                                added_cols = processed_cols - original_cols
                                removed_cols = original_cols - processed_cols
                                
                                if added_cols:
                                    st.markdown("##### Added Columns")
                                    for col in added_cols:
                                        st.markdown(f"- {col}")
                                
                                if removed_cols:
                                    st.markdown("##### Removed Columns")
                                    for col in removed_cols:
                                        st.markdown(f"- {col}")
                                
                                # Show row count difference
                                row_diff = processed_shape[0] - original_shape[0]
                                if row_diff != 0:
                                    st.markdown(f"##### Row Count Change: {row_diff}")
                                    if row_diff < 0:
                                        st.markdown(f"{abs(row_diff)} rows were removed")
                                    else:
                                        st.markdown(f"{row_diff} rows were added")
                            
                            # Display preprocessing messages
                            if st.session_state.preprocessing_messages:
                                st.markdown("<div class='sidebar-header'>Preprocessing Log</div>", unsafe_allow_html=True)
                                
                                for msg in st.session_state.preprocessing_messages:
                                    if msg["is_error"]:
                                        st.error(f"[{msg['timestamp']}] {msg['message']}")
                                    else:
                                        st.markdown(f"<div class='message-container'>[{msg['timestamp']}] {msg['message']}</div>", unsafe_allow_html=True)
            
            # ML Pipeline Tab
            with tabs[2]:
                render_ml_pipeline_tab()

    # Right column for AI recommendations
    with right_col:
        if st.session_state.df is not None:
            # Display content based on active tab
            if st.session_state.active_tab == "eda":
                # EDA recommendations
                st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>AI Visualization Recommendations</div>", unsafe_allow_html=True)
                
                if "recommended_visualizations" in st.session_state and st.session_state.recommended_visualizations:
                    for i, recommendation in enumerate(st.session_state.recommended_visualizations):
                        plot_type = recommendation["type"]
                        reason = recommendation["reason"]
                        config = recommendation["config"]
                        
                        # Create a unique key for this recommendation
                        rec_key = f"recommendation_{i}"
                        
                        # Display recommendation in a card
                        st.markdown(f"""
                        <div class="ai-recommendation">
                            <div class="ai-recommendation-title">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                                {plot_type.capitalize()} Visualization
                                <span class="viz-badge {plot_type}">{plot_type}</span>
                            </div>
                            <div class="ai-recommendation-reason">{reason}</div>
                        """, unsafe_allow_html=True)
                        
                        # Add button to add to dashboard
                        if st.button(f"Add to Dashboard", key=f"add_rec_{i}"):
                            add_plot_to_dashboard(plot_type, config)
                            st.experimental_rerun()
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    if st.session_state.df is not None:
                        st.info("Load a dataset to see AI visualization recommendations.")
            
            elif st.session_state.active_tab == "preprocessing":
                # Preprocessing recommendations
                st.subheader("AI Preprocessing Recommendations")
                st.markdown("---")
                
                # Button to get preprocessing suggestions
                if st.button("Get AI Preprocessing Suggestions", key="get_right_preprocessing_suggestions"):
                    with st.spinner("Analyzing data and generating preprocessing suggestions..."):
                        # Detect data issues
                        issues = detect_data_issues(st.session_state.df)
                        
                        # Get preprocessing suggestions from the agent
                        suggestions = st.session_state.preprocessing_agent.suggest_preprocessing_steps(
                            st.session_state.df, issues
                        )
                        
                        # Store suggestions in session state
                        st.session_state.preprocessing_suggestions = suggestions
                        
                        add_preprocessing_message("Generated AI preprocessing suggestions")
                        st.experimental_rerun()
                
                # Display AI suggestions if available
                if "preprocessing_suggestions" in st.session_state and st.session_state.preprocessing_suggestions:
                    # Group suggestions by step type
                    grouped_suggestions = {}
                    for suggestion in st.session_state.preprocessing_suggestions:
                        step_type = suggestion.get("step", "")
                        if step_type not in grouped_suggestions:
                            grouped_suggestions[step_type] = []
                        grouped_suggestions[step_type].append(suggestion)
                    
                    # Display grouped suggestions
                    for step_type, suggestions in grouped_suggestions.items():
                        step_info = PREPROCESSING_STEPS.get(step_type, {})
                        
                        # Create an expander for each group
                        with st.expander(f"{step_info.get('icon', '🔧')} {step_info.get('name', step_type)} ({len(suggestions)})", expanded=True):
                            # Display each suggestion in the group
                            for i, suggestion in enumerate(suggestions):
                                # Create a clean card-like container for each suggestion
                                st.markdown(f"""
                                <div style="border-left: 4px solid #4CAF50; padding: 10px; margin-bottom: 15px; background-color: #f9f9f9; border-radius: 4px;">
                                    <div style="font-weight: 600; margin-bottom: 8px; color: #2E7D32;">Suggestion {i+1}</div>
                                """, unsafe_allow_html=True)
                                
                                # Display column information if available
                                if "column" in suggestion:
                                    st.markdown(f"**Column:** {suggestion['column']}", unsafe_allow_html=False)
                                elif "columns" in suggestion and isinstance(suggestion["columns"], list):
                                    st.markdown(f"**Columns:** {', '.join(suggestion['columns'])}", unsafe_allow_html=False)
                                
                                # Display method information if available
                                if "method" in suggestion:
                                    method_name = step_info.get("methods", {}).get(suggestion["method"], suggestion["method"])
                                    st.markdown(f"**Method:** {method_name}", unsafe_allow_html=False)
                                
                                # Display reason
                                if "reason" in suggestion:
                                    st.markdown(f"**Reason:** {suggestion.get('reason', '')}", unsafe_allow_html=False)
                                
                                # Add button to configure this step
                                if st.button(f"Configure", key=f"config_rec_{step_type}_{i}"):
                                    st.session_state.configuring_step = step_type
                                    # Pre-fill configuration with suggestion
                                    st.session_state.prefill_config = suggestion
                                    st.experimental_rerun()
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Add a "Configure All" button if there are multiple suggestions of the same type
                            if len(suggestions) > 1 and step_type in ["handle_correlation", "impute_missing", "handle_outliers", "transform", "scale", "drop_column"]:
                                st.markdown("### Configure All Together")
                                
                                # Get all columns from suggestions
                                all_columns = []
                                for suggestion in suggestions:
                                    if "column" in suggestion:
                                        all_columns.append(suggestion["column"])
                                    elif "columns" in suggestion and isinstance(suggestion["columns"], list):
                                        all_columns.extend(suggestion["columns"])
                                
                                # Remove duplicates
                                all_columns = list(set(all_columns))
                                
                                # Create a combined suggestion
                                combined_suggestion = {
                                    "step": step_type,
                                    "columns": all_columns,
                                    "reason": f"Combined {len(suggestions)} AI suggestions for {step_info.get('name', step_type)}"
                                }
                                
                                # If all suggestions have the same method, use it
                                methods = [s.get("method") for s in suggestions if "method" in s]
                                if methods and all(m == methods[0] for m in methods):
                                    combined_suggestion["method"] = methods[0]
                                
                                if st.button(f"Configure All {step_info.get('name', step_type)}", key=f"config_all_{step_type}"):
                                    st.session_state.configuring_step = step_type
                                    # Pre-fill configuration with combined suggestion
                                    st.session_state.prefill_config = combined_suggestion
                                    st.experimental_rerun()
                else:
                    st.info("Click 'Get AI Preprocessing Suggestions' to analyze your data and get recommendations for preprocessing steps.")
                
                # Display data issues section
                if "analysis" in st.session_state and st.session_state.analysis and "issues" in st.session_state.analysis:
                    issues = st.session_state.analysis["issues"]
                    if any(issues.values()):
                        st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Data Issues</div>", unsafe_allow_html=True)
                        
                        if issues.get("missing_values"):
                            st.markdown("**Missing Values:**")
                            for col, pct in issues["missing_values"].items():
                                st.markdown(f"- {col}: {pct:.1f}%")
                        
                        if issues.get("outliers"):
                            st.markdown("**Outliers Detected:**")
                            for col in issues["outliers"]:
                                st.markdown(f"- {col}")
                        
                        if issues.get("high_correlation"):
                            st.markdown("**High Correlations:**")
                            for corr in issues["high_correlation"][:5]:  # Show top 5
                                st.markdown(f"- {corr['col1']} & {corr['col2']}: {corr['correlation']:.2f}")
            
            elif st.session_state.active_tab == "ml":
                # ML Pipeline recommendations
                st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>ML Pipeline Recommendations</div>", unsafe_allow_html=True)
                
                st.info("ML Pipeline recommendations will be available in Phase 3.")
            
            # Add dataset insights section if available (shown in all tabs)
            if "analysis" in st.session_state and st.session_state.analysis:
                st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Dataset Insights</div>", unsafe_allow_html=True)
                
                analysis = st.session_state.analysis
                
                # Display dataset summary
                if "summary" in analysis:
                    with st.expander("Dataset Summary", expanded=True):
                        summary = analysis["summary"]
                        st.markdown(f"**Rows:** {summary.get('rows', 'N/A')}")
                        st.markdown(f"**Columns:** {summary.get('columns', 'N/A')}")
                        
                        if "column_types" in summary:
                            st.markdown("**Column Types:**")
                            for col_type, count in summary["column_types"].items():
                                st.markdown(f"- {col_type}: {count}")

def train_and_evaluate_models(
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    selected_models, 
    problem_type, 
    cv_folds=5, 
    hyperparameter_tuning=False
):
    """
    Train and evaluate multiple models.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training targets
        y_test: Testing targets
        selected_models: List of model IDs to train
        problem_type: 'classification' or 'regression'
        cv_folds: Number of cross-validation folds
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        trained_models, evaluation_results, plots
    """
    from ai_eda_pipeline.components.ml_pipeline import (
        get_model_instance, 
        train_and_evaluate_model, 
        generate_model_evaluation_plots,
        get_feature_importance,
        perform_cross_validation,
        get_model_hyperparameters,
        perform_hyperparameter_tuning
    )
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Initialize results
    trained_models = {}
    evaluation_results = {}
    all_plots = {}
    
    # Get feature names (try to use meaningful names if possible)
    try:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    except Exception as e:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1] if hasattr(X_train, 'shape') else 0)]
    
    # Create a default feature importance plot in case we can't generate one
    default_feature_importance = px.bar(
        x=["No feature importance available"], 
        y=[0],
        title="Feature Importance Not Available"
    )
    default_feature_importance.update_layout(
        xaxis_title="Feature",
        yaxis_title="Importance",
        showlegend=False
    )
    
    # Create a default confusion matrix in case we can't generate one
    default_confusion_matrix = px.imshow(
        [[0, 0], [0, 0]],
        labels=dict(x="Predicted", y="Actual"),
        x=["Class 0", "Class 1"],
        y=["Class 0", "Class 1"],
        title="Confusion Matrix Not Available"
    )
    
    # Create a default regression plot in case we can't generate one
    default_regression_plot = px.scatter(
        x=[0], y=[0],
        labels={"x": "Actual", "y": "Predicted"},
        title="Actual vs Predicted Values Not Available"
    )
    
    # Process each selected model
    for model_id in selected_models:
        try:
            # Get model instance
            model = get_model_instance(model_id, problem_type)
            
            # Perform hyperparameter tuning if enabled
            if hyperparameter_tuning:
                try:
                    param_grid = get_model_hyperparameters(model_id)
                    if param_grid:
                        tuning_results = perform_hyperparameter_tuning(
                            model_id, X_train, y_train, param_grid, cv=cv_folds, problem_type=problem_type
                        )
                        model = tuning_results["best_model"]
                except Exception as e:
                    # If hyperparameter tuning fails, continue with default model
                    print(f"Hyperparameter tuning failed for {model_id}: {str(e)}")
            
            # Train and evaluate model
            results = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, problem_type)
            
            # Generate evaluation plots
            try:
                plots = generate_model_evaluation_plots(results, problem_type, feature_names)
            except Exception as e:
                # If plot generation fails, use default plots
                print(f"Plot generation failed for {model_id}: {str(e)}")
                plots = {}
                if problem_type == "classification":
                    plots["confusion_matrix"] = default_confusion_matrix
                else:
                    plots["actual_vs_predicted"] = default_regression_plot
            
            # Get feature importance if available
            try:
                feature_importance = get_feature_importance(model, feature_names)
            except Exception as e:
                # If feature importance fails, use default
                print(f"Feature importance failed for {model_id}: {str(e)}")
                feature_importance = {"feature_importance": {}, "plot": default_feature_importance}
            
            # Perform cross-validation
            try:
                cv_results = perform_cross_validation(model, X_train, y_train, cv=cv_folds, problem_type=problem_type)
            except Exception as e:
                # If cross-validation fails, use default values
                print(f"Cross-validation failed for {model_id}: {str(e)}")
                cv_results = {
                    "cv_scores": [0.0] * cv_folds,
                    "mean_score": 0.0,
                    "std_score": 0.0,
                    "plot": None
                }
            
            # Store results
            trained_models[model_id] = {
                "model": model,
                "metrics": results["metrics"],
                "cv_results": cv_results
            }
            
            # Store evaluation results
            evaluation_results[model_id] = results["metrics"]
            
            # Store plots
            for plot_name, plot in plots.items():
                if plot_name not in all_plots:
                    all_plots[plot_name] = {}
                all_plots[plot_name][model_id] = plot
            
            # Add feature importance plot if available
            if feature_importance and "plot" in feature_importance and feature_importance["plot"] is not None:
                if "feature_importance" not in all_plots:
                    all_plots["feature_importance"] = {}
                all_plots["feature_importance"][model_id] = feature_importance["plot"]
        
        except Exception as e:
            # If model training fails, log the error and continue with other models
            print(f"Model training failed for {model_id}: {str(e)}")
            
            # Create dummy metrics for the failed model
            dummy_metrics = {}
            if problem_type == "classification":
                dummy_metrics = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                }
            else:  # regression
                dummy_metrics = {
                    "mse": float('inf'),
                    "rmse": float('inf'),
                    "mae": float('inf'),
                    "r2": 0.0
                }
            
            # Add a note about the failure
            dummy_metrics["error"] = str(e)
            
            # Store dummy results
            evaluation_results[model_id] = dummy_metrics
    
    # Process plots to create combined visualizations
    processed_plots = {}
    
    # For classification, create confusion matrix
    if problem_type == "classification" and "confusion_matrix" in all_plots and all_plots["confusion_matrix"]:
        try:
            # Just use the first model's confusion matrix for now
            first_model_id = list(all_plots["confusion_matrix"].keys())[0]
            processed_plots["confusion_matrix"] = all_plots["confusion_matrix"][first_model_id]
        except Exception as e:
            processed_plots["confusion_matrix"] = default_confusion_matrix
    elif problem_type == "classification":
        processed_plots["confusion_matrix"] = default_confusion_matrix
    
    # For regression, create actual vs predicted plot
    if problem_type == "regression" and "actual_vs_predicted" in all_plots and all_plots["actual_vs_predicted"]:
        try:
            # Just use the first model's plot for now
            first_model_id = list(all_plots["actual_vs_predicted"].keys())[0]
            processed_plots["actual_vs_predicted"] = all_plots["actual_vs_predicted"][first_model_id]
        except Exception as e:
            processed_plots["actual_vs_predicted"] = default_regression_plot
    elif problem_type == "regression":
        processed_plots["actual_vs_predicted"] = default_regression_plot
    
    # Create feature importance plot
    if "feature_importance" in all_plots and all_plots["feature_importance"]:
        try:
            # Just use the first model's feature importance for now
            first_model_id = list(all_plots["feature_importance"].keys())[0]
            processed_plots["feature_importance"] = all_plots["feature_importance"][first_model_id]
        except Exception as e:
            processed_plots["feature_importance"] = default_feature_importance
    else:
        processed_plots["feature_importance"] = default_feature_importance
    
    return trained_models, evaluation_results, processed_plots

def identify_best_model(evaluation_results, problem_type):
    """
    Identify the best model based on evaluation metrics.
    
    Args:
        evaluation_results: Dictionary of model_id -> metrics
        problem_type: 'classification' or 'regression'
        
    Returns:
        ID of the best model
    """
    if not evaluation_results:
        return None
    
    # Filter out models with errors
    valid_models = {k: v for k, v in evaluation_results.items() if "error" not in v}
    
    # If no valid models, return None
    if not valid_models:
        return None
    
    try:
        # Choose metric based on problem type
        if problem_type == "classification":
            # For classification, prioritize balanced metrics
            if all("balanced_accuracy" in metrics for model_id, metrics in valid_models.items()):
                metric = "balanced_accuracy"
            elif all("f1" in metrics for model_id, metrics in valid_models.items()):
                metric = "f1"
            else:
                metric = "accuracy"
            
            # Find model with highest metric
            best_model_id = max(valid_models.items(), key=lambda x: x[1].get(metric, 0))[0]
        else:  # regression
            # For regression, prioritize R² but also consider RMSE
            if all("r2" in metrics for model_id, metrics in valid_models.items()):
                # Higher R² is better
                best_model_id = max(valid_models.items(), key=lambda x: x[1].get("r2", 0))[0]
            elif all("rmse" in metrics for model_id, metrics in valid_models.items()):
                # Lower RMSE is better
                best_model_id = min(valid_models.items(), key=lambda x: x[1].get("rmse", float('inf')))[0]
            else:
                # Use MSE as fallback
                best_model_id = min(valid_models.items(), key=lambda x: x[1].get("mse", float('inf')))[0]
        
        return best_model_id
    except Exception as e:
        # If there's an error, just return the first model ID
        print(f"Error identifying best model: {str(e)}")
        return list(valid_models.keys())[0] if valid_models else None

# Run the application
if __name__ == "__main__":
    main() 