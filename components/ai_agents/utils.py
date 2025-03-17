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

