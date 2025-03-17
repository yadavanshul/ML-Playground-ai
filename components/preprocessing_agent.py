import os
import json
import time
import openai
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

class PreprocessingMiniAgent:
    """
    Mini AI agent for preprocessing recommendations and evaluation.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the preprocessing mini agent.
        
        Args:
            openai_api_key: OpenAI API key (optional)
        """
        self.openai_api_key = openai_api_key
        self.use_mock = not bool(openai_api_key)
        
        if openai_api_key:
            openai.api_key = openai_api_key
    
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
        
        # Get basic suggestions from data_utils
        suggestions = suggest_preprocessing_steps(df, issues)
        
        # If we have ChromaDB initialized and not in mock mode, try to store the suggestions
        try:
            if not self.use_mock and 'chroma_client' in globals():
                # Store the suggestions in the database for future reference
                chroma_client.add(
                    ids=[f"preprocessing_suggestion_{int(time.time())}"],
                    documents=[json.dumps(suggestions)],
                    metadatas=[{
                        "type": "preprocessing_suggestion",
                        "dataset_shape": str(df.shape),
                        "timestamp": time.time()
                    }]
                )
        except Exception as e:
            print(f"Error storing preprocessing suggestions: {str(e)}")
        
        return suggestions
    
    def evaluate_preprocessing_pipeline(self, steps: List[Dict]) -> Tuple[float, str]:
        """
        Evaluate a preprocessing pipeline and provide feedback.
        
        Args:
            steps: List of preprocessing steps
            
        Returns:
            Tuple of (score, feedback)
        """
        if not steps:
            return 0.0, "No preprocessing steps provided."
        
        if self.use_mock:
            return self._generate_mock_evaluation(steps)
        
        # Create a prompt for evaluation
        prompt = f"""
        Evaluate the following preprocessing pipeline for data preparation:
        
        {json.dumps(steps, indent=2)}
        
        Provide feedback on:
        1. Completeness: Does it address common preprocessing needs?
        2. Order: Are steps in a logical order?
        3. Appropriateness: Are the methods suitable for the data types?
        4. Potential issues: Any concerns or missing steps?
        
        Also provide a score from 0-10 where 10 is excellent.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data science expert evaluating preprocessing pipelines."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            feedback = response.choices[0].message.content.strip()
            
            # Extract score from feedback
            score = 5.0  # Default middle score
            
            # Look for patterns like "Score: 8/10" or "I give this a 7 out of 10"
            import re
            score_patterns = [
                r"(\d+(\.\d+)?)\s*\/\s*10",  # 8/10
                r"score\s*:?\s*(\d+(\.\d+)?)",  # Score: 8
                r"(\d+(\.\d+)?)\s*out of\s*10",  # 8 out of 10
                r"rating\s*:?\s*(\d+(\.\d+)?)",  # Rating: 8
            ]
            
            for pattern in score_patterns:
                matches = re.search(pattern, feedback, re.IGNORECASE)
                if matches:
                    try:
                        score = float(matches.group(1))
                        break
                    except:
                        pass
            
            return score / 10.0, feedback  # Normalize to 0-1 range
            
        except Exception as e:
            return 0.3, f"Error evaluating pipeline: {str(e)}"
    
    def _generate_mock_evaluation(self, steps: List[Dict]) -> Tuple[float, str]:
        """
        Generate a mock evaluation when OpenAI API is not available.
        
        Args:
            steps: List of preprocessing steps
            
        Returns:
            Tuple of (score, feedback)
        """
        # Count different types of preprocessing steps
        step_types = [step.get("step") for step in steps]
        
        has_missing_handling = "impute_missing" in step_types
        has_outlier_handling = "handle_outliers" in step_types
        has_encoding = "encode" in step_types
        has_scaling = "scale" in step_types
        has_feature_selection = "drop_column" in step_types or "handle_correlation" in step_types
        
        # Generate feedback based on steps present
        feedback_parts = []
        
        if has_missing_handling:
            feedback_parts.append("✓ Good job handling missing values.")
        else:
            feedback_parts.append("✗ Consider adding steps to handle missing values.")
        
        if has_outlier_handling:
            feedback_parts.append("✓ Outlier handling is included.")
        else:
            feedback_parts.append("✗ The pipeline might benefit from outlier detection and handling.")
        
        if has_encoding:
            feedback_parts.append("✓ Categorical encoding is properly addressed.")
        else:
            feedback_parts.append("✗ If there are categorical features, consider adding encoding steps.")
        
        if has_scaling:
            feedback_parts.append("✓ Feature scaling is included, which is important for many algorithms.")
        else:
            feedback_parts.append("✗ Consider adding feature scaling for better model performance.")
        
        if has_feature_selection:
            feedback_parts.append("✓ The pipeline includes feature selection/reduction steps.")
        else:
            feedback_parts.append("✗ Consider adding feature selection to improve model efficiency.")
        
        # Calculate a mock score based on coverage
        covered_aspects = sum([has_missing_handling, has_outlier_handling, has_encoding, has_scaling, has_feature_selection])
        score = 0.4 + (covered_aspects / 5.0) * 0.5  # Score between 0.4 and 0.9
        
        # Add general feedback
        feedback = f"""
        Preprocessing Pipeline Evaluation:
        
        {' '.join(feedback_parts)}
        
        The pipeline contains {len(steps)} steps, covering {covered_aspects}/5 key preprocessing aspects.
        
        Score: {int(score * 10)}/10
        """
        
        return score, feedback 