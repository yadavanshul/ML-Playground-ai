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


