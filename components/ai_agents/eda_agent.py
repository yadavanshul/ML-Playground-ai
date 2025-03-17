class EDAMiniAgent:
    """
    Mini AI agent specialized in exploratory data analysis.
    """
    
    def __init__(self, api_key=None):
        # Configure OpenAI API key if provided
        self.api_key = api_key
        if is_valid_api_key(api_key):
            self.use_mock = False
        else:
            self.use_mock = USE_MOCK
    
    def generate_insight(self, df: pd.DataFrame, plot_data: Dict, dataset_name: str) -> str:
        """
        Generate an insight for a specific visualization.
        
        Args:
            df: Input DataFrame
            plot_data: Plot data including statistics
            dataset_name: Name of the dataset
            
        Returns:
            String with insight
        """
        if "error" in plot_data:
            return f"Error: {plot_data['error']}"
        
        plot_type = plot_data.get("plot_type", "unknown")
        stats = plot_data.get("stats", {})
        
        # If using mock functionality, return a generic insight
        if self.use_mock:
            return self._generate_mock_insight(plot_type, stats)
        
        # Create a prompt based on the plot type and statistics
        prompt = f"Generate a concise, data-driven insight about the following {plot_type} visualization for the dataset '{dataset_name}'.\n\n"
        prompt += "Statistics:\n"
        
        for key, value in stats.items():
            if isinstance(value, dict):
                prompt += f"- {key}: {json.dumps(value)[:100]}...\n"
            else:
                prompt += f"- {key}: {value}\n"
        
        prompt += "\nProvide a concise, factual insight based ONLY on these statistics. Do not make assumptions beyond what the data shows."
        prompt += "\nFocus on patterns, outliers, distributions, or correlations as appropriate for this visualization type."
        prompt += "\nKeep the insight to 2-3 sentences maximum."
        
        # Get similar insights from ChromaDB for context if initialized
        if CHROMA_INITIALIZED:
            try:
                similar_insights = insights_collection.query(
                    query_texts=[prompt],
                    n_results=3
                )
                
                if similar_insights["ids"][0]:
                    prompt += "\n\nHere are some similar insights for context (but generate a new insight specific to this data):\n"
                    for insight in similar_insights["documents"][0]:
                        prompt += f"- {insight}\n"
            except Exception as e:
                logger.error(f"Error querying insights collection: {str(e)}")
        
        # Generate insight using OpenAI
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analysis assistant that provides concise, factual insights based only on the statistics provided. Do not make assumptions beyond what the data shows."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.5
            )
            
            insight = response.choices[0].message.content.strip()
            
            # Store in ChromaDB if initialized
            if CHROMA_INITIALIZED:
                try:
                    insight_id = f"{dataset_name}_{plot_type}_{int(time.time())}"
                    insights_collection.upsert(
                        ids=[insight_id],
                        documents=[insight],
                        metadatas=[{
                            "dataset": dataset_name,
                            "plot_type": plot_type,
                            "timestamp": time.time()
                        }]
                    )
                except Exception as e:
                    logger.error(f"Error storing insight in ChromaDB: {str(e)}")
            
            return insight
        
        except Exception as e:
            logger.error(f"Error generating insight with OpenAI: {str(e)}")
            return self._generate_mock_insight(plot_type, stats)
    
    def _generate_mock_insight(self, plot_type: str, stats: Dict) -> str:
        """
        Generate a mock insight when OpenAI API is not available.
        
        Args:
            plot_type: Type of plot
            stats: Statistics from the plot
            
        Returns:
            String with mock insight
        """
        if plot_type == "histogram":
            if "mean" in stats and "median" in stats:
                mean = stats.get("mean", 0)
                median = stats.get("median", 0)
                skew_direction = "right" if mean > median else "left" if mean < median else "not"
                return f"The distribution is {skew_direction} skewed with a mean of {mean:.2f} and median of {median:.2f}. The data shows a central tendency around these values."
            return "This histogram shows the distribution of values across different bins. The height of each bar represents the frequency or count of observations in that bin."
        
        elif plot_type == "boxplot":
            return "This boxplot shows the distribution of the data through quartiles. The box represents the interquartile range (IQR), with the median shown as a line inside the box. Whiskers extend to show the range of the data, and points beyond the whiskers may represent outliers."
        
        elif plot_type == "scatter":
            return "This scatter plot shows the relationship between two variables. Each point represents an observation with its position determined by the values of the two variables. Patterns in the scatter plot may indicate correlation or other relationships between the variables."
        
        elif plot_type == "correlation_heatmap":
            return "This correlation heatmap shows the strength of relationships between pairs of variables. Darker colors typically indicate stronger correlations, with positive correlations shown in one color and negative correlations in another."
        
        elif plot_type == "bar" or plot_type == "pie":
            return "This chart shows the distribution of a categorical variable. Each segment represents a category, with the size proportional to the frequency or percentage of that category in the dataset."
        
        else:
            return f"This {plot_type} visualization shows patterns and relationships in the data. Examine the axes and legend to understand what is being displayed."


