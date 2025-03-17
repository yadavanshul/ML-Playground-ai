# AI-Powered Machine Learning Playground

An intelligent platform for exploratory data analysis, preprocessing, and machine learning with AI-driven insights and recommendations.

## Features

- **Dataset Management**: Upload custom datasets or select from predefined options
- **AI-Driven EDA**: Dynamic visualization with AI-generated insights
- **Interactive Dashboard**: Drag & drop interface for building custom EDA dashboards
- **Multi-Agent AI System**: Specialized AI agents for different tasks
- **RAG & ChromaDB**: Enhanced AI decision-making with retrieval-augmented generation
- **Error Detection**: Automatic identification of data issues
- **Preprocessing Pipeline**: AI-guided preprocessing workflow with multi-column operations
- **Comprehensive Preprocessing**: 30+ preprocessing techniques across 10 categories
- **ML Pipeline**: Supervised learning with multiple models, evaluation, and insights
- **End-to-End Workflow**: Seamless transition from EDA to preprocessing to ML

## Blog Posts & Resources

- [Phase 1: Machine Learning Playground Using OpenAI](https://medium.com/@workaholicanshul/machine-learning-playground-using-openai-9a7054bf0381)
- [Phase 2: Building an AI-Powered Machine Learning Playground](https://medium.com/@workaholicanshul/building-an-ai-powered-machine-learning-playground-a-step-by-step-guide-3f66e1c07faf)
- [LinkedIn Update](https://www.linkedin.com/feed/update/urn:li:activity:7307186640152641536/)
- [GitHub Repository](https://github.com/yadavanshul/ML-Playground)

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key (copy from `.env.example`):
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```
   export PYTHONPATH=$(pwd) && streamlit run ai_eda_pipeline/main.py
   ```

## Project Structure

```
/ML_Playground
├── ai_eda_pipeline/
│   ├── components/          # AI agents & ML functions
│   ├── data/                # Predefined datasets
│   ├── utils/               # Helper functions
│   ├── main.py              # Streamlit App
├── blog_images/             # Images for blog posts
├── chromadb_store/          # ChromaDB Embeddings
├── medium_blog_post.md      # Blog post about the project
├── PROGRESS_TRACKER.md      # Detailed progress tracking
├── README.md                # This file
├── requirements.txt         # Dependencies
```

## Usage

### Phase 1: AI-Powered EDA Dashboard

1. Select or upload a dataset
2. Explore AI-recommended visualizations
3. Add plots to your dashboard
4. Get AI insights for your visualizations
5. View the AI reasoning log for transparency

### Phase 2: AI-Guided Preprocessing

1. Switch to the Preprocessing tab
2. Get AI-suggested preprocessing steps
3. Configure preprocessing operations (supports multi-column selection)
4. Build your preprocessing workflow
5. Apply and evaluate your workflow
6. Export your workflow as JSON, Python code, or PDF

### Phase 3: Supervised Machine Learning

1. Switch to the ML Pipeline tab
2. Select target variable (automatic problem type detection)
3. Configure train-test split and select features
4. Choose ML models to train and compare
5. View comprehensive evaluation metrics and visualizations
6. Analyze feature importance and model insights
7. Export model summary and get AI recommendations

## Preprocessing Techniques

The platform offers a comprehensive set of preprocessing techniques:

### Data Cleaning
- Missing value imputation (mean, median, mode, KNN, forward/backward fill)
- Duplicate removal
- Outlier handling (Z-score, IQR, winsorization)
- Data type conversion

### Feature Scaling
- Standardization (Z-score normalization)
- Min-Max scaling
- Robust scaling
- Log transformation

### Encoding Categorical Variables
- One-Hot encoding
- Label encoding
- Frequency encoding
- Target encoding
- Binary encoding
- Ordinal encoding

### Feature Engineering
- Polynomial features
- Feature interactions (multiplication, addition, subtraction, division)
- Time features extraction
- Text processing

### Dimensionality Reduction
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- t-SNE
- UMAP

### Feature Selection
- Correlation-based selection
- Chi-square test
- Recursive Feature Elimination (RFE)
- LASSO
- Random Forest feature importance

### Handling Imbalanced Data
- SMOTE oversampling
- ADASYN oversampling
- Random undersampling
- Class weighting

### Data Transformation
- Binning (equal-width, equal-frequency, K-means)
- Log transformation
- Power transformation (Box-Cox, Yeo-Johnson)

## Machine Learning Models

The platform supports various supervised learning models:

### Classification Models
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Regression Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor

## Project Phases

### Phase 1: AI-Powered EDA Dashboard ✅

- Upload or select datasets
- Dynamic visualization generation
- AI insights on demand
- Interactive dashboard with drag & drop interface
- ChromaDB for storing dataset metadata and insights

### Phase 2: AI-Guided Preprocessing ✅

- AI-driven preprocessing recommendations
- Multi-column preprocessing operations
- Grouped preprocessing suggestions by type
- Drag & drop preprocessing workflow builder
- Real-time AI feedback on data transformations
- Workflow evaluation with AI scoring
- Export options (JSON, Python, PDF)
- Comprehensive preprocessing techniques across multiple categories

### Phase 3: Supervised Machine Learning ✅

- Target variable selection with automatic problem type detection
- Feature selection and train-test split configuration
- Multiple model selection and comparison
- Comprehensive model evaluation with interactive visualizations
- Feature importance analysis and model insights
- Model export functionality
- AI-generated recommendations for model improvement

### Phase 4: Advanced ML Features and Deployment 🔜

- Hyperparameter tuning with Optuna
- Ensemble methods (stacking, blending)
- Cross-validation strategies
- Model explainability with SHAP
- Deployment options (API, Docker)
- Time series forecasting models
- Deep learning integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT 