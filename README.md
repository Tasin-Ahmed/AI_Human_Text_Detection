# AI vs Human Text Detection

## Project Overview
This project is a Machine Learning benchmarking system designed to distinguish between AI-generated and Human-written text in **Bengali and English**. It leverages rigorous statistical testing and graph-based feature extraction (such as centrality, connectivity, and density) to identify patterns unique to each source. The system benchmarks multiple machine learning models to determine the most effective approach for classification.

## Key Features
- **Graph-Based Feature Extraction**: Converts text into graph structures to compute metrics like Transitivity, Density, Eigenvalues, and Centrality.
- **Multi-Source Tokenization**: detailed analysis using various tokenization and lemmatization methods:
  - Spacy
  - NLTK
  - Simple Tokenization
- **Statistical Validation**: Employment of statistical tests (T-Test) to validate feature significance before modeling. Before training machine learning models, all extracted graph-based features undergo rigorous statistical validation to ensure they meaningfully distinguish between **AI-generated** and **Human-written** text.

    - **Independent Two-Sample T-Test**
        An independent two-sample t-test (Welch’s t-test) is applied to each feature to compare AI and Human distributions.

        - **Null Hypothesis (H₀)**: Feature means are equal
        - **Alternative Hypothesis (H₁)**: Feature means differ
        - Missing and non-finite values are removed
        - Unequal variance assumption is used
        - Significance threshold: **α = 0.05**

    - **Consistency (Robustness) Test**
        Each feature is tested multiple times using random sub-sampling.

        - Measures stability of statistical significance
        - Filters out noisy or sample-dependent features
        - Retains only consistently significant features

    - **Optimal Feature Selection**
        Features are ranked based on:
        - Mean difference (effect magnitude)
        - p-value
        - Consistency score
        - **Key Metrics Analyzed**:
            - Density, Connectivty, and Transitivity
            - Centrality Measures (Degree, Betweenness, Closeness)
            - Eigenvalues and Graph Energy
            - Shortest Path and Diameter
        - **Top Discriminating Features**:
            - **Top Eigenvalue**: Significantly higher in AI texts.
            - **Avg Closeness & Degree Centrality**: Higher in AI texts.
            - **Transitivity & Size of Largest Component**: Higher in Human texts.
- **Comprehensive Benchmarking**: evaluation of a wide range of ML models including:
  - K-Nearest Neighbors (KNN)
  - XGBoost
  - Support Vector Machines (SVM)
  - Random Forest
  - Gradient Boosting
  - Neural Network based approaches (via library implementations)

## Project Structure
The project is organized into the following key directories:

- **`Codes/`**: Contains the core Jupyter notebooks for the analysis pipeline:
  - `Human_vs_AI_Models.ipynb`: The main notebook for training, benchmarking, and tuning ML models.
  - `Feature_Sort.ipynb`: Analyzes and recommends features based on statistical significance.
  - `PMI.ipynb`: Likely used for Pointwise Mutual Information analysis.
  - `Statistical_test.ipynb`: Conducts statistical tests on the extracted features.
- **`Bangla Graph Analysis/`**: Resources and analysis specific to Bangla text datasets.
- **`English Text Analysis/`**: Resources and analysis specific to English text datasets.
- **`Other Paper Model/`**: Contains comparative models or baselines from other research.

## Dependencies
To run the analysis and models, you will need **Python 3.x** and the following libraries:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn openpyxl joblib
```

## Usage

### 1. Data Preparation
Ensure your data is formatted correctly (typically Excel files as seen in the code) containing the extracted graph features. The system anticipates specific feature sets derived from NLTK, Spacy, etc.

### 2. Running the Benchmarking System
1. Navigate to the `Codes/` directory.
2. Open `Human_vs_AI_Models.ipynb` in Jupyter Notebook or Google Colab.
3. Run the cells sequentially to:
   - Load and preprocess the data.
   - Build the mixed-source dataset from the best feature recommendations.
   - Train and benchmark all configured models.
   - Perform hyperparameter tuning on the top candidates.
   - Generate evaluation reports and visualizations.

### 3. Interpreting Results
Results will be saved in the `ml_classification_results/` directory (created during execution), including:
- **`model_comparison_results.xlsx`**: Summary metrics for all models.
- **`best_model_KNN.pkl`**: The saved best-performing model instance.
- **`confusion_matrices_tuned.png`**: Visual performance analysis.

## Key Findings
Based on current benchmarks on the provided dataset:
- **KNN (K-Nearest Neighbors)** has shown exceptional performance, achieving near-perfect classification metrics.
- **XGBoost** and **SVM (RBF Kernel)** are also strong contenders, consistently performing well in the benchmarking suite.
