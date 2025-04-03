# Fraud Detection using Machine Learning

## Project Overview
This project focuses on detecting fraudulent transactions using machine learning techniques. Given the severe class imbalance in financial transaction data, special attention is paid to minimizing false negatives (fraudulent transactions classified as non-fraudulent).

## Dataset
- **File**: `Fraud.csv`
- **Description**: Contains financial transaction details with fraud labels.
- **Key Challenge**: Highly imbalanced data, requiring specific handling techniques.

## Technologies Used
- **Python**: Data processing and modeling
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

## Steps Involved
### 1. Data Preprocessing
- Read and clean the dataset.
- Encode categorical variables.
- Handle class imbalance using techniques like resampling and class weighting.

### 2. Exploratory Data Analysis (EDA)
- Visualizing transaction patterns.
- Identifying key features affecting fraud detection.

### 3. Model Training
- Logistic Regression
- Decision Trees
- XGBoost Classifier
- Isolation Forest (for anomaly detection)

### 4. Model Evaluation
- **Metrics Used**:
  - Accuracy (not the primary focus due to imbalance)
  - Precision, Recall, and F1-score
  - Confusion Matrix
  - Precision-Recall Curve

## Key Observations
- A simple classification based on majority class results in **99.87% accuracy**, but fails in detecting fraud.
- The goal is to minimize **false negatives**, as allowing fraud transactions is worse than flagging a legitimate one.
- Feature selection shows that `nameOrig` is not a significant predictor.

## Next Steps
- Further fine-tuning of hyperparameters.
- Experimenting with ensemble models.
- Implementing cost-sensitive learning techniques to improve fraud detection.

## How to Run the Notebook
1. Install the necessary libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
2. Load the dataset (`Fraud.csv`).
3. Execute the Jupyter notebook to preprocess, train, and evaluate models.

---
### Author
Arnab

