# Telecom Customer Churn Predictor (KNN Classification)

A machine learning portfolio project demonstrating the application of the **K-Nearest Neighbors (KNN)** algorithm to a real-world business optimization problem: predicting subscription user churn.

##  Project Overview
Customer retention is one of the primary drivers of profitability in subscription-based industries. This project builds an end-to-end data pipeline to analyze customer profiles and predict who is most likely to cancel their service next month, enabling proactive marketing interventions.

##  Tech Stack & Key Concepts
*   **Language:** Python (VS Code with `ipykernel`)
*   **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
*   **Core Concepts:** Data preprocessing, one-hot encoding, data leakage prevention, feature scaling, hyperparameter tuning, and metric evaluation.

##  Machine Learning Pipeline

### 1. Data Cleaning & Imputation
*   Identified that the `TotalCharges` column contained blank spaces disguised as text objects due to unbilled new accounts.
*   Coerced missing fields to numeric formats and dropped records containing null observations.

### 2. Feature Engineering & Preprocessing
*   **Categorical Encoding:** Leveraged `pd.get_dummies(drop_first=True)` to convert categorical strings into numeric indicators while consciously preventing multi-collinearity (the *dummy variable trap*).
*   **Data Leakage Prevention:** Stratified the dataset into an **80/20 train-test split**.
*   **Feature Scaling:** Applied `StandardScaler` to calculate scale limits *only* on the training split, and transformed the test partition afterward. This ensured features with large numerical ranges (e.g., `TotalCharges`) did not artificially drown out vital small-scale signals (e.g., `tenure`).

### 3. Hyperparameter Tuning
*   Iterated through a sequence of odd $K$ neighbors ($K \in [1, 25]$) to track model error patterns. 
*   Identified the optimal value at **$K = 19$**, which successfully balanced the bias-variance trade-off by smoothing local out-of-distribution noise without over-generalizing customer profiles.

