# Credit Card Fraud Detection

This project focuses on building a machine learning model to detect fraudulent credit card transactions using anonymized and highly imbalanced transaction data.

## Dataset

* **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Description**: The dataset contains transactions made by European cardholders in September 2013. It includes 284,807 transactions, with 492 classified as fraud.
* **Features**: 30 anonymized features (`V1` to `V28`), `Time`, `Amount`, and target variable `Class` (1 for fraud, 0 for legitimate).

## Workflow

1. **Data Preprocessing**

   * Checked for missing values
   * Scaled `Amount` and `Time` features
   * Visualized class imbalance

2. **Modeling**

   * Logistic Regression
   * Random Forest
   * Decision Tree
   * XGBoost

3. **Evaluation**

   * Confusion Matrix
   * Precision, Recall, F1-Score
   * ROC-AUC Curve
   * Focused on minimizing False Negatives

4. **Handling Imbalance**

   * Used undersampling and SMOTE (Synthetic Minority Over-sampling Technique)
   * Compared model performance on original and balanced datasets

## Tools & Libraries

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib, Seaborn
* Imbalanced-learn (`imblearn`)
 
