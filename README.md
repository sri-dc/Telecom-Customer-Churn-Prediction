# Telecom Customer Churn Prediction and Risk Scoring System

## Project Overview
This project is designed to predict customer churn in the telecommunications sector. By accurately identifying high-risk customers, the system enables businesses to launch targeted retention outreach campaigns and reduce customer turnover.

## Business Problem
- Identify customers who are likely to churn.
- Score every customer with a precise churn-risk probability.
- Flag high-risk customers for targeted retention campaigns.

## Dataset Summary
- **Train Set**: 4,250 rows with labeled `churn` outcomes.
- **Test Set**: 750 unlabeled rows (scored as new customers for the risk deliverable).
- **Key Features**: `international_plan`, `voice_mail_plan`, `number_customer_service_calls`, and day/evening/night/international usage minutes.

## Workflow & Pipeline
1. **EDA & Data Cleaning**: Analyzed missing values and addressed severe class imbalance (~14% churn rate).
2. **Feature Engineering**:
   - Removed redundant `total_*_charge` columns due to perfect correlation (1.0) with usage minutes.
   - Encoded binary flags and integrated categorical feature handling into a leak-proof `ColumnTransformer` pipeline.
3. **Handling Class Imbalance**: Applied the **SMOTE** oversampling technique to balance the minority churn class.
4. **Model Training**:
   - Evaluated 8 distinct model configurations, including Logistic Regression, Random Forest Classifier, and XGBoost Classifier.
   - Selected the optimal model based on the **ROC-AUC** evaluation metric.
   - Fine-tuned hyperparameters using `RandomizedSearchCV.
5. **Risk Scoring**: Evaluated held-out test data to generate churn risk probabilities for unlabeled customers.

## Results & Key Insights
- **Best Performing Model**: Fine-tuned XGBoost Classifier / Random Forest Classifier evaluated on ROC-AUC.
- **Top Churn Predictors**: `number_customer_service_calls`, `total_day_minutes`, and `international_plan.
