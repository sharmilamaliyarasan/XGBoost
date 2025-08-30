# 🎯 XGBoost Classification Project

## 📖 Description

This project demonstrates XGBoost (Extreme Gradient Boosting) for binary classification using Python.
It includes EDA, visualization, preprocessing, model training, hyperparameter tuning, and deployment with Streamlit + Pickle.

## 📂 Dataset

File: WA_Fn-UseC_-Telco-Customer-Churn.csv (Telco Customer Churn dataset)

Features: Customer demographics, account info, services

Target: Churn (Yes / No)

Size: ~7,043 rows

## ⚙️ Methods

EDA (Exploratory Data Analysis): Missing values, distributions, correlation heatmap

Preprocessing: Label encoding, feature scaling, train-test split

Model: XGBoostClassifier with hyperparameter tuning via GridSearchCV

Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

Deployment: Streamlit app for interactive prediction + Pickle for model saving

## 📊 Results

✅ Best Parameters: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}

✅ Accuracy: 85.3%

✅ ROC-AUC: 0.81

⚠️ Precision & Recall show imbalance (low recall = many false negatives)

✅ Confusion Matrix shows good classification for non-churn, but churn prediction is harder

## 📦 Requirements

Install dependencies with:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit


## Run Streamlit app:

streamlit run app.py

## 📈 Conclusion

XGBoost is highly effective for churn prediction with good accuracy and ROC-AUC

However, recall is low → the model misses some actual churn cases

Feature importance analysis helps understand key drivers of churn

## 🚀 Future Improvements

🔹 Handle class imbalance with SMOTE or class weights

🔹 Feature engineering (interaction features, contract length categories, etc.)

🔹 Try advanced models (LightGBM, CatBoost) for comparison

🔹 Deploy with Flask or FastAPI as a production API
