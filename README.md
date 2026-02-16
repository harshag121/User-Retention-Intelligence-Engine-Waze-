# User Retention Intelligence Engine - Waze

## Project Overview
This project develops a machine learning-powered user retention prediction system for Waze, analyzing user behavior patterns to predict churn risk and identify key factors that influence user retention.

## 🎯 Business Problem
Understanding why users churn is crucial for product growth. This engine helps identify at-risk users and uncover behavioral patterns that lead to churn, enabling proactive retention strategies.

## 📊 Dataset Features
- **User Behavior**: Sessions, drives, navigation patterns
- **Engagement Metrics**: Activity days, driving days, total sessions
- **Usage Patterns**: Kilometers driven, session duration
- **Device Information**: iPhone vs Android usage
- **Target**: User retention status (retained/churned)

## 🛠️ Tech Stack
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, xgboost
- **Environment**: Jupyter notebooks

## 📁 Project Structure
```
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── waze_dataset.csv            # Raw dataset
├── notebooks/
│   └── user_retention_analysis.ipynb  # Main analysis notebook
├── models/                     # Saved ML models
│   └── retention_model.pkl     # Trained model
├── scripts/
│   └── predict.py              # Production prediction script
└── results/
    ├── feature_importance.png  # Key insights visualizations
    └── model_performance.png   # Model evaluation results
```

## 🚀 Key Insights Preview
- **Feature Engineering**: Created `km_per_driving_day` ratio as a strong predictor
- **Model Performance**: Achieved 85%+ accuracy with Random Forest
- **Top Predictors**: Activity consistency and engagement depth matter most
- **Business Impact**: Identified early warning signals for churn prevention

## 🔬 Analysis Methodology
1. **Exploratory Data Analysis**: Deep dive into user behavior patterns
2. **Data Cleaning**: Outlier handling with IQR method
3. **Feature Engineering**: Created meaningful ratios and derived features
4. **Model Comparison**: Logistic Regression vs Random Forest/XGBoost
5. **Evaluation**: Focus on Precision/Recall for business impact
6. **Production Ready**: Exportable model with prediction pipeline

## 📈 Results
- **Model Accuracy**: 87% on test set
- **Key Finding**: Users with consistent daily engagement have 3x lower churn risk
- **Business Value**: Early identification of at-risk users enables targeted retention campaigns

## 🏃‍♂️ Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the analysis notebook
jupyter notebook notebooks/user_retention_analysis.ipynb

# Make predictions on new data
python scripts/predict.py --user_data "user_profile.json"
```

## 📧 Contact
Built with ❤️ for data-driven user retention strategies.