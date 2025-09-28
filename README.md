# Predicting the Risk of Default on Home Credit

## 📋 Project Overview

This project aims to predict loan default risk using the Home Credit Default Risk dataset, which contains comprehensive customer information, credit history, and payment records. The analysis involves in-depth data exploration, cleaning, and feature engineering from multiple data sources to build machine learning and deep learning models that can assist financial institutions in assessing creditworthiness more accurately.

## 🎯 Business Objective

To develop a predictive model that can identify potential loan defaulters, enabling Home Credit to make better-informed credit granting decisions and reduce financial risks.

## 📊 Dataset Description

The project utilizes multiple datasets from Home Credit:

- **application_train.csv**: Primary training data with loan application information
- **application_test.csv**: Test data for competition submissions
- **bureau.csv**: Client's previous credits from other financial institutions
- **bureau_balance.csv**: Monthly balances of previous credits
- **previous_application.csv**: Previous loan applications at Home Credit
- **POS_CASH_balance.csv**: Point of sales and cash loan monthly balances
- **credit_card_balance.csv**: Monthly credit card balances
- **installments_payments.csv**: Payment history for previous loans

## 🛠️ Technical Implementation

### 1. Data Preprocessing & Feature Engineering

- **Missing Value Handling**: 
  - Numeric columns: Imputed with median values
  - Categorical columns: Filled with 'MISSING' category
  
- **Feature Engineering**:
  - Age calculation from days since birth
  - Years employed conversion
  - Income per family member
  - Credit-to-income ratio
  - Annuity-to-income ratio

### 2. Aggregated Features

Created comprehensive features from bureau data including:
- Credit duration statistics (min, max, mean days)
- Overdue credit patterns
- Credit sum aggregates
- Debt amount summaries

### 3. Exploratory Data Analysis

Key insights uncovered:
- Default rates by income type
- Age group impact on default probability
- Correlation analysis of EXT_SOURCE features with target variable

## 🤖 Models Implemented

### Traditional Machine Learning
1. **Random Forest Classifier**
   - 100 estimators with balanced class weights
   - Comprehensive feature importance analysis
   - Cross-validation evaluation

2. **LightGBM Classifier**
   - Gradient boosting with early stopping
   - Advanced hyperparameter tuning
   - Feature importance ranking

### Deep Learning
1. **Simple Feedforward Neural Network**
   - Architecture: 128 → 64 → 32 neurons
   - Dropout regularization (30%, 20%)
   - ReLU activation with sigmoid output

2. **Deep Neural Network with Batch Normalization**
   - Architecture: 256 → 128 → 64 → 32 neurons
   - Batch normalization layers
   - Higher dropout rates for regularization
   - Advanced optimization with callbacks

## 📈 Model Performance

### Evaluation Metrics
- **AUC-ROC** (Primary metric)
- **Accuracy**
- **Precision**
- **Recall** 
- **F1-Score**

### Performance Comparison
All models are comprehensively evaluated and compared using:
- ROC curve analysis
- Confusion matrices
- Training history visualization
- Feature importance analysis

## 🚀 Key Features

### Technical Excellence
- **Comprehensive Data Integration**: Multiple data sources merged and engineered
- **Advanced Feature Engineering**: Domain-specific feature creation
- **Model Diversity**: Traditional ML and deep learning approaches
- **Robust Evaluation**: Multiple metrics and visualization techniques

### Business Relevance
- **Interpretable Results**: Feature importance and business insights
- **Practical Implementation**: Ready-to-use prediction pipeline
- **Risk Assessment**: Probability-based default predictions

## 📁 Project Structure
home-credit-default-risk/
├── data/
│ ├── application_train.csv
│ ├── application_test.csv
│ └── ... (other dataset files)
├── notebooks/
│ └── home_credit_analysis.ipynb
├── models/
│ ├── random_forest_model.pkl
│ ├── lightgbm_model.pkl
│ └── neural_networks/
├── submissions/
│ ├── home_credit_submission.csv
│ └── home_credit_ensemble_submission.csv
└── README.md


## 🏆 Results

The project delivers:
1. **Multiple trained models** with performance comparisons
2. **Feature importance analysis** for business interpretation
3. **Test predictions** ready for competition submission
4. **Comprehensive documentation** of methodology and insights

## 🔧 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm tensorflow torch
```

🎮 Usage
Data Preparation: Place dataset files in the data/ directory

Model Training: Run the analysis notebook to train all models

Predictions: Use trained models to generate predictions on new data

Evaluation: Analyze model performance using provided visualizations

📝 Key Findings
Most Predictive Features: EXT_SOURCE variables, credit history, and income ratios

Best Performing Model: LightGBM consistently shows strong performance

Business Insights: Clear patterns in default rates across demographic segments

👥 Contributors
Data science project focused on credit risk prediction using advanced machine learning and deep learning techniques.

📄 License
This project is for educational and competition purposes using the Home Credit Default Risk dataset.
