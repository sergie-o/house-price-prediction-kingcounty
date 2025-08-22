# 🏡📊 Decoding King County’s Housing Market with Machine Learning  

Can algorithms decode the housing market?  
In this project, I apply **machine learning techniques** on **21,000+ home sales** from King County (Seattle area, May 2014 – May 2015) to uncover which features — from square footage to renovations — drive property values, and to build predictive models that **estimate sale prices with accuracy**.  

<p align="center">
  <img src="https://github.com/sergie-o/house-price-prediction-kingcounty/blob/main/651ECB65-0269-4014-9BCF-FAC3D2A46F2F.png" width="800">
</p>

---

## 📊 **Dataset Overview**
📌 **Source:** [King County House Sales Dataset (Kaggle)](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)  
📏 **Size:** 21,613 rows × 21 columns  
🔑 **Key Variables:**  
- `price` 💵 – Sale price of the home (target variable)  
- `sqft_living` 📐 – Living space area  
- `bedrooms`, `bathrooms` 🛏 – House features  
- `floors`, `waterfront`, `view` 🌅 – Condition and luxury indicators  
- `yr_built`, `yr_renovated` 🏗 – House age & renovations  
- `lat`, `long` 📍 – Geographical location  

⚠ **Data quirks:** Duplicate dates with varying prices and large differences in feature scales (e.g., sqft vs bedrooms) required feature scaling and careful preprocessing.  

---

## 🎯 **Research Goal**
💡 **Core Question:** *What features truly drive house prices in King County, and which machine learning models can best predict them?*  

This project explores both:  
- **Feature importance** → uncovering the real drivers of value.  
- **Model benchmarking** → testing and comparing multiple ML algorithms.  

---

## 🛠 **Steps Taken**
1. **Data Cleaning** 🧹 – Removed irrelevant IDs, handled duplicates, extracted `year`, `month`, and `dayofweek` from dates.  
2. **Exploratory Data Analysis (EDA)** 🔍 – Correlation heatmaps, trend analysis, and visualization of pricing distributions.  
3. **Feature Engineering** ⚙️ – Standardization, cyclical encoding of date features, and dropping weakly correlated predictors.  
4. **Modeling** 🤖 – Evaluated multiple models:
   - Linear Regression (baseline + variants with feature engineering)  
   - Ridge, Lasso, ElasticNet regularization  
   - Random Forest & Gradient Boosting  
   - XGBoost (with hyperparameter tuning)  
5. **Hyperparameter Tuning** 🎚 – Used RandomizedSearchCV to optimize Random Forest and XGBoost.  
6. **Results Leaderboard** 🏆 – Compared models using MSE, RMSE, and R² on train/test sets.  

---

## 🚀 **Main Insights**
- **Square footage & grade** are the strongest predictors of price 🏗.  
- **Year built/renovated** has moderate influence, while latitude/longitude correlations are weak.  
- **Linear Regression** baseline: R² ≈ 0.70.  
- **Ridge/Lasso** gave small stability improvements.  
- **Random Forest & XGBoost** dominated with R² ≈ 0.85–0.89 after tuning, reducing RMSE significantly.  
- **Feature engineering + tuning** proved essential for squeezing extra performance.  

---
🛠 Steps Taken
Data Cleaning 🧹 – Removed irrelevant IDs, handled duplicates, extracted year, month, and dayofweek from dates.

Exploratory Data Analysis (EDA) 🔍 – Correlation heatmaps, trend analysis, and visualization of pricing distributions.

Feature Engineering ⚙️ – Standardization, cyclical encoding of date features, and dropping weakly correlated predictors.

Modeling 🤖 – Evaluated multiple models:

Linear Regression (baseline + variants with feature engineering)

Ridge, Lasso, ElasticNet regularization

Random Forest & Gradient Boosting

XGBoost (with hyperparameter tuning)

Hyperparameter Tuning 🎚 – Used RandomizedSearchCV to optimize Random Forest and XGBoost.

Results Leaderboard 🏆 – Compared models using MSE, RMSE, and R² on train/test sets.

🚀 Main Insights
Square footage & grade are the strongest predictors of price 🏗.

Year built/renovated has moderate influence, while latitude/longitude correlations are weak.

Linear Regression baseline: R² ≈ 0.70.

Ridge/Lasso gave small stability improvements.

Random Forest & XGBoost dominated with R² ≈ 0.85–0.89 after tuning, reducing RMSE significantly.

Feature engineering + tuning proved essential for squeezing extra performance.

🔄 How to Reproduce
Prerequisites:

Python 3.12+ 🐍

Libraries:

Data Handling: pandas, numpy

Visualization: matplotlib, seaborn, plotly

Machine Learning: scikit-learn (for LinearRegression, StandardScaler, train_test_split, mean_squared_error, Ridge, RidgeCV, Pipeline), xgboost (for XGBRegressor), statsmodels

Hyperparameter Tuning: scipy (for randint, uniform)

Utilities: re, tabulate, prettytable

Run Instructions:

Clone this repository

Bash

git clone https://github.com/sergie-o/house-price-prediction-kingcounty.git
Navigate to the project folder

Bash

cd house-price-prediction-kingcounty
Open the Jupyter Notebook

If you use Jupyter Notebook:

Bash

jupyter notebook "king_county_housing_analysis.ipynb"
Or, open it in VSCode by double-clicking the file or using:

Bash

code "king_county_housing_analysis.ipynb"
Ensure the dataset is in the correct location

The file kc_house_data.csv must be in the data folder inside your project directory.

Run all cells

Select Cell > Run All in Jupyter Notebook or VSCode to reproduce the analysis.

🚀 Next Steps
Incorporate geographical features 🗺️: Analyze the impact of lat and long in more detail. Use clustering (like K-Means) to group similar neighborhoods and see how location-based features influence pricing.

Implement a more robust time-series model 🕰️: Use date to analyze price trends over the year. A time-series model could better capture seasonal fluctuations in the housing market.

Explain model predictions 🤖: Use a library like SHAP or LIME to explain why your best-performing models (XGBoost or Random Forest) made specific price predictions, providing more transparent and trustworthy insights.
