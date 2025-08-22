# 🏡📊 Decoding King County’s Housing Market with Machine Learning  

Can algorithms decode the housing market?  
In this project, I apply **machine learning techniques** on **21,000+ home sales** from King County (Seattle area, May 2014 – May 2015) to uncover which features — from square footage to renovations — drive property values, and to build predictive models that **estimate sale prices with accuracy**.  

<p align="center">
  <img src="visuals/house_price_feature.png" alt="King County House Price Prediction" width="800">
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

## 🔄 **How to Reproduce**
**Prerequisites:**  
- Python 3.12+ 🐍  
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `xgboost`, `df2img`  

**Run Instructions:**  
1. Clone this repository  
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
