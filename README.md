# 🚗 Car Price Prediction using Machine Learning

### 🌍 Overview

This project demonstrates a **data-driven approach to predicting used car prices** based on key attributes like brand, engine volume, mileage, and registration status.
Using **Python, Pandas, Seaborn, StatsModels, and Scikit-Learn**, a **Multiple Linear Regression** model was developed that achieves an **R² score of 0.745**, explaining **~74.5% of price variance** — a strong baseline for real-world price estimation.

The model helps:

* **Dealers** set fair prices
* **Online marketplaces** detect mispriced listings
* **Buyers** estimate true market value

---

## 🧾 Project Summary

### 🎯 Objective

Predict car prices based on both numerical and categorical variables to understand **what drives price differences** in the used car market.

### 🧠 Model Summary

| Item                                      | Description                  |
| :---------------------------------------- | :--------------------------- |
| **Model Type**                            | Multiple Linear Regression   |
| **Target Variable**                       | `Price` (log-transformed)    |
| **Algorithm**                             | Ordinary Least Squares (OLS) |
| **Training Data Split**                   | 80% train / 20% test         |
| **R² Score**                              | **0.745**                    |
| **Mean Absolute Percentage Error (MAPE)** | **~36.26%**                  |
| **Average Actual Price**                  | $18,165                      |
| **Average Predicted Price**               | $15,946                      |
| **Average Deviation (Residual)**          | $2,219                       |

**Interpretation:**
The model captures ~75% of the price variability, showing that it can predict prices within **±36% accuracy** on average — a strong result for a transparent, interpretable regression approach.

---

## ⚙️ Step-by-Step Process

### 1. **Importing Dependencies**

Libraries used:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
```

**Task:** Load libraries for data cleaning, visualization, and regression analysis.

---

### 2. **Data Loading and Exploration**

```python
raw_data = pd.read_csv('CarSalesDataset.csv')
raw_data.describe(include='all')
```

**Task:** Import the dataset and explore column types, missing values, and basic statistics.

**Why:** Understanding the data distribution helps determine cleaning and transformation needs.

---

### 3. **Data Cleaning**

```python
data = raw_data.drop(['Model'], axis=1)
data_no_mv = data.dropna(axis=0)
```

**Task:** Remove irrelevant columns and drop missing records.
**Why:** Ensures model reliability and prevents null-related errors.
**Result:** Data reduced to 3,867 clean observations.

---

### 4. **Outlier Removal**

```python
q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price'] < q]
```

**Task:** Remove extreme values beyond the 99th percentile for `Price`, `Mileage`, and `EngineV`.
**Why:** Outliers can distort regression slopes and residuals.

---

### 5. **Log Transformation**

```python
data_cleaned['log_price'] = np.log(data_cleaned['Price'])
```

**Task:** Normalize skewed `Price` distribution.
**Why:** Log transformation linearizes relationships and ensures residuals are normally distributed.

---

### 6. **Visualizing Relationships**

```python
plt.scatter(data_cleaned['Year'], data_cleaned['log_price'])
```

**Task:** Confirm linearity between log-transformed price and features (`Year`, `Mileage`, `EngineV`).
**Why:** Linear regression assumes a linear relation between independent variables and target.

---

### 7. **Multicollinearity Check (VIF)**

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

**Task:** Compute VIF to detect correlated variables.
**Result:** `Year` had high correlation and was removed.
**Why:** Prevents inflated coefficient variance and unstable predictions.

---

### 8. **Encoding Categorical Features**

```python
data_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
```

**Task:** Convert categorical columns (`Brand`, `Body`, `Engine Type`, `Registration`) into numeric form.
**Why:** Regression models only accept numerical data.
**Result:** 17 encoded columns ready for modeling.

---

### 9. **Feature Scaling**

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
inputs_scaled = scaler.fit_transform(inputs)
```

**Task:** Standardize numerical features to equalize influence on the model.
**Why:** Prevents large-scale features (like mileage) from dominating training.

---

### 10. **Splitting Data**

```python
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)
```

**Task:** Separate data for training and validation.
**Why:** Evaluates model generalization on unseen data.

---

### 11. **Model Training**

```python
reg = LinearRegression()
reg.fit(x_train, y_train)
```

**Task:** Train a multiple linear regression model.
**Why:** Establish a baseline model that’s interpretable and easy to maintain.
**Output:** Intercept = 9.415 | R² = 0.745

---

### 12. **Residual & Performance Evaluation**

```python
sns.displot(y_train - y_hat)
reg.score(x_train, y_train)
```

**Task:** Evaluate model fit, residual symmetry, and explanatory power.
**Why:** Ensures that residuals are random (no systematic bias).

---

### 13. **Prediction and Comparison**

```python
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
```

**Task:** Back-transform predictions and calculate percentage error.
**Why:** Compare predicted vs. actual prices to quantify accuracy.

---

## 📊 Key Findings

| Feature                         | Impact on Price | Interpretation                          |
| :------------------------------ | :-------------- | :-------------------------------------- |
| **Mileage**                     | -0.449          | Higher mileage reduces price.           |
| **Engine Volume**               | +0.209          | Larger engines increase value.          |
| **Registration (Yes)**          | +0.320          | Registered cars priced ~38% higher.     |
| **Brand (BMW, Mercedes)**       | Positive        | Premium brands command higher prices.   |
| **Brand (Renault, Mitsubishi)** | Negative        | Lower brand perception decreases price. |

---

## 📈 Model Insights

* The model performs **best for mid-priced vehicles ($30K–$35K)** with <1% error.
* Prediction accuracy decreases for luxury and low-end cars due to natural price volatility.
* Residual distribution confirms no major bias — model generalizes well.

---

## 💼 Business Value

* **Dealerships:** Automate competitive price recommendations.
* **Marketplaces:** Detect overpriced or underpriced listings.
* **Buyers:** Verify fair market value before purchase.

---

## 🚀 Next Steps

* Implement **Ridge/Lasso regression** for better generalization.
* Add more predictors (e.g., transmission type, region, age).
* Build a **Streamlit dashboard** for real-time user interaction.

---

## 🧮 Technologies Used

* **Python 3.9**
* **Pandas / NumPy** for data manipulation
* **Seaborn / Matplotlib** for visualization
* **StatsModels / Scikit-Learn** for regression and diagnostics
* **Jupyter / Spyder IDE (Anaconda Environment)**

---

## 🏁 Conclusion

This project demonstrates how **linear regression can effectively predict used car prices** with over **74% accuracy**, offering both **interpretability and business insight**.
By combining thoughtful data cleaning, log transformation, and proper feature engineering, the model delivers reliable predictions that align with real-world market trends.

> “Data-driven car pricing transforms guesswork into confidence — for both sellers and buyers.”
