# U.S. Housing Price Feature Analysis

A machine learning project to identify the key features that influence rental property pricing across the United States — covering CONUS, Alaska, and Hawaii.

---

## Overview

[This project uses a dataset of ~385,000 Craigslist rental listings](https://www.kaggle.com/datasets/austinreese/usa-housing-listings) to build and compare three ensemble regression models. The goal is not just to predict price, but to understand **which property features drive pricing**, giving actionable insight for real estate stakeholders.


---

## Dataset

- **Size**: 384,977 listings (after cleaning: ~362,781)
- **Coverage**: All 50 U.S. states
- **Features**: Square footage, beds, baths, pet policies, laundry/parking options, amenities, geographic coordinates, state, and region

---

## Pipeline

```
Raw Data → EDA → Cleaning → Feature Engineering → Modeling → Evaluation
```

1. **Exploratory Data Analysis** — Summary statistics, missing value analysis, geographic scatter plots
2. **Data Cleaning** — Geographic bounding boxes (CONUS + Alaska/Hawaii), IQR-based price filtering per region, removing extreme outliers
3. **Feature Engineering** — K-Means location clustering (125 clusters), target encoding for categorical features
4. **Modeling** — Three ensemble regressors trained on an 80/20 split (290,224 train / 72,557 test)

---

## Models & Results

| **Model** | **MAE** | **RMSE** | **R²** |
|---|---|---|---|
| **Random Forest** | **$62.70** | **$137.80** | **0.922** |
| XGBoost | $124.33 | $185.62 | 0.858 |
| Gradient Boosting | $150.68 | $221.11 | 0.799 |

Random Forest achieved the best performance with an **R² of 0.922**, meaning it explains ~92% of the variance in rental prices.

---

## Key Features Influencing Price

Across all three models, the top predictors of rental price were consistently:

- **Location** (state, region, and geographic cluster)
- **Square footage**
- **Number of bedrooms and bathrooms**
- **Property type**

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Modeling | `scikit-learn`, `xgboost` |
| Encoding | `category_encoders` (Target Encoding) |
| Visualization | `matplotlib`, `seaborn` |
| Persistence | `joblib` |

---

## Project Structure

```
├── Capstone.ipynb                          # Full analysis notebook
├── CS703_Applied_Data_Science_report.pdf   # Project report
├── housing_resized.csv                     # Dataset
├── requirements.txt                        # Dependencies
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Then open and run `Capstone_Final.ipynb` in order.

---

## Report

The full methodology, visualizations, and findings are documented in [`CS703_Applied_Data_Science_report.pdf`](./CS703_Applied_Data_Science_report.pdf).
