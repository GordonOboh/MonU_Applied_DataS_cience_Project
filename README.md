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

<p align="center">
  <img src="charts/Housing Listing Locations, Not Cleaned.png" width="48%" alt="Listings before cleaning"/>
  <img src="charts/Housing Listing Locations, Cleaned.png" width="48%" alt="Listings after cleaning"/>
</p>
<p align="center">
  <img src="charts/Pearson Correlation with Price.png" width="48%" alt="Pearson Correlation"/>
  <img src="charts/Spearman Correlation with Price.png" width="48%" alt="Spearman Correlation"/>
</p>

---

## Models & Results

> Full metrics: [`KPM/model_performance_metrics.csv`](./KPM/model_performance_metrics.csv)

| **Model** | **MAE** | **RMSE** | **R²** |
|---|---|---|---|
| **Random Forest** | **$62.70** | **$137.80** | **0.922** |
| XGBoost | $124.33 | $185.62 | 0.858 |
| Gradient Boosting | $150.68 | $221.11 | 0.799 |

Random Forest achieved the best performance with an **R² of 0.922**, meaning it explains ~92% of the variance in rental prices.

<p align="center">
  <img src="charts/R2 by Model.png" width="32%" alt="R² by Model"/>
  <img src="charts/MAE by Model.png" width="32%" alt="MAE by Model"/>
  <img src="charts/RMSE by Model.png" width="32%" alt="RMSE by Model"/>
</p>

---

## Key Features Influencing Price

Across all three models, the top predictors of rental price were consistently:

- **Location** (state, region, and geographic cluster)
- **Square footage**
- **Number of bedrooms and bathrooms**
- **Property type**

<p align="center">
  <img src="charts/Random Forest Feature Importance (Top 4).png" width="32%" alt="RF Feature Importance"/>
  <img src="charts/Gradient Boosting Feature Importance (Top 4).png" width="32%" alt="GB Feature Importance"/>
  <img src="charts/XGBoost Feature Importance (Top 4).png" width="32%" alt="XGB Feature Importance"/>
</p>

### Observed vs Predicted

<p align="center">
  <img src="charts/Random Forest: Observed vs Predicted.png" width="32%" alt="RF Observed vs Predicted"/>
  <img src="charts/Gradient Boosting: Observed vs Predicted.png" width="32%" alt="GB Observed vs Predicted"/>
  <img src="charts/XGBoost: Observed vs Predicted.png" width="32%" alt="XGB Observed vs Predicted"/>
</p>

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
├── Capstone_Final.ipynb                    # Full analysis notebook
├── CS703_Applied_Data_Science_report.pdf   # Project report
├── housing_resized.csv                     # Dataset
├── charts/                                 # Generated plots
├── KPM/
│   └── model_performance_metrics.csv       # Model metrics
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
