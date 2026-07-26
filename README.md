# Housing Amenities Impact on Apartment Listing Prices in the Southern USA

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
[![Report](https://img.shields.io/badge/report-PDF-red?logo=adobeacrobatreader&logoColor=white)](https://github.com/GordonOboh/MonU_Applied_DataS_cience_Project/blob/main/CS703_Applied_Data_Science_report.pdf)

A machine learning project to identify how apartment amenities impact listing prices across the Southern United States, using the CRISP-DM methodology.

---

## Overview

This project analyzes apartment rental listings to determine which amenities and property features most influence pricing. Three ensemble regression models are built and compared to identify the key drivers of listing price, providing actionable insight for property managers, landlords, and renters in the Southern US market.

---

## Dataset

- **Coverage**: Southern United States (apartment listings)
- **Features**: Square footage, beds, baths, pet policies (cats/dogs allowed), laundry options, parking options, wheelchair access, EV charging, smoking policy, furnished status, and geographic location

---

## Pipeline

```
Raw Data → EDA → Cleaning → Feature Engineering → Modeling → Evaluation
```

1. **Exploratory Data Analysis** — Summary statistics, distributions, correlation analysis
2. **Data Cleaning** — Handling missing values, outlier removal, geographic filtering
3. **Feature Engineering** — Encoding categorical amenity variables, feature scaling
4. **Modeling** — Three ensemble regressors trained on an 80/20 split

<p align="center">
  <img src="charts/Housing Listing Locations, Not Cleaned.png" width="48%" alt="Listings before cleaning"/>
  <img src="charts/Southern Apartment Listing Locations, Cleaned.png" width="48%" alt="Southern apartment listings after cleaning"/>
</p>
<p align="center">
  <img src="charts/Pearson Correlation with Price.png" width="48%" alt="Pearson Correlation"/>
  <img src="charts/Spearman Correlation with Price.png" width="48%" alt="Spearman Correlation"/>
</p>

---

## Models & Results

> Full metrics: [`charts/model_performance_metrics.csv`](./charts/model_performance_metrics.csv)

| **Model** | **MAE** | **RMSE** | **R²** |
|---|---|---|---|
| **Random Forest** | **$35.58** | **$77.32** | **0.946** |
| HistGradientBoosting | $47.07 | $85.70 | 0.934 |
| XGBoost | $46.47 | $82.58 | 0.939 |

Random Forest achieved the best performance with an **R² of 0.946**, explaining ~95% of the variance in apartment listing prices.

<p align="center">
  <img src="charts/R2 by Model.png" width="32%" alt="R² by Model"/>
  <img src="charts/MAE by Model.png" width="32%" alt="MAE by Model"/>
  <img src="charts/RMSE by Model.png" width="32%" alt="RMSE by Model"/>
</p>

---

## Key Features Influencing Price

Across all models, the top predictors of listing price were:

- **Square footage**
- **Location** (geographic coordinates, state, region)
- **Number of bedrooms and bathrooms**
- **Amenities** (pool, garage, pet policies, laundry options)

<p align="center">
  <img src="charts/Random Forest Feature Importance (Top 4).png" width="32%" alt="RF Feature Importance"/>
  <img src="charts/HistGradientBoosting Feature Importance (Top 4).png" width="32%" alt="HGB Feature Importance"/>
  <img src="charts/XGBoost Feature Importance (Top 4).png" width="32%" alt="XGB Feature Importance"/>
</p>

### Observed vs Predicted

<p align="center">
  <img src="charts/Random Forest: Observed vs Predicted.png" width="32%" alt="RF Observed vs Predicted"/>
  <img src="charts/HistGradientBoosting: Observed vs Predicted.png" width="32%" alt="HGB Observed vs Predicted"/>
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
├── Capstone_Final_cc_ST_2.ipynb            # Full analysis notebook
├── CS703_Applied_Data_Science_report.pdf   # Project report
├── charts/                                 # Generated plots and metrics
├── requirements.txt                        # Dependencies
├── Status Report/                          # Weekly status reports (PPTX + MD)
├── Project Management/                     # Project plan, presentations
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Then open and run `Capstone_Final_cc_ST_2.ipynb` in order.

---

## Report

The full methodology, visualizations, and findings are documented in [`CS703_Applied_Data_Science_report.pdf`](./CS703_Applied_Data_Science_report.pdf).
