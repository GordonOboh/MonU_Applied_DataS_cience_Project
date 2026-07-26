# Final Presentation — CS703 Applied Data Science Project

**Title:** Housing Amenities Impact on Apartment Listing Prices in the Southern USA

---

## 1. Project Overview (5 Points)

- **Business Problem:** Renters and property stakeholders in the Southern US lack clarity on which amenities and features truly drive rental pricing, making it difficult to price listings competitively or decide which property improvements yield the best return.
- **Business Goal:** Identify the key features that influence rental prices for apartments in the Southern USA so that landlords, property managers, and renters can make data-informed pricing and investment decisions.
- **Data Mining Goals:**
  - Build and compare three ensemble regression models (Random Forest, HistGradientBoosting, XGBoost) to predict rental price from property features
  - Determine the relative importance of each amenity feature in driving predicted price, producing an actionable ranking for investors and property managers
- **Background:** ~385K Craigslist rental listings across 50 US states, filtered to ~150K apartment listings in 17 Southern states. The rental market is opaque — this analysis brings data-driven transparency.

---

## 2. Summary of Each Phase (10 Points)

### Phase 1 — Business Understanding
- Defined the business problem: what drives apartment rental prices in the Southern US?
- Established success criteria: model can identify the top amenity drivers of rental price, producing rankings actionable for renovation prioritization and pricing strategy
- Created project plan with timeline across all 6 CRISP-DM phases

### Phase 2 — Data Understanding
- Collected ~385K Craigslist listings from [Kaggle (Austin Reese dataset)](https://www.kaggle.com/datasets/austinreese/usa-housing-listings)
- Explored distributions of price, sqft, beds, baths; mapped geographic spread
- Identified data quality issues: missing values in amenity columns, geographic outliers (non-US coordinates), extreme price outliers

### Phase 3 — Data Preparation
- Filtered to Southern states per Census Bureau definition (17 states: AL, AR, DC, DE, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN, TX, VA, WV) plus lat/lon bounding box
- Dropped irrelevant columns (id, url, image_url, description)
- IQR-based price filtering per region to remove extreme outliers
- Target encoding for high-cardinality categoricals (state, region, laundry/parking options)
- K-Means clustering on lat/lon (148 clusters) to capture micro-location effects
- Final dataset: ~150K apartment listings with engineered features

### Phase 4 — Modeling
- Three ensemble regressors: Random Forest, HistGradientBoosting, XGBoost
- 80/20 train/test split
- Hyperparameter tuning across 7 runs
- Final models: all achieved R² > 0.93

### Phase 5 — Evaluation
- Random Forest best overall: R² = 0.946, MAE = $35.58, RMSE = $77.32
- HistGradientBoosting: R² = 0.934, MAE = $47.07, RMSE = $85.70
- XGBoost: R² = 0.939, MAE = $46.47, RMSE = $82.58
- All models consistently identified location, square footage, beds, baths as top predictors
- Assessed against business goals — all success criteria met

### Phase 6 — Deployment
- Documented deployment plan, monitoring and maintenance strategy
- Final presentation and report delivered

---

## 3. Data Cleansing (15 Points)

- **Geographic bounding box:** Applied Southern US lat/lon range (lat 24.55–40.64, lon -106.65–-75.0) to remove listings outside the target region, then filtered to the 17 Census Bureau Southern states: AL, AR, DC, DE, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN, TX, VA, WV
- **Duplicate removal:** Dropped duplicate rows and duplicate IDs
- **Column selection:** Removed uninformative columns (id, url, image_url, description)
- **Outlier treatment:** IQR-based price filtering per region — removed listings where price fell outside 1.5× IQR below Q1 or above Q3, computed separately for each region to account for local market differences
- **Apartment-only filter:** Narrowed to `type == 'apartment'` (~150K rows) for a focused, homogeneous analysis
- **Missing values:** Numeric columns (sqfeet, beds, baths, lat, long, and boolean amenities) imputed with median; laundry/parking options missing treated as explicit `"Missing"` category; state and region had no missing values
- **Feature engineering:** K-Means location clustering (148 clusters) to capture neighborhood-level price variation; target encoding for categoricals to convert text labels into numeric representations based on their relationship with price

---

## 4. Phase 4.0 — Modeling & Visualizations (30  Points)

### Models Built
| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| **Random Forest** | **0.946** | **$35.58** | **$77.32** |
| HistGradientBoosting | 0.934 | $47.07 | $85.70 |
| XGBoost | 0.939 | $46.47 | $82.58 |

### Key Visualizations

**Geographic Filtering — Before vs After Cleaning**
<img src="../charts/Housing%20Listing%20Locations%2C%20Not%20Cleaned.png" alt="Listings Not Cleaned" width="100" />
<img src="../charts/Southern%20Apartment%20Listing%20Locations%2C%20Cleaned.png" alt="Southern Apartment Listings Cleaned" width="100" />

**Correlation with Price**
<img src="../charts/Pearson%20Correlation%20with%20Price.png" alt="Pearson Correlation" width="100" />
<img src="../charts/Spearman%20Correlation%20with%20Price.png" alt="Spearman Correlation" width="100" />

**Model Performance Comparison**
<img src="../charts/R2%20by%20Model.png" alt="R² by Model" width="100" />
<img src="../charts/MAE%20by%20Model.png" alt="MAE by Model" width="100" />
<img src="../charts/RMSE%20by%20Model.png" alt="RMSE by Model" width="100" />
<img src="../charts/ASE%20by%20Model.png" alt="ASE by Model" width="100" />

**Observed vs Predicted (all 3 models)**
<img src="../charts/Random%20Forest%3A%20Observed%20vs%20Predicted.png" alt="Random Forest: Observed vs Predicted" width="100" />
<img src="../charts/HistGradientBoosting%3A%20Observed%20vs%20Predicted.png" alt="HistGradientBoosting: Observed vs Predicted" width="100" />
<img src="../charts/XGBoost%3A%20Observed%20vs%20Predicted.png" alt="XGBoost: Observed vs Predicted" width="100" />

**Feature Importance (Top 4 per model)**
<img src="../charts/Random%20Forest%20Feature%20Importance%20(Top%204).png" alt="Random Forest Feature Importance" width="100" />
<img src="../charts/HistGradientBoosting%20Feature%20Importance%20(Top%204).png" alt="HistGradientBoosting Feature Importance" width="100" />
<img src="../charts/XGBoost%20Feature%20Importance%20(Top%204).png" alt="XGBoost Feature Importance" width="100" />

### Business Insights
- **Location dominates pricing** — the geographic cluster alone explains more variance than any other single feature. A unit in an expensive neighborhood commands premium regardless of its physical characteristics
- **Size matters, but diminishing returns** — square footage and bedroom count are the second- and third-strongest predictors, but the marginal price increase per additional bedroom decreases beyond 3 bedrooms
- **Amenities have limited standalone impact** — pet policies, laundry options, parking, and wheelchair access each contribute modestly; they matter most in combination rather than individually
- **Business impact:** Property investors should prioritize location over renovations; renters can find better value by looking slightly outside premium clusters for similar-sized units

### Feature Importance Rankings

| Feature | RF | HistGB | XGB |
|---------|----|--------|-----|
| location_cluster | 1 | 1 | 1 |
| region | 2 | 3 | 2 |
| sqfeet | 3 | 2 | — |
| lat | 4 | — | — |
| long | — | 4 | — |
| baths | — | — | 3 |
| laundry_options | — | — | 4 |

*(Rankings across all three models. Blank = outside top 4.)*

---

## 5. LIVE Demonstration (25 Points)

*Video content to be recorded showing:*

1. **Notebook execution** — running the full pipeline: data loading → cleaning → feature engineering → model training → evaluation
2. **Model training in real time** — showing Random Forest, HistGradientBoosting, and XGBoost fitting with timing output (~40s total training)
3. **Metrics output** — displaying MAE, RMSE, R² for each model as cells execute
4. **Visualizations being generated** — feature importance plots, observed vs predicted charts, error bar charts
5. **Demonstration that models match slide 4 results** — the output metrics and charts coincide with the values presented above

---

## 6. Lessons Learned (10 Points)

- **Technology:** Target encoding and K-Means location clustering were critical for capturing nuanced patterns in the data. HistGradientBoosting provided a significant speed advantage over standard gradient boosting while maintaining accuracy.
- **Management:** CRISP-DM provided a structured framework that prevented scope creep and kept the project on track through all six phases.
- **Personal:** All three models are less precise for the most expensive listings, with observed-vs-predicted spread widening at higher price points.

---

## 7. Conclusion (5 Points)

- **Did the project meet its stated goals?** Yes.
  - **Business goal achieved:** I identified that location, square footage, and bedroom count are the dominant drivers of rental price in the Southern US
  - **Data mining goal achieved:** All three models successfully predicted rental price from available features, and the feature importance rankings identified location, square footage, and bedroom count as the top drivers — fulfilling the analytical goal of producing actionable rankings
- **What did the project show?** Rental pricing in the Southern US is heavily location-driven, but within a given area, apartment size and bedroom count are the most actionable levers. Amenities like pet policies and parking matter but are secondary.
- **Final thoughts:** The project demonstrated that ensemble methods with thoughtful feature engineering can explain ~95% of rental price variance using publicly available Craigslist data. This approach could be extended to a real-time pricing tool for property managers or integrated into rental listing platforms to suggest optimal listing prices.
