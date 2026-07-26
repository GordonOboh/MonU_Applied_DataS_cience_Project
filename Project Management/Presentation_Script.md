# Presentation Script — Housing Amenities Impact on Apartment Listing Prices in the Southern USA

> **Pacing:** ~120 wpm | **Total time:** ~16 min scripted + ~4 min live demo = 20 min

---

## Title Slide (~15 sec | ~30 words)

Good afternoon. My project is titled **"Housing Amenities Impact on Apartment Listing Prices in the Southern USA."**

---

## 1. Project Overview (~1 min | ~120 words)

The business problem is straightforward: renters, landlords, and property managers in the Southern US lack clear data on which features actually drive rental pricing. Without this insight, properties may be mispriced, and investment in upgrades may not yield the expected return.

The business goal is to identify the key features that influence rental prices for apartments in the Southern US, enabling data-informed pricing and investment decisions.

I had two data mining goals. First, build and compare three ensemble regression models — Random Forest, HistGradientBoosting, and XGBoost — to predict rental price from property features. Second, determine the relative importance of each amenity feature in driving that predicted price, producing an actionable ranking.

The dataset started at about 385,000 Craigslist rental listings across all 50 states, which was filtered down to roughly 150,000 apartment listings in 17 Southern states.

---

## 2. Summary of Each Phase (~3 min | ~360 words)

### Phase 1 — Business Understanding

I defined the business problem, established success criteria around producing actionable rankings, and created a project plan spanning all six CRISP-DM phases.

### Phase 2 — Data Understanding

I collected the dataset from Kaggle — the Austin Reese housing listings dataset. I explored distributions of price, square footage, beds, and baths, and mapped the geographic spread. This revealed several data quality issues like missing values in amenity columns, geographic outliers like listings with non-US coordinates, and extreme price outliers that needed handling.

### Phase 3 — Data Preparation

This was the most labor-intensive phase. I filtered to the 17 Census Bureau Southern states using both a lat-lon bounding box and a state list. I dropped uninformative columns like ID, URL, and image URLs. I applied IQR-based price filtering per region to handle outliers and it was computed separately for each region because market conditions differ dramatically between, say, rural Mississippi and downtown Atlanta.

I narrowed to apartments only for a focused analysis. For missing values, I imputed numeric columns with median, treated missing laundry and parking options as their own category, and state and region had no missing values. For feature engineering, I applied target encoding to high-cardinality categoricals and K-Means clustering on coordinates to create a location cluster feature that captures neighborhood-level price variation.

### Phase 4 — Modeling

I trained three ensemble regressors on an 80-20 train-test split: Random Forest, HistGradientBoosting, and XGBoost. I did hyperparameter tuning across several iterations, and all three final models achieved an R-squared above 0.93.

### Phase 5 — Evaluation

Random Forest performed best with an R-squared of 0.946, a mean absolute error of $35.58, and RMSE of $77.32. XGBoost came second at 0.939, and HistGradientBoosting at 0.934. All three models consistently identified location, square footage, and bedroom count as the top predictors. I assessed the results against the business goals and confirmed all success criteria were met.

### Phase 6 — Deployment

I documented a deployment plan and a monitoring and maintenance strategy, and prepared the final presentation and report.

---

## 3. Data Cleansing (~3 min | ~360 words)

I want to walk through the data cleansing in more detail because it was critical to the results.

**Geographic filtering.** I applied a Southern US bounding box spanning from Key West, Florida up to the northern West Virginia panhandle, and from far west Texas to the Maryland-Delaware coast. On top of that, I filtered to the 17 Census Bureau Southern states. This removed all non-Southern listings in one pass.

**Duplicate removal.** I dropped duplicate rows and duplicate IDs.

**Column selection.** I removed uninformative columns — ID, URL, image URL, and the description field — since these don't contribute to structured price prediction.

**Outlier treatment.** This is where I had to be careful. Rental prices vary enormously between markets — a luxury apartment in Atlanta and a modest unit in rural Mississippi are both legitimate listings but at very different price points. If I applied a single global IQR filter, I would have discarded valid high-end listings from expensive markets. So I computed the IQR separately for each region, removing only listings that were outliers within their own local market.

**Apartment-only filter.** I narrowed the dataset to apartment listings only, which gave me about 150,000 rows. This made the analysis more focused and homogeneous since I wasn't comparing apartments to houses or condos.

**Missing values.** For numeric columns like square footage, beds, baths, and the boolean amenity flags, I imputed with the median. For laundry and parking options, I treated missing values as an explicit "Missing" category — because the absence of information about laundry or parking is itself meaningful. State and region had no missing values, so no imputation was needed.

**Feature engineering.** I created a location cluster feature using K-Means clustering on latitude and longitude with 148 clusters, matching the number of unique Craigslist metro areas. This captured neighborhood-level price effects that state and region labels alone could not represent. I also applied target encoding to all categorical features — this converts each category to a numeric value based on its average price relationship, which is much more efficient than one-hot encoding for high-cardinality features like region with 148 levels.

---

## 4. Phase 4.0 — Modeling & Visualizations (~7 min | ~840 words)

### Models Built

Let me walk through the three models and their results.

**Random Forest** achieved an R-squared of 0.946 with a mean absolute error of $35.58. This means on average, its predictions were off by about $36 — roughly 3.5 percent of the average listing price, which is quite strong for this domain.

**HistGradientBoosting** achieved an R-squared of 0.934 with an MAE of $47.07.

**XGBoost** landed in between at 0.939 with an MAE of $46.47.

The gap between the best and worst model is only 0.012 in R-squared, so all three performed well. Random Forest came out on top.

### Geographic Filtering — Before vs After

*[Point to the before and after maps.]*

The first map shows the raw listings scattered across the entire US, including listings in Alaska, Hawaii, and even some outside the country. The second map shows the cleaned dataset limited to Southern states and apartments only. <!--You can see the listings now cluster in population centers across the South — from Texas through to Florida and up the East Coast.-->

### Correlation with Price

*[Point to the Pearson and Spearman heatmaps.]*

These heatmaps show how each feature correlates with price. Pearson captures linear relationships, Spearman captures rank-order relationships. You can see that square footage, beds, and baths have moderate positive correlations, while most of the boolean amenities show very weak correlation. This tells me that amenities alone don't drive price — it's mainly location and size.

### Model Performance Charts

*[Point to the R-squared, MAE, RMSE, and ASE bar charts.]*

These bar charts make the comparison easy. Random Forest leads across every metric — highest R-squared, lowest error. HistGradientBoosting and XGBoost are close behind. The consistent ranking across all four metrics gives me confidence that Random Forest is genuinely the best model for this data.

### Observed vs Predicted

*[Point to the three scatter plots.]*

These plots show predicted price on one axis versus actual price on the other. A perfect model would have all points on the diagonal line. Random Forest shows the tightest clustering around the diagonal, confirming its superior performance. You can see that all three models handle mid-range prices well, but the spread widens at higher prices — above roughly $2,000 — meaning predictions for expensive units are less reliable. This is a limitation worth noting.

### Feature Importance

*[Point to the feature importance charts.]*

*[Point to the ranking table.]*

Here is a key finding: location cluster is ranked number one across all three models. Not just one model — all three. Region is number two in two models and three in the third. Square footage is consistently in the top three. This cross-model agreement is strong evidence that these are genuinely the most important drivers, not artifacts of a particular algorithm.

Notably, latitude appears in Random Forest's top four and longitude in HistGradientBoosting's top four, but neither appears in XGBoost's top four <!-- — which makes sense because XGBoost has region and location cluster already capturing that geographic signal.-->

### Business Insights

What does this mean in practical terms?

**Location dominates pricing.** The geographic cluster explains more variance than any other single feature. A unit in an expensive neighborhood commands a premium regardless of its physical characteristics. This is the single most important takeaway.

**Size matters, but with diminishing returns.** Square footage and bedroom count are the second- and third-strongest predictors, but the marginal price increase per additional bedroom decreases beyond three bedrooms. Adding a fourth bedroom doesn't add as much value as the first three.

**Amenities have limited standalone impact.** Pet policies, laundry options, parking — these each contribute modestly on their own. They matter most in combination rather than individually.

**The business impact** is clear: property investors should prioritize location over renovations when possible, and renters can find better value by looking slightly outside premium clusters for comparable sized units.

---

## 5. LIVE DEMONSTRATION (~4 min)

*[Transition to live demo — show Jupyter notebook running:*

1. *Load the CSV and run the cleaning cells*
2. *Show feature engineering steps — target encoding and K-Means clustering*
3. *Run model training for all three models — point out the timing output*
4. *Display the metrics output and compare to the values shown on slides*
5. *Generate a feature importance plot and an observed-vs-predicted plot live*

*]*

---

## 6. Lessons Learned (~1.5 min | ~180 words)

**Technology lessons.** Target encoding and K-Means location clustering were critical for capturing nuanced patterns in the data. Without them, model performance was noticeably weaker. HistGradientBoosting provided a significant speed advantage over standard gradient boosting while maintaining comparable accuracy — a useful trade-off to know about.

**Management lessons.** The CRISP-DM framework provided structure that prevented scope creep and kept the project on track through all six phases. Having a clear phase-by-phase plan made it easier to know what to focus on at each stage.

**Personal lessons.** All three models are less precise for the most expensive listings. The observed-versus-predicted plots show the spread widens at higher price points, meaning the model is reliable for mid-range pricing but needs refinement for luxury properties.

---

## 7. Conclusion (~30 sec | ~60 words)

Did the project meet its stated goals? Yes.

The business goal was achieved — I identified that location, square footage, and bedroom count are the dominant drivers of rental price in the Southern US. The data mining goal was also achieved — all three models successfully predicted rental price, and the feature importance rankings produced actionable insights.

The bottom line: rental pricing in the Southern US is heavily location-driven. But within a given area, apartment size and bedroom count are the most actionable levers for pricing decisions. Amenities matter, but they're secondary.

Thank you.

about 15min without code running
