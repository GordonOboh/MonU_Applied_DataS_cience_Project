# Task 1.0 Business Understanding — Planning Document
> CS703 Applied Data Science Project — Week 1

## Overview

Four tasks, each with one deliverable. All content scoped to the approved project:
**Housing Amenities Impact on Apartment Listing Prices in the Southern USA**
(Austin TX, Dallas TX, Miami FL, Atlanta GA, Nashville TN, Charlotte NC)

---

## Task 1.1: Determine Business Objectives
**Deliverable:** Project Scope Document Part 1

- Business objective: identify which amenity features most drive apartment rental price in Southern USA markets
- Target audience: real estate investors and property managers operating in Southern markets
- Business questions:
  - Which amenities have the strongest impact on rental pricing?
  - How does amenity importance vary across geographic regions (Austin vs. Miami vs. Atlanta)?
  - Which amenity upgrades yield the highest price premium relative to cost?
  - Do certain amenity bundles (parking + laundry) have compounding effects on price?
- Success criteria: model identifies top amenity drivers; output is actionable for pricing and renovation decisions

---

## Task 1.2: Assess the Situation
**Deliverable:** Project Scope Document Part 2

- Stakeholders: investors (maximize ROI on upgrades), managers (competitive pricing vs. local market)
- Data source: Kaggle Craigslist scrape, January 2020 (384,977 rows, 22 columns, 558 MB)
- Constraints:
  - 2020 dataset only, no real-time pricing data
  - No GPU/TPU hardware available, limits NLP on description field
  - Geographic scope restricted to 6 Southern cities
- Risks:
  - Missing values in laundry (20.5%) and parking (36.5%) fields
  - Anomalous records ($0 prices, <1 sqft listings)
  - Outlier contamination skewing regional price distributions
- Contingencies:
  - IQR-based outlier removal applied at state/region level
  - Missing categoricals imputed as "missing" category

---

## Task 1.3: Determine Data-Mining Goals
**Deliverable:** Data-Mining Scope Document

- Goal 1 (Predictive): predict apartment rental price using all available features
- Goal 2 (Analytical): rank amenity features by relative importance in driving predicted price
- Models: Random Forest, Gradient Boosting, XGBoost
- Feature engineering: K-Means clustering on lat/long (125 clusters) to create location_cluster feature
- Success metrics: MAE, RMSE, R² on held-out test set (20%)

---

## Task 1.4: Produce a Project Plan
**Deliverable:** Data Mining Project/Resource Plan

### Tools and Hardware

| Category | Details |
|----------|---------|
| Language / IDE | Python 3.12.3, VS Code, Jupyter Notebook |
| Key Libraries | pandas 3.0.1, numpy 2.4.2, matplotlib 3.10.8, scipy 1.17.1, scikit-learn 1.8.0, xgboost 3.2.0, category-encoders 2.9.0, joblib 1.5.3 |
| Data Source | Kaggle (Craigslist scrape, Jan 2020) |
| Hardware | 2 vCPU, 2 GB RAM, Ubuntu 24 Server |

### Complete Project Timeline

| Task | Description | Original Start | Original End |
|------|-------------|----------------|--------------|
| 1.1 | Determine Business Objectives | 2 May 2026 | 9 May 2026 |
| 1.2 | Assess the Situation | 2 May 2026 | 9 May 2026 |
| 1.3 | Determine Data-Mining Goals | 2 May 2026 | 9 May 2026 |
| 1.4 | Produce a Project Plan | 2 May 2026 | 9 May 2026 |
| 2.1 | Gather Data | 9 May 2026 | 23 May 2026 |
| 2.2 | Describe Data | 9 May 2026 | 23 May 2026 |
| 2.3 | Explore Data | 9 May 2026 | 23 May 2026 |
| 2.4 | Verify Data Quality | 9 May 2026 | 23 May 2026 |
| 3.1 | Select Data | 23 May 2026 | 6 Jun 2026 |
| 3.2 | Clean Data | 23 May 2026 | 6 Jun 2026 |
| 3.3 | Construct Data | 23 May 2026 | 6 Jun 2026 |
| 3.4 | Integrate Data | 23 May 2026 | 6 Jun 2026 |
| 3.5 | Format Data | 23 May 2026 | 6 Jun 2026 |
| -- | Midterm Presentation | -- | 13 Jun 2026 |
| 4.1 | Select Modeling Technique | 6 Jun 2026 | 20 Jun 2026 |
| 4.2 | Generate Test Design | 6 Jun 2026 | 20 Jun 2026 |
| 4.3 | Build Model | 6 Jun 2026 | 20 Jun 2026 |
| 4.4 | Assess Model | 6 Jun 2026 | 20 Jun 2026 |
| 5.1 | Evaluate Results | 20 Jun 2026 | 4 Jul 2026 |
| 5.2 | Review Process | 20 Jun 2026 | 4 Jul 2026 |
| 5.3 | Determine Next Steps | 20 Jun 2026 | 4 Jul 2026 |
| 6.1 | Plan Deployment | 4 Jul 2026 | 11 Jul 2026 |
| 6.2 | Plan Monitoring | 4 Jul 2026 | 11 Jul 2026 |
| 6.3 | Final Report and Presentation | 11 Jul 2026 | 1 Aug 2026 |
| 6.4 | Review Project | 11 Jul 2026 | 1 Aug 2026 |

---

## LaTeX Structure (Phase_1.tex)

```
\section*{Phase 1: 1.0 Business Understanding}
  \subsection*{Task 1.1: Determine Business Objectives}
    \subsubsection*{Deliverable: Project Scope Document -- Part 1}
  \subsection*{Task 1.2: Assess the Situation}
    \subsubsection*{Deliverable: Project Scope Document -- Part 2}
  \subsection*{Task 1.3: Determine Data-Mining Goals}
    \subsubsection*{Deliverable: Data-Mining Scope Document}
  \subsection*{Task 1.4: Produce a Project Plan}
    \subsubsection*{Deliverable: Data Mining Project/Resource Plan}
```

Style rules: `\begin{itemize}[itemsep=-9pt]` for bullets, `tabular` for tables, no em dashes (use commas/colons/parentheses).
