# CS703 Applied Data Science — Master Schedule

**Project:** Housing Amenities Impact on Apartment Listing Prices in the Southern USA
**Course:** CS703-152W Applied Data Science Project
**Professor:** Nicholas Nardi
**Semester:** May 2026 – August 2026

---

## Schedule Summary

| Week | Submission Date | CRISP-DM Phase Due | Major Deliverable | Status Report | Status |
|------|-----------------|--------------------|-------------------|---------------|--------|
| Week 1  | 9 May 2026  | 1.0 Business Understanding | Business Understanding Doc | Due 9 May  | [ ] |
| Week 2  | 16 May 2026 | *(buffer)*                 | —                         | Due 16 May | [ ] |
| Week 3  | 23 May 2026 | 2.0 Data Understanding     | Data Understanding Doc    | Due 23 May | [ ] |
| Week 4  | 30 May 2026 | *(buffer)*                 | —                         | Due 30 May | [ ] |
| Week 5  | 6 Jun 2026  | 3.0 Data Preparation       | Data Preparation Doc      | Due 6 Jun  | [ ] |
| Week 6  | 13 Jun 2026 | Midterm Presentation       | Midterm Presentation      | Due 13 Jun | [ ] |
| Week 7  | 20 Jun 2026 | 4.0 Data Modeling          | Data Modeling Doc         | Due 20 Jun | [ ] |
| Week 8  | 27 Jun 2026 | *(buffer)*                 | —                         | Due 27 Jun | [ ] |
| Week 9  | 4 Jul 2026  | 5.0 Modeling Evaluation    | Evaluation Doc            | Due 4 Jul  | [ ] |
| Week 10 | 11 Jul 2026 | 6.0 Data Deployment        | Deployment Doc            | Due 11 Jul | [ ] |
| Week 11 | 18 Jul 2026 | *(buffer)*                 | —                         | Due 18 Jul | [ ] |
| Week 12 | 25 Jul 2026 | Task 6.3 Final Presentation | Final Presentation        | LAST Report | [ ] |
| Final   | 1 Aug 2026  | —                          | Final Report              | —           | [ ] |

---

## Legend

| Checkbox | Meaning |
|----------|---------|
| `[x]` | Done |
| `[-]` | In Progress |
| `[ ]` | Not Started |

---

## Week 1 — Week Ending 9 May 2026

**Phase Due:** 1.0 Business Understanding

- [-] Status Report submitted

### CRISP-DM Tasks
- [-] 1.1 Determine Business Objectives
  - [-] Define business problem and stakeholder goals
  - [-] Document project scope (Project Scope Document Part 1)
- [-] 1.2 Assess the Situation
  - [-] Identify data sources and constraints
  - [-] Document risks and contingencies (Project Scope Document Part 2)
- [-] 1.3 Determine Data-Mining Goals
  - [-] Define predictive goal (predict rental price)
  - [-] Define analytical goal (feature importance ranking)
  - [-] Document data-mining scope
- [-] 1.4 Produce Project Plan
  - [-] Define CRISP-DM phase timeline
  - [-] Document tools and hardware

### Deliverable
- [-] Submit 1.0 Business Understanding document (PDF)

---

## Week 2 — Week Ending 16 May 2026

**Phase Due:** *(none — buffer week before 2.0)*

- [ ] Status Report submitted

### CRISP-DM Tasks
- [ ] Begin 2.0 Data Understanding work

---

## Week 3 — Week Ending 23 May 2026

**Phase Due:** 2.0 Data Understanding

- [ ] Status Report submitted

### CRISP-DM Tasks
- [ ] 2.1 Gather Data
  - [ ] Source dataset (Kaggle — Craigslist housing listings, Jan 2020)
  - [ ] Confirm dataset size (384,977 rows, 22 columns, 558 MB)
- [ ] 2.2 Describe Data
  - [ ] Classify features (numeric, categorical binary, categorical multi-class, geographic, text)
  - [ ] Identify target variable (price)
  - [ ] Document feature descriptions
- [ ] 2.3 Explore Data
  - [ ] Summary statistics (quartiles, distributions)
  - [ ] Property type breakdown (apartments most common)
  - [ ] Visualize geographic distribution
  - [ ] Distribution plots (price, sqfeet, beds, baths)
  - [ ] Categorical breakdowns (pet policies, amenities)
- [ ] 2.4 Verify Data Quality
  - [ ] Missing value analysis (laundry 20.5%, parking 36.5%, lat/long 0.5%)
  - [ ] Identify anomalies ($0 listings, <1 sqft listings)
  - [ ] Document quality issues

### Deliverable
- [ ] Submit 2.0 Data Understanding document (PDF)

---

## Week 4 — Week Ending 30 May 2026

**Phase Due:** *(none — buffer week before 3.0)*

- [ ] Status Report submitted

### CRISP-DM Tasks
- [ ] Begin 3.0 Data Preparation work

---

## Week 5 — Week Ending 6 June 2026

**Phase Due:** 3.0 Data Preparation

- [ ] Status Report submitted

### CRISP-DM Tasks
- [ ] 3.1 Select Data
  - [ ] Drop non-predictive columns (id, url, image_url, description)
  - [ ] Confirm retained features list
- [ ] 3.2 Clean Data
  - [ ] Remove price < $0 records
  - [ ] Remove sqfeet < 1 records
  - [ ] Filter lat/long to USA geographic bounds
  - [ ] Check for duplicates (none found)
  - [ ] Apply IQR outlier removal by state/region
- [ ] 3.3 Construct Data
  - [ ] Engineer location_cluster feature (K-Means, 125 clusters on lat/long)
  - [ ] 80/20 train/test split
- [ ] 3.4 Integrate Data
  - [ ] Confirm single dataset (no integration required)
- [ ] 3.5 Format Data
  - [ ] Separate numeric and categorical columns
  - [ ] Impute missing values in laundry_options and parking_options ("missing" category)
  - [ ] Apply TargetEncoder to multi-categorical features

### Deliverable
- [ ] Submit 3.0 Data Preparation document (PDF)

---

## Week 6 — Week Ending 13 June 2026

**Phase Due:** Midterm Presentation

- [ ] Status Report submitted

### Deliverable
- [ ] Midterm Presentation (covers phases 1.0–3.0)

---

## Week 7 — Week Ending 20 June 2026

**Phase Due:** 4.0 Data Modeling

- [ ] Status Report submitted

### CRISP-DM Tasks
- [ ] 4.1 Select Modeling Technique
  - [ ] Select Random Forest
  - [ ] Select Gradient Boosting
  - [ ] Select XGBoost
- [ ] 4.2 Generate Test Design
  - [ ] Define 80/20 train/test split strategy
- [ ] 4.3 Build Model
  - [ ] Train Random Forest (~330 sec)
  - [ ] Train Gradient Boosting (~250 sec)
  - [ ] Train XGBoost (~9 sec)
  - [ ] Generate predictions on test set
- [ ] 4.4 Assess Model
  - [ ] Calculate MAE, MSE, RMSE, R², ASE, SSE per model
  - [ ] Generate feature importance plots
  - [ ] Generate observed vs. predicted scatter plots
  - [ ] Compare models via metric bar charts

### Deliverable
- [ ] Submit 4.0 Data Modeling document (PDF)

---

## Week 8 — Week Ending 27 June 2026

**Phase Due:** *(none — buffer week before 5.0)*

- [ ] Status Report submitted

### CRISP-DM Tasks
- [ ] Begin 5.0 Evaluation work

---

## Week 9 — Week Ending 4 July 2026

**Phase Due:** 5.0 Modeling Evaluation

- [ ] Status Report submitted

### CRISP-DM Tasks
- [ ] 5.1 Evaluate Results
  - [ ] Compare model metrics (MAE, RMSE, R²)
  - [ ] Identify best-performing model
  - [ ] Analyze feature importance rankings
  - [ ] Identify best iteration per model (ASE/SSE plots)
- [ ] 5.2 Review Process
  - [ ] Assess whether data-mining goals were met
  - [ ] Document process gaps and lessons learned
- [ ] 5.3 Determine Next Steps
  - [ ] Decide: deploy or revise
  - [ ] Document recommended improvements (cluster tuning, more data, GPU boosting, NLP on description)

### Deliverable
- [ ] Submit 5.0 Modeling Evaluation document (PDF)

---

## Week 10 — Week Ending 11 July 2026

**Phase Due:** 6.0 Data Deployment

- [ ] Status Report submitted

### CRISP-DM Tasks
- [ ] 6.1 Plan Deployment
  - [ ] Define deployment approach (or document revision decision)
- [ ] 6.2 Plan Monitoring and Maintenance
  - [ ] Define model monitoring strategy
- [ ] 6.3 Produce Final Report and Presentation
  - [ ] Draft final report
  - [ ] Build final presentation slides
- [ ] 6.4 Review Project
  - [ ] Document lessons learned
  - [ ] Summarize project outcomes

### Deliverable
- [ ] Submit 6.0 Data Deployment document (PDF)

---

## Week 11 — Week Ending 18 July 2026

**Phase Due:** *(none — buffer week before final deliverables)*

- [ ] Status Report submitted

### Tasks
- [ ] Finalize final report
- [ ] Finalize final presentation

---

## Week 12 — Week Ending 25 July 2026 *(LAST Status Report)*

**Phase Due:** Task 6.3 Final Presentation

- [ ] LAST Status Report submitted

### Deliverable
- [ ] Final Presentation delivered (Task 6.3)

---

## Final Deadline — 1 August 2026

### Deliverable
- [ ] Final Report submitted (Task 6.3)

---

## Current Position (as of 6 May 2026)

> **Schedule says:** Week 1 — working on 1.0 Business Understanding (due 9 May)
>
> **Actual work done:** Phases 1–5 complete in notebook (`Capstone_Final.ipynb`) and phases 1–3 documented in report (`CS703_Applied_Data_Science_report.pdf`)
>
> **Gap:** Phase documents (PDFs) for phases 2.0–6.0 not yet submitted. Phase 6 not implemented. Process review (5.2–5.3) not fully written up.
>
> **Bottom line:** Technical work is well ahead of schedule. Remaining work is documentation, Phase 6 write-up, and final deliverables.
