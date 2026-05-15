# Weekly Status Report — Week 2
> CS703 Applied Data Science Project — Professor Nicholas Nardi
> Each week: create new status, prepend to previous week's PPTX. By end of semester = 1 PPTX with all weeks stacked.

---

> [!NOTE]
> **BEFORE YOU START EACH WEEK**
> 1. Make a copy of `CS703 - Weekly Status Report Template.pptx`
> 2. Rename the copy to `CS703 - Week[X] Status Report.pptx` (e.g., `CS703 - Week1 Status Report.pptx`)
> 3. Fill in the renamed copy — never edit the original template directly

---

> [!IMPORTANT]
> **BULLET POINT HIERARCHY — ALWAYS ENFORCE**
> - Level 0: Main items (task names, phase headers, section headers: Issues/Risks/Concerns)
> - Level 1: Sub-items (status, dates, descriptions, mitigation notes)
> - Level 2: Further detail under a Level 1 item (e.g., sub-note under a date, clarification under a description)
> - Up to 3 levels maximum. Never mix levels out of order. Every item sits under its logical parent.

---

## SLIDE 1 — Title
- **Project Name:** Housing Amenities Impact on Apartment Listing Prices in the Southern USA
- **Week Ending:** May 22, 2026
- **Submitted by:** Gordon Oboh
- **Submit Date:** May 23, 2026

---

## SLIDE 2 — Agenda (FIXED — never changes)
1. Status Overview
2. Items Completed This Week
3. Items In Progress
4. Items To Be Started
5. Samples of Items Completed This Week
6. Issues, Risks, Concerns
7. Next Steps
8. Personal Reflection

---

## SLIDE 3 — Status Overview
- **Project Status:** GREEN

- Task 1.4: Produce Project Plan — Level 0
  - Task is GREEN — 100% complete — Level 1
  - Original Start: 2 May 2026 | Revised Start: N/A | Actual Start: 7 May 2026 — Level 1
  - Original End:   9 May 2026 | Revised End:   N/A | Actual End: 13 May 2026 — Level 1
  - Completed 4 days after original end date — Level 1

- Task 2.1: Gathering Data — Deliverable: Data Collection Report — Level 0
  - Task is GREEN — 100% complete — Level 1
  - Original Start: 11 May 2026 | Revised Start: N/A | Actual Start: 12 May 2026 — Level 1
  - Original End:   23 May 2026 | Revised End:   N/A | Actual End: 23 May 2026 — Level 1

- Task 2.2: Describing Data — Deliverable: Data Description Report — Level 0
  - Task is GREEN — 100% complete — Level 1
  - Original Start: 11 May 2026 | Revised Start: N/A | Actual Start: 12 May 2026 — Level 1
  - Original End:   23 May 2026 | Revised End:   N/A | Actual End: 23 May 2026 — Level 1

- Task 2.3: Exploring Data — Deliverable: Data Exploration Report — Level 0
  - Task is GREEN — 100% complete — Level 1
  - Original Start: 11 May 2026 | Revised Start: N/A | Actual Start: 12 May 2026 — Level 1
  - Original End:   23 May 2026 | Revised End:   N/A | Actual End: 23 May 2026 — Level 1

- Task 2.4: Verifying Data Quality — Level 0
  - Task is GREEN — 100% complete — Level 1
  - Original Start: 11 May 2026 | Revised Start: N/A | Actual Start: 12 May 2026 — Level 1
  - Original End:   23 May 2026 | Revised End:   N/A | Actual End: 23 May 2026 — Level 1

---

## SLIDE 4 — Items Completed This Week
> Only tasks FULLY completed THIS week (not a running list). If nothing completed: "Per plan, no items completed this week."

- Task 1.4: Produce Project Plan — Deliverable: Data Mining Project/Resource Plan — Level 0
  - Task is GREEN — 100% complete — Level 1
  - Original Start: 2 May 2026 | Revised Start: N/A | Actual Start: 7 May 2026 — Level 1
  - Original End:   9 May 2026 | Revised End:   N/A | Actual End: 13 May 2026 — Level 1
  - Full CRISP-DM project timeline built out with task dates, deliverables, and resource plan — Level 1
  - Completed 4 days after original end; delay due to scope of CRISP-DM timeline build-out required — Level 1

- Task 2.1: Gathering Data — Deliverable: Data Collection Report — Level 0
  - Task is GREEN — 100% complete — Level 1
  - Original Start: 11 May 2026 | Revised Start: N/A | Actual Start: 12 May 2026 — Level 1
  - Original End:   23 May 2026 | Revised End:   N/A | Actual End: 23 May 2026 — Level 1
  - Dataset sourced from Kaggle: Craigslist housing listings scraped January 2020; 384,977 rows, 22 columns, 558.44 MB — Level 1

- Task 2.2: Describing Data — Deliverable: Data Description Report — Level 0
  - Task is GREEN — 100% complete — Level 1
  - Original Start: 11 May 2026 | Revised Start: N/A | Actual Start: 12 May 2026 — Level 1
  - Original End:   23 May 2026 | Revised End:   N/A | Actual End: 23 May 2026 — Level 1
  - Features classified into numeric (price, sqfeet, beds, baths), categorical binary (amenity flags), categorical multi-class (region, type, laundry_options, parking_options), geographic (lat, long, state), and text/URL (excluded from modeling) — Level 1

- Task 2.3: Exploring Data — Deliverable: Data Exploration Report — Level 0
  - Task is GREEN — 100% complete — Level 1
  - Original Start: 11 May 2026 | Revised Start: N/A | Actual Start: 12 May 2026 — Level 1
  - Original End:   23 May 2026 | Revised End:   N/A | Actual End: 23 May 2026 — Level 1
  - Summary statistics computed; apartments most common listing type; Q3 rental price $1,395, Q3 square footage 1,150 sqft; majority pet-friendly; furnished, EV charging, and wheelchair access represent minority listings; roughly 25% of listings smoke-friendly — Level 1

- Task 2.4: Verifying Data Quality — Level 0
  - Task is GREEN — 100% complete — Level 1
  - Original Start: 11 May 2026 | Revised Start: N/A | Actual Start: 12 May 2026 — Level 1
  - Original End:   23 May 2026 | Revised End:   N/A | Actual End: 23 May 2026 — Level 1
  - Missing value analysis complete: laundry_options 20.5% missing, parking_options 36.5% missing, lat/long 0.5% missing; anomalies identified including $0 price listings and sub-1 sqft listings; all quality issues documented for remediation in Data Preparation phase — Level 1

---

## SLIDE 5 — Items In Progress

- Task 3.1: Select Data — Level 0
  - Task is GREEN — 25% complete — Level 1
  - Original Start: 24 May 2026 | Revised Start: N/A | Actual Start: 12 May 2026 — Level 1
  - Original End:   6 Jun 2026 | Revised End:   N/A | Actual End: N/A — Level 1
  - Started ahead of schedule; initial column selection in progress (dropping non-predictive columns: id, url, image_url, description) — Level 1

---

## SLIDE 6 — Items To Be Started
> Always show 2-3 upcoming items. Scope to subsections of current CRISP-DM phase AND the next CRISP-DM phase.

- Task 3.2: Clean Data — Level 0
  - Original Start: 24 May 2026 | Revised Start: N/A | Actual Start: N/A — Level 1
  - Original End:   6 Jun 2026 | Revised End:   N/A | Actual End: N/A — Level 1

- Task 3.3: Construct Data — Level 0
  - Original Start: 24 May 2026 | Revised Start: N/A | Actual Start: N/A — Level 1
  - Original End:   6 Jun 2026 | Revised End:   N/A | Actual End: N/A — Level 1

- Task 4.1: Select Modeling Technique — Level 0
  - Original Start: 7 Jun 2026 | Revised Start: N/A | Actual Start: N/A — Level 1
  - Original End:   20 Jun 2026 | Revised End:   N/A | Actual End: N/A — Level 1

---

## SLIDE 7 — Sample of Items Completed This Week
> Show 1 significant artifact from this week. Options: image, code snippet, table, document excerpt.

- **What:** Dataset Acquisition and Exploratory Distributions — Price and Square Footage
- **Why:** The Kaggle Craigslist dataset (384,977 rows, 558 MB) is the foundation for all subsequent analysis. Confirming the data was successfully acquired and understanding the distributions of the two most important numeric features — rental price and square footage — validates that the Data Understanding phase produced a usable dataset for Southern USA modeling.
- **Content:**

  > **Data Source:** Kaggle — Craigslist Housing Listings (January 2020 scrape)
  > Dataset: 384,977 rows × 22 columns, 558.44 MB
  > Target variable: `price` (monthly rental in USD)
  >
  > **Price Distribution (Southern States Subset)**
  > Right-skewed; majority of listings concentrated below $3,000/month; median near $1,100; long upper tail with outliers above $10,000 identified for removal in Data Preparation.
  >
  > **Square Footage Distribution (Southern States Subset)**
  > Right-skewed; Q3 = 1,150 sqft; extreme outliers (< 1 sqft, impossibly large values) flagged as anomalies; IQR-based removal planned per state/region in Task 3.2.

---

## SLIDE 8 — Issues, Risks, Concerns
> Issues = real/already happened | Risks = potential future | Concerns = external, out of control

- Issues — Level 0
  - None at this time — Level 1

- Risks — Level 0
  - None at this time — Level 1
 <!--- Dataset currency: Craigslist data scraped January 2020; rental market conditions have changed significantly since then — Level 1
    - Impact: model predictions may not reflect current Southern USA rental pricing dynamics — Level 2
    - Mitigation: scope findings to historical 2020 market; clearly document dataset date in all deliverables — Level 2
  - Missing value concentration: laundry_options (20.5%) and parking_options (36.5%) have substantial missing rates — Level 1
    - Impact: imputation strategy will influence model results for two key amenity features — Level 2
    - Mitigation: impute as "missing" category (separate from "no laundry" / "no parking") to preserve signal — Level 2 --> 

- Concerns — Level 0
  - None at this time — Level 1

---

## SLIDE 9 — Next Steps
> Scope to current CRISP-DM section (subsections) + next CRISP-DM section only.

- Task 3.1: Select Data (continuing) — Level 0
  - Original Start: 24 May 2026 | Revised: N/A | Actual Start: 12 May 2026 — Level 1
  - Original End:   6 Jun 2026 | Revised: N/A | Actual End: N/A — Level 1

- Task 3.2: Clean Data — Level 0
  - Original Start: 24 May 2026 | Revised: N/A | Actual Start: N/A — Level 1
  - Original End:   6 Jun 2026 | Revised: N/A | Actual End: N/A — Level 1

- Task 3.3: Construct Data — Level 0
  - Original Start: 24 May 2026 | Revised: N/A | Actual Start: N/A — Level 1
  - Original End:   6 Jun 2026 | Revised: N/A | Actual End: N/A — Level 1

- Task 3.4: Integrate Data — Level 0
  - Original Start: 24 May 2026 | Revised: N/A | Actual Start: N/A — Level 1
  - Original End:   6 Jun 2026 | Revised: N/A | Actual End: N/A — Level 1

- Task 3.5: Format Data — Level 0
  - Original Start: 24 May 2026 | Revised: N/A | Actual Start: N/A — Level 1
  - Original End:   6 Jun 2026 | Revised: N/A | Actual End: N/A — Level 1

- Task 4.1: Select Modeling Technique — Level 0
  - Original Start: 7 Jun 2026 | Revised: N/A | Actual Start: N/A — Level 1
  - Original End:   20 Jun 2026 | Revised: N/A | Actual End: N/A — Level 1

- Task 4.2: Generate Test Design — Level 0
  - Original Start: 7 Jun 2026 | Revised: N/A | Actual Start: N/A — Level 1
  - Original End:   20 Jun 2026 | Revised: N/A | Actual End: N/A — Level 1

---

## SLIDE 10 — Personal Reflection
> Thoughts and feelings — NOT an activity log. ~1 paragraph. Only you and Professor Nardi see this.

The Data Understanding phase went better than expected this week — having all four tasks (2.1 through 2.4) wrapped up on time gave me a real sense of momentum after Task 1.4 running a few days late. What surprised me most was the sheer size of the dataset; nearly 385,000 rows felt overwhelming at first, but once I started working through the distributions and quality checks it became clear there is strong signal in the data. The missing value rates for laundry and parking options (20% and 36%) are higher than I hoped, but finding them now means the Data Preparation phase has a clear plan. I am also slightly ahead of schedule on Task 3.1, which feels good given the midterm presentation is on the horizon. My confidence in the project direction is high.

---

## Date Format Reference

> [!NOTE]
> **Professor Nardi's exact guidance on dates:**
>
> **ORIGINAL START/END DATE:** When you create your approved project plan, every task is given a start date and an end date. These are the ORIGINAL Start and End Dates. These dates NEVER change.
>
> **ACTUAL START/END DATE:** Hopefully, most tasks start on the date we planned and end on the date we planned. But that is not always the case. Sometimes we start or end a task a day or two (or more) earlier or later. The date that you ACTUALLY start or end a task is the ACTUAL Start and End Date. Only started and/or completed tasks have Actual Start/End Dates.
>
> **REVISED START/END DATE:** We plan with all good intentions. Then life happens and we are sometimes forced to make major revisions to our plan. In that case, we need to create revised dates. Revised dates are RARELY used. They are not intended to correct poor planning. YOU CANNOT USE A REVISED DATE UNLESS YOU DISCUSS WITH ME AND GET MY APPROVAL.

| Type | When to use | Changes? |
|------|------------|---------|
| Original Start/End | Set when plan approved | Never |
| Actual Start/End | When task actually starts/ends | Only for started/completed tasks |
| Revised Start/End | Major plan change — requires professor approval | Rarely — not for poor planning |
| N/A | No actual or revised date yet | Use as placeholder always |

## CRISP-DM Phase Reference
| Phase | Description |
|-------|------------|
| 1.0 Business Understanding | Define business problem, objectives, project plan |
| 1.1 Determine Business Objectives | |
| 1.2 Assess Situation | |
| 1.3 Determine Data Mining Goals | |
| 1.4 Produce Project Plan | |
| 2.0 Data Understanding | Collect, explore, verify data |
| 3.0 Data Preparation | Clean, transform, engineer features |
| 4.0 Modeling | Build and tune models |
| 5.0 Evaluation | Assess results against business goals |
| 6.0 Deployment | Deliver findings and documentation |
