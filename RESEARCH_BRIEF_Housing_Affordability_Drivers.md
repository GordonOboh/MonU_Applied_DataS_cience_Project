# Research Brief: What Drives Rental Housing Costs Across the United States?

**Author:** Gordon Oboh  
**Date:** July 2026  
**Contact:** goboh@position.8shield.net

---

## Executive Summary

This brief summarizes a national-scale analysis of rental housing listings across all 50 U.S. states, drawing on approximately 385,000 Craigslist rental postings. Using ensemble machine learning methods, we identified the property features that most strongly influence rental prices. The goal is to provide an evidence-based understanding of housing cost drivers: a foundational step for analyzing housing affordability, identifying at-risk populations, and informing policy interventions aimed at preventing housing instability.

---

## Key Findings

1. **Location is the dominant driver of rental cost.** Geographic region, state, and local market cluster account for the majority of explained price variance across all models tested.

2. **Unit characteristics matter, but less than place.** Square footage, bedroom count, and bathroom count are the next most influential features. Together with location, they explain over 90% of rental price variation.

3. **Amenities have measurable but secondary effects.** Pet policies, laundry and parking options, and property type contribute modest predictive value beyond the core location and size variables.

4. **Rental costs vary dramatically across regions.** Listings range from under $500/month in some markets to over $3,000/month in high-cost metros: a spread that directly maps to housing affordability challenges facing low-income families.

---

## Data and Methodology

**Dataset:** ~385,000 unique rental listings from Craigslist covering all 50 states. After cleaning and filtering (geographic bounding boxes, IQR-based price trimming), approximately 363,000 listings remained for analysis.

**Features analyzed:** Square footage, bedrooms, bathrooms, property type, pet policies, laundry/parking options, state, region, and geographic cluster (via K-Means, 125 clusters).

**Models:** Three ensemble regression models were trained on an 80/20 train-test split:
- Random Forest (R² = 0.922)
- XGBoost (R² = 0.858)
- Gradient Boosting (R² = 0.799)

Random Forest achieved the best performance, explaining approximately 92% of variance in rental prices. Full methodology is documented in the companion technical report.

---

## Relevance to Housing Instability

Housing affordability is the primary structural driver of housing instability and homelessness (Joint Center for Housing Studies, 2025; National Low Income Housing Coalition, 2024). Understanding which factors drive rental costs at a national scale is a necessary step toward:

- **Identifying high-risk markets** where families are most cost-burdened
- **Forecasting affordability trends** as input to program planning and advocacy
- **Targeting interventions** such as rental assistance, voucher programs, and affordable housing development
- **Supporting evidence-informed policy** at local, state, and federal levels

This analysis provides a reproducible framework for studying rental market dynamics using publicly available data: a methodology that can be extended to focus specifically on low-cost listings, voucher-accepting units, or neighborhood-level instability indicators.

---

## Limitations

- Craigslist data is a convenience sample and may not fully represent the formal rental market, including subsidized and public housing units
- The analysis captures listed price, not actual rent paid, which may differ due to concessions, vouchers, or negotiated rates
- Census-level demographic and socioeconomic variables (e.g., median income, poverty rate, eviction rate) were not included but would strengthen future work on the link between pricing and instability

---

## Next Steps

1. **Integrate public data sources** (ACS, HUD, Eviction Lab) to connect rental pricing patterns with instability outcomes at the census tract level
2. **Segment analysis by price tier** to identify neighborhoods where affordable units are scarce relative to demand
3. **Publish interactive dashboard** showing regional affordability metrics for use by advocates, policymakers, and program planners

---

## References

- Joint Center for Housing Studies of Harvard University. (2025). *The State of the Nation's Housing 2025.*
- National Low Income Housing Coalition. (2024). *Out of Reach: The High Cost of Housing.*
- U.S. Department of Housing and Urban Development. (2024). *Annual Homeless Assessment Report (AHAR) to Congress.*

---

*This research brief was prepared as a portfolio submission for the Chief Research Officer (Volunteer) role with Mentor A Promise (MAP). The companion technical report and full codebase are available in this repository.*
