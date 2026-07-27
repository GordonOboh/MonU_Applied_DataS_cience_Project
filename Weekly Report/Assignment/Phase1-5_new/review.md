# Phase 1–6 TeX Line-by-Line Review

## Priority Summary

| Level | Count | Description |
|-------|-------|-------------|
| **Critical** | 4 | Contradictory numbers or factual errors |
| **High** | 5 | CRISP-DM scope violations or significant duplication |
| **Medium** | 6 | Inconsistent numbers or justifications |
| **Low** | 5 | Minor clarifications or improvements |

---

## Critical Issues

### 1. sqft < 70 threshold count mismatch (P3 L99 vs P2 L323)
Phase 3 claims **48 records** removed for `sqfeet < 70` (0.01% of data), but Phase 2 only reports **48 records** for `sqfeet < 1`. If the threshold was raised from `< 1` to `< 70`, the record count must increase (records with sqft between 1–69 would also be removed).
> **Fix:** Determine actual count of records with `sqfeet < 70` and correct the number.

### 2. `max_leaf_nodes` default is wrong (P4 L106)
States `max_leaf_nodes=63` was "Increased from the default of 31." The scikit-learn `HistGradientBoostingRegressor` default for `max_leaf_nodes` is **`None`** (unlimited), not 31. The "31" confusion likely comes from LightGBM's `num_leaves` default.
> **Fix:** Change to "Increased from unlimited (None) to 63 to control model complexity."

### 3. Early stopping default is wrong (P4 L110)
States `early_stopping="auto"` stops when validation loss stops improving for **50 consecutive iterations**. The scikit-learn default for `n_iter_no_change` is **10**, not 50.
> **Fix:** Either add `n_iter_no_change=50` to the parameter table, or change "50" to "10."

### 4. MAE validation criterion invalid (P6 L66)
Task 3 says to "confirm MAE matches the research-phase value of $35.58." But Task 2 proposes retraining on **all 142,103** rows. A model trained on more data will produce different MAE — it cannot "match" the research value.
> **Fix:** Change validation criterion to: "Confirm MAE on held-out test set falls within expected range and does not degrade significantly."

---

## High-Priority Issues

### 5. P4 and P5 duplicate entire metrics table (P4 L169–171, P5 L18–20)
The metrics table, feature importance table, RF interpretation paragraph, and Observed-vs-Predicted figure are all duplicated verbatim between Phase 4 and Phase 5. CRISP-DM expects Phase 4 (technical model assessment) and Phase 5 (business-oriented evaluation) to be distinct.
> **Fix:** Keep table in P4 as technical results. In P5, either (a) drop the duplicate and reference P4, or (b) present with business-context commentary.

### 6. P4/P5 duplicate feature importance table (P4 L265–288, P5 L42–62)
Same table appears identically in both phases.
> **Fix:** Remove from one phase; cross-reference the other.

### 7. P6 missing CRISP-DM subsections (P6 L160 end)
The CRISP-DM Deployment phase has **4 tasks**: Plan Deployment, Plan Monitoring, **Produce Final Report**, **Review Project**. Tasks 3 and 4 appear in the Phase 1 timeline (L130–133) but have **no corresponding subsections** in Phase 6.
> **Fix:** Add `\subsection{Produce Final Report}` and `\subsection{Review Project}`.

### 8. P4/P5 interpretation paragraphs duplicated (P4 L178–182, P5 L34–37)
The paragraph interpreting Random Forest results (highest R², lowest error) is nearly identical in both phases.
> **Fix:** Differentiate — P4 discusses model selection rationale (why RF > XGBoost technically), P5 discusses business meaning (amenity ROI, actionable insights).

### 9. P1 constraints duplicated verbatim (P1 L17–19 vs L30–32)
The **Constraints** under "Determine Business Objectives" and the "Assess the Situation" items are identical word-for-word. "Assess the Situation" should provide inventory/assessment, not duplicate the objectives section.
> **Fix:** Remove the duplicate list from "Assess the Situation" and replace with genuine situation assessment (available resources, assumptions, data dictionary status).

---

## Medium-Priority Issues

### 10. Inconsistent max price (P2 L33 vs L48)
Line 33 says `over $2.7B`, line 48 says `max $2.77B`.
> **Fix:** Standardize on `$2.77B`.

### 11. Inconsistent max sqft (P2 L48 vs L275)
Line 48 says `8.4M sqft`, line 275 says `8.39M sqft`.
> **Fix:** Use `8.39M` consistently.

### 12. Label naming inconsistency: "Not Specified" vs "Missing" (P2 L30–33 vs P3 L174)
Phase 2 proposes creating a `"Not Specified"` category for missing laundry/parking. Phase 3 implements `"Missing"`. Functionally equivalent but inconsistent.
> **Fix:** Align to one term across both phases.

### 13. P3→P6 row count gap not bridged (P3 L137 vs P6 L53)
Phase 3 says `~144,000` after basic filters; Phase 6 says `142,103` total. The ~1,897-record gap is from IQR filtering but not explicitly connected.
> **Fix:** Add sentence in P3 after IQR filtering: "After region-level IQR filtering, 142,103 records remain."

### 14. "56× faster" claim unsourced (P4 L11)
"approximately 56 times faster on a sample dataset" — no source, no sample described.
> **Fix:** Cite a source, present actual benchmarks, or soften the claim.

### 15. P2 data-quality anomalies: negative latitude not flagged (P2 L80)
Min lat is `-43.53°` — negative latitude means southern hemisphere, impossible for US. Should be flagged in Task 2.4.
> **Fix:** Add `lat < 0` to the anomalies checklist.

---

## Low-Priority Issues

### 16. P5 next steps: GPU contradicts P1 constraint (P5 L142 vs P1 L18)
P1 says "No GPU or TPU hardware available." P5 recommends "Use GPU/CUDA to speed up training."
> **Fix:** Note hardware upgrade required, or remove GPU as a near-term next step.

### 17. P5 next steps: NLP contradicts excluded feature (P5 L143 vs P3 L88)
Recommends NLP on `description` field that was explicitly excluded in Phase 3.
> **Fix:** Acknowledge the hardware/scope trade-off or move NLP to long-term suggestion.

### 18. P1 Ubuntu version name wrong (P1 L90)
`Ubuntu 24 Server` — correct name is `Ubuntu 24.04 LTS (Server)`.
> **Fix:** Change to `Ubuntu 24.04 LTS (Server)`.

### 19. P4 K-Means definition belongs in Phase 3 (P4 L37–38)
"K-Means is an unsupervised algorithm that groups data points into clusters" — this definition belongs in Phase 3 (Construct Data), not Phase 4 (Modeling).
> **Fix:** Move K-Means definition to Phase 3. Phase 4 should only discuss parameter choices.

### 20. P5 MAE range notation confusing (P5 L31)
"MAE ($35.58--$47.07)" — the en-dash suggest a range, but MAE is a single value per model.
> **Fix:** Write: "MAE ranges from $35.58 (RF) to $47.07 (HistGB)."

---

## Cross-Phase Verification Summary

| # | Items | Status |
|---|-------|--------|
| 62 | P2→P3 sqft threshold count | **Critically inconsistent** |
| 63 | P2→P3 label naming | Inconsistent ("Not Specified" vs "Missing") |
| 64 | P3→P6 row counts | Gap not bridged |
| 65 | P4→P5 duplicate content | 4+ items duplicated verbatim |
| 66 | P1→P5 GPU constraint conflict | Contradiction unacknowledged |
| 67 | P4/P5→CSV RMSE precision | `85.70` vs `85.7` — formatting difference |
| 68 | P4→scikit-learn defaults | Two parameter defaults stated incorrectly |
| 69 | P6→P1 timeline gaps | Tasks 6.3/6.4 missing from P6 content |
| 70 | Total features (16) across P3/P4/P6 | ✓ Consistent |
| 71 | Metrics ($35.58, 0.946, etc.) P4/P5/CSV | ✓ Consistent (numbers match CSV) |
| 72 | Training row count (113,682) P4/P6 | ✓ Consistent |
| 73 | Test row count (28,421) P4 | ✓ Consistent |
| 74 | 80/20 split P3/P4 | ✓ Consistent |
| 75 | Southern states definition P2/P3 | ✓ Consistent (17 states, same list) |
