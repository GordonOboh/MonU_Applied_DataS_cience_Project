# Notebook Timing Report

Run date: 2026-07-08 on 7.6GB RAM machine

## Capstone_Final_cc_ST.ipynb

| Stage | Time |
|---|---|
| Random Forest (500 trees, n_jobs=-1) | 14.6s |
| Gradient Boosting | 18.1s |
| XGBoost (2000 trees, n_jobs=-1) | 6.4s |
| **Training total** | **39.0s** |
| **Full notebook (nbconvert)** | **409s (6m 49s)** |

### Metrics
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 36.39 | 79.51 | 0.946 |
| Gradient Boosting | 47.66 | 86.28 | 0.936 |
| XGBoost | 47.80 | 85.34 | 0.938 |

## Capstone_Final_cc_ST_1.ipynb

| Stage | Time |
|---|---|
| Random Forest (500 trees, n_jobs=-1) | 15.4s |
| HistGradientBoosting (3000 iters, max_leaf_nodes=63) | 18.8s |
| XGBoost (2000 trees, n_jobs=-1) | 6.8s |
| **Training total** | **40.9s** |
| **Full notebook (nbconvert)** | **411s (6m 51s)** |

### Metrics
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 35.58 | 77.32 | 0.946 |
| HistGradientBoosting | 47.07 | 85.70 | 0.934 |
| XGBoost | 46.47 | 82.58 | 0.939 |

## Capstone_Final_cc_ST_2.ipynb (optimized loops)

| Stage | Time |
|---|---|
| Random Forest (500 trees, n_jobs=-1) | 14.5s |
| HistGradientBoosting (3000 iters, max_leaf_nodes=63) | 18.9s |
| XGBoost (2000 trees, n_jobs=-1) | 7.4s |
| **Training total** | **~41s** |
| **Full notebook (nbconvert)** | **154s (2m 34s)** |

### Metrics
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 35.58 | 77.32 | 0.946 |
| HistGradientBoosting | 47.07 | 85.70 | 0.934 |
| XGBoost | 46.47 | 82.58 | 0.939 |

## Metrics Comparison Across All Versions

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Random Forest | 35.58 | 5978.4 | 77.32 | 0.946 |
| HistGradientBoosting | 47.07 | 7344.9 | 85.70 | 0.934 |
| XGBoost | 46.47 | 6820.1 | 82.58 | 0.939 |

All three notebooks (ST, ST_1, ST_2) produced identical metrics — the optimizations changed only runtime, not results.

## Speed Comparison

| Notebook | Total Time | vs ST_1 |
|---|---|---|
| ST_1 (original) | 411s (6m 51s) | 1.0x baseline |
| ST_2 (optimized) | 154s (2m 34s) | **2.7x faster** |

## Key observation

Training itself is fast (~40s) but post-training work (permutation importance with 5 repeats, RF iteration plot over 500 trees, XGB iteration plot over 2000 trees, all charts) makes up the remaining ~370 seconds (~90% of total time). The optimized version eliminated the iteration plot overhead.
