# DS605 Lab 5 — Machine Learning with Scikit-learn and From Scratch

**Dataset:** UCI Productivity Prediction of Garment Employees
**Targets:**
- Regression → `actual_productivity` (Linear Regression)
- Classification → `MeetsTarget = 1` if `actual_productivity >= targeted_productivity`, else `0` (Logistic Regression)

## Repo structure
```
├── 202618056_Lab05.ipynb   # Full notebook: Part A, Part B, Part C
├── garments_worker_productivity.csv
└── README.md
```

## Workflow
```
Raw Data → Preprocessing → Train-Test Split → Model Training → Prediction → Evaluation → Comparison → Optimization
```

A single fixed 80/20 split (`np.random.seed(42)`) is generated once and reused across every model in the notebook — Part A, Part B, and the classification section all index into the same `train_indices` / `test_indices`, so every comparison below is on identical rows.

## Preprocessing
- Missing `wip` values (506 of 1197 rows, ~42%) filled with 0.
- `day`, `department`, `quarter` label-encoded to integers.
- Outliers inspected via IQR boxplots (not removed — kept as legitimate production variance).
- Manual (Part B) features standardized with mean/std computed on the training split only, then applied to test.

## Part A vs. Part B — Regression Results

| Model | MAE | RMSE | R² | Train time (s) |
|---|---|---|---|---|
| Manual (raw, closed-form) | 0.1052 | 0.1442 | 0.3256 | 0.0010 |
| Manual (scaled) | 0.1052 | 0.1442 | 0.3256 | — |
| Manual (reduced features) | 0.1053 | 0.1438 | 0.3290 | — |
| Manual (ridge, λ=0.1) | 0.1055 | 0.1439 | 0.3284 | — |
| Scikit-learn (full features) | 0.1052 | 0.1442 | 0.3256 | 0.0386 |
| Scikit-learn (reduced, fair) | 0.1053 | 0.1438 | 0.3290 | 0.0043 |
| Scikit-learn Ridge (reduced, fair) | 0.1055 | 0.1439 | 0.3284 | 0.0020 |

**Key observation:** on matching feature sets, the manual closed-form solution and scikit-learn's `LinearRegression`/`Ridge` produce **identical MAE/RMSE/R²** — confirming the from-scratch normal-equation implementation is mathematically correct. Manual training is faster in wall-clock time here because it's a single matrix inversion on a small feature set, versus scikit-learn's general-purpose fit overhead.

## Part A vs. Part B — Classification Results

| Model | Accuracy | Precision | Recall | F1 | Train time (s) |
|---|---|---|---|---|---|
| Manual (baseline, lr=0.001, 5000 iters) | 0.7208 | 0.7342 | 0.9532 | 0.8295 | 0.2162 |
| Manual (optimized, lr=0.001, 3000 iters) | 0.7417 | 0.7853 | 0.8771 | 0.8287 | 0.1260 |
| Scikit-learn `LogisticRegression` | 0.7167 | 0.7373 | 0.9357 | 0.8247 | 0.0078 |

**Iteration tuning** (manual gradient descent, lr=0.001):

| Iterations | F1 | Time (s) |
|---|---|---|
| 500 | 0.7372 | 0.024 |
| 1000 | 0.7372 | 0.061 |
| 2000 | 0.7734 | 0.145 |
| 3000 | 0.8287 | 0.245 |
| 5000 | 0.8295 | 0.285 |

F1 improves sharply up to ~3000 iterations, then plateaus — 3000 iterations was chosen as the optimized configuration, trading a marginal F1 drop (0.8295 → 0.8287) for meaningfully less compute.

## Part C — Optimization Summary
- **Feature selection**: correlation with `actual_productivity` (computed on training data only) used to drop weakly-correlated columns before regression.
- **Regularization**: Ridge (λ=0.1, chosen from a sweep over `[0.001, 0.01, 0.1, 1, 10, 100]`) applied to the reduced feature set; λ=0.1 gave the best RMSE/R² tradeoff.
- **Classification convergence**: gradient descent capped at 3000 iterations (see table above) instead of 5000, cutting training time by ~40% for a negligible F1 cost.
- **Runtime gap**: scikit-learn's implementations are consistently faster in wall-clock time (highly optimized C/Cython backends), but the manual implementation matches or nearly matches scikit-learn on every predictive metric, which was the primary goal of Part C.

## How to run
```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook 202618056_Lab05.ipynb
```
Run all cells top to bottom (Kernel → Restart & Run All) — the notebook uses a single fixed random seed (42) so results are reproducible.

## Notes
- Scikit-learn preprocessing, model, metric, and split utilities are used only in the Part A (scikit-learn) sections; Part B (manual) uses only NumPy/Pandas, per assignment requirements.
- `actual_productivity` is excluded from the classification feature set to avoid target leakage into `MeetsTarget`.
