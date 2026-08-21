# DS605 – Lab Assignment 3
## Scikit-learn: Data Preprocessing and Model Performance Evaluation

**Name:** Keval Anilbhai Patodiya
**ID:** 202618056
**Dataset:** [Kaggle Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (`hotel_bookings.csv`)

---

## Preprocessing Choices

- **Feature/column cleanup:**
  - `company` dropped — over 90% of its values were missing.
  - `is_canceled` (target), `reservation_status`, and `reservation_status_date` dropped from the feature set (the latter two directly leak the target).
- **Outlier inspection:** Boxplots and an IQR-based summary were computed for all numerical columns, and a conservative extreme-outlier filter (Q1 − 3·IQR to Q3 + 3·IQR on `lead_time`, `adr`, `adults`, `children`, `babies`) was tested during exploration. This filtered set (`X_clean`/`y_clean`) was **not** carried into the final modeling pipeline — the models were trained on the full cleaned dataset (post column-drop, pre-outlier-removal).
- **Missing values:**
  - Numerical columns → `KNNImputer(n_neighbors=5)`
  - Categorical columns → `SimpleImputer(strategy="most_frequent")`
- **Scaling (two pipelines compared):**
  - **Pipeline A** → `KNNImputer` + `StandardScaler` on numerical features
  - **Pipeline B** → `KNNImputer` + `MinMaxScaler` on numerical features
- **Categorical encoding:** `OneHotEncoder(handle_unknown="ignore")`
- **Train/test split:** 80/20, stratified on the target (`is_canceled`), `random_state=42`
- Preprocessing was assembled with `ColumnTransformer` + `Pipeline`, combined with the model in a single end-to-end `sklearn.Pipeline` for each run.

## Models

- `LogisticRegression(max_iter=1000)`
- `DecisionTreeClassifier(random_state=42)`

Each model was trained under both Pipeline A and Pipeline B (4 combinations total).

## Final Comparison Table

| Model + Pipeline | Train Accuracy | Test Accuracy | Precision | Recall | F1-score | Train-Test Gap |
|---|---|---|---|---|---|---|
| Decision Tree + Pipeline A (StandardScaler) | 0.9963 | 0.8602 | 0.8084 | 0.8159 | 0.8122 | 0.1361 |
| Decision Tree + Pipeline B (MinMaxScaler) | 0.9963 | 0.8600 | 0.8078 | 0.8164 | 0.8121 | 0.1363 |
| Logistic Regression + Pipeline A (StandardScaler) | 0.8186 | 0.8183 | 0.8111 | 0.6642 | 0.7304 | 0.0003 |
| Logistic Regression + Pipeline B (MinMaxScaler) | 0.8129 | 0.8132 | 0.8085 | 0.6495 | 0.7203 | -0.0003 |

(Full table also produced in-notebook as `results_df`.)

## Final Observations

1. **Best overall combination:** Decision Tree + Pipeline A (StandardScaler) gives the best test accuracy (0.8602) and F1-score (0.8122), narrowly ahead of Decision Tree + Pipeline B. Both Decision Tree combinations beat both Logistic Regression combinations on every metric except train-test gap.
2. **Scaler effect on Logistic Regression:** StandardScaler (0.8183 test accuracy, F1 0.7304) edges out MinMaxScaler (0.8132 test accuracy, F1 0.7203) by a small but consistent margin — in line with LR's gradient-based optimizer benefiting from zero-centred, unit-variance features.
3. **Scaler effect on Decision Tree:** Scaling barely matters (0.8602 vs 0.8600 test accuracy; 0.8122 vs 0.8121 F1) since trees split on per-feature thresholds and both scalers are monotonic transforms.
4. **Overfitting:** The Decision Tree shows a large train-test gap (~0.136), a classic sign of an unconstrained tree (no `max_depth`) memorising the training data. Logistic Regression shows almost no gap (~0.0003), indicating much better generalization but at a lower ceiling of predictive performance.
5. **Precision vs. recall:** The best Decision Tree has fairly balanced precision/recall (0.8084 / 0.8159) on the cancelled class, while Logistic Regression is precision-heavy and recall-weak (0.8111 / 0.6642) — it misses a substantial share of true cancellations. For a hotel acting on cancellation risk, the Tree's much higher recall is a practically meaningful advantage, despite its overfitting.

## Repository Contents

- `202618056_lab03.ipynb` — full runnable notebook (data exploration, missing-value/outlier analysis, both pipelines, both models, evaluation, confusion matrices)
- `hotel_bookings.csv` — raw dataset (not included; download from the Kaggle link below)
- `README.md` — this file

