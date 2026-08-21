# Lab Assignment 3: Scikit-learn Data Preprocessing and Model Performance Evaluation

**Student Name:** Keval Anilbhai Patodiya  
**Student ID:** 202618056
**Dataset:** [Kaggle Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

---

## 1. Project Overview

The objective of this assignment is to build and compare Scikit-learn preprocessing pipelines and evaluate two classification models for predicting hotel booking cancellations.

The target variable is `is_canceled`:

- `0` = Not Canceled
- `1` = Canceled

Two preprocessing pipelines were compared with:

- Logistic Regression
- Decision Tree Classifier

The same stratified train-test split was used for all experiments to ensure a fair comparison.

---

## 2. Data Preprocessing and Cleaning

### Target Selection and Leakage Prevention

The target variable was set to `is_canceled`.

The following columns were removed:

- `reservation_status`
- `reservation_status_date`

These columns directly reveal the final booking outcome and could cause target leakage.

The `company` column was also removed because it contained an extremely high percentage of missing values.

### Missing Value Handling

#### Numerical Features

Missing numerical values were handled using:

```python
KNNImputer(n_neighbors=5)

