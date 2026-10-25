# DS605 – Lab 8: Decision Trees and Support Vector Machines

**Name:** Keval Anilbhai Patodiya
**Student ID:** 202618056

---

## Overview

This lab implements a **Decision Tree** and a **linear soft-margin SVM from scratch** (NumPy/Pandas only), compares them with Scikit-learn, and explores kernel SVMs on a nonlinear dataset.

- **Dataset:** UCI Banknote Authentication (1,372 samples, 4 features: variance, skewness, curtosis, entropy; 2 classes — 762 genuine/class 0, 610 forged/class 1). No missing values; 24 duplicate rows were kept.
- **Split:** one stratified 80/20 split (`random_state=17`) → 1,097 train / 275 test, reused for every model.
- **Two features (variance, skewness)** are used for all 2D boundary plots and Parts A–B experiments; **all four features** are used for the final comparison in Part D.

## Repository contents

| File | Description |
|---|---|
| `202618056_Lab08.ipynb` | Runnable notebook with all code, plots and results (Parts A–D) |
| `README.md` | This file |

Run: open the notebook and *Run All*. It needs `numpy, pandas, matplotlib, scikit-learn` and internet access (the dataset is loaded from the UCI URL).

## What is implemented

**Part A – Decision tree (from scratch)**
- Gini impurity and entropy; best feature/threshold search over midpoints of sorted values (vectorised with cumulative class counts).
- Recursive tree growth with stopping rules: max depth, pure node, minimum leaf size.
- `predict`, a printed rule listing, a drawn tree diagram, and decision-boundary plots next to Scikit-learn's tree.

**Part B – Linear SVM (from scratch)**
- Features standardised with training-set mean/std only; labels mapped to −1/+1.
- Soft-margin objective `½‖w‖² + C·mean(hinge)` minimised with full-batch subgradient descent and a slowly decaying step size.
- Plots show the separating line, ±1 margins and margin-violating training points; compared with `SVC(kernel='linear')` using `C_sklearn = C_manual / n_train` to match the objectives.

**Part C – Kernel SVM:** `make_moons` (500 samples, noise 0.22) with linear, polynomial (degree 3) and RBF kernels.

**Part D – Final comparison:** manual/Sklearn tree, manual/Sklearn linear SVM and RBF SVM on all four features.

## Key observations

### Part A – Decision tree (2 features)
- The manual tree **matches Scikit-learn exactly**: 90.2% test accuracy at depth 4 with 100% prediction agreement. Across 12 criterion/depth settings, agreement is 100% in every case except one (entropy, depth 10: 99.6%, i.e. one sample, likely a tie-break difference between equally good splits).
- **Overfitting with depth:** training accuracy keeps climbing (Gini: 86.1% → 99.3% from depth 1 to 10) while test accuracy peaks around depth 4 (90.2%) and then dips (88.0% at depth 6). The differences between depths 3–10 are only a few test samples, though: 5-fold CV on the training set gives 91.9–92.9% for depths 3–10, so beyond depth ≈ 3 the tree gains little on unseen data and only depths 1–2 are clearly underfitting.
- **Gini vs entropy:** very similar. Gini was slightly better at shallow depths (e.g. depth 4: 90.2% vs 87.6%); entropy was slightly better at depth 10 (90.5% vs 89.5%). Differences are within a few samples on a 275-sample test set, so neither criterion is clearly superior.
- The boundaries are axis-aligned rectangles; the learned rules show that low variance is the strongest forged-note indicator, with skewness used to refine the middle region.

### Part B – Linear SVM (2 features)
- The manual SVM **matches Scikit-learn's linear SVM** (82.9% test accuracy for both) and its objective decreases smoothly.
- A linear boundary on variance and skewness is limited (~83% test vs ~90% for the depth-4 tree): the two classes overlap in this 2D projection and are not linearly separable, so **432 of 1,097 training points violate the margin**.
- **Regularisation C:** larger C gave slightly higher accuracy (test 81.5% at C=1 → 83.6% at C=80), but the final objective grows with C (0.68 → 23.0) because the hinge term gets more weight. Small C means a wider margin and more violations.
- **Learning rate and iterations:** results were almost identical for step ∈ {0.001, 0.01, 0.05} and epochs ∈ {250, 1000, 3000} (train 88.9%, test 82.9%, objective ≈ 6.766). The optimiser converges early, so more iterations only cost time; the objective is convex, so it reaches the same solution regardless.

### Part C – Kernel SVM (moons)
| Kernel | Test accuracy |
|---|---|
| Linear | 74.4% |
| Polynomial (deg 3) | 77.6% |
| RBF | **88.0%** |

The two moons are interleaved and cannot be separated by a straight line, so the linear kernel underfits. Polynomial and RBF kernels implicitly map the data to a higher-dimensional space where a linear separator exists, producing curved boundaries; RBF follows the local shape of the moons best.

### Part D – Final comparison (4 features, test set)

| Model | Accuracy | Precision | Recall | F1 | Train (s) | Predict (s) |
|---|---|---|---|---|---|---|
| Manual tree (depth 5) | 0.9564 | 0.9435 | 0.9590 | 0.9512 | 0.0077 | 0.0003 |
| Sklearn tree (depth 5) | 0.9564 | 0.9435 | 0.9590 | 0.9512 | 0.0044 | 0.0005 |
| Manual linear SVM | 0.9636 | 0.9375 | 0.9836 | 0.9600 | 0.2503 | 0.0000 |
| Sklearn linear SVM | 0.9636 | 0.9375 | 0.9836 | 0.9600 | 0.0110 | 0.0024 |
| RBF SVM | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.0066 | 0.0031 |

*(Timings vary from run to run.)*

- **All four features help a lot:** trees go from ~90% (2 features) to ~95.6%, and the linear SVM from ~83% to ~96.4%.
- **Manual = Scikit-learn** for both algorithms (identical metrics), which validates the from-scratch implementations.
- **Tree vs linear SVM:** similar accuracy; the SVM has higher recall (fewer forged notes missed), the tree slightly higher precision.
- **RBF SVM is best** (100% on this split), showing the full four-feature data has a mildly nonlinear structure. With only 275 test samples this should not be read as a guarantee — cross-validation would give a more reliable estimate.
- **Speed:** the manual SVM is the slowest to train (0.25 s, a Python loop of 3,000 full-batch iterations vs. an optimised solver). Prediction is negligible for all models.

### Interpretability, scaling, overfitting, complexity
- **Interpretability:** the tree is the most interpretable (readable if/else rules, drawn tree). The linear SVM is interpretable through its weights; the RBF SVM is essentially a black box.
- **Scaling:** SVMs need standardised features (fit on training data only, to avoid leakage); trees are scale-invariant and use raw values.
- **Overfitting:** trees overfit as depth grows (Part A); SVMs are controlled via C (and gamma for RBF).
- **Complexity:** tree complexity = depth/number of leaves; linear SVM has just one weight per feature plus a bias; the RBF SVM can model complex boundaries but stores many support vectors.

## Notes and limitations
- The main results come from a single 80/20 split, so differences of a few test samples (1 sample ≈ 0.36%) are not significant.
- Hyperparameters in Parts A–B were explored on the test split for illustration. As a check, the same comparisons were re-run outside the notebook:
  - **Repeated 5-fold CV (3 repeats, all 1,372 rows, 4 features):** manual tree 97.1%, Sklearn tree 97.2%, manual and Sklearn linear SVM 97.9% each, RBF SVM 100% in all 15 folds. The model ranking and the manual = Scikit-learn agreement are the same as on the single split.
  - **Hyperparameters chosen by CV on the training set only:** tree depth 10 (test 89.5%, vs 90.2% at depth 4) and C = 10 (test 83.3%, vs 82.9% at C = 20). Test accuracy changes by about 2 samples or fewer, so the conclusions do not change.
