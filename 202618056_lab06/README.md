# DS605 Lab 6 - Feature Extraction and Machine Learning with Image and Text Data

Converting raw images and word-count data into numerical features and classifying them with traditional (non-deep-learning) models.

- **Part A / C (images):** Asphalt Crack dataset (Mendeley Data), 400 images, crack vs non-crack.
- **Part B (text):** Email Spam Classification dataset (Kaggle), 5,172 emails, spam vs non-spam.

No CNNs, deep-learning models or pretrained embeddings are used. Image features come from NumPy/OpenCV only; scikit-learn is used for the traditional classifiers.

## Repository contents

| File | Description |
|---|---|
| `202618056_lab06.ipynb` | Full runnable notebook (Part A, Part C, Part B) |
| `image_features_baseline.csv` | Baseline feature table: 9 features + label (400 rows) |
| `image_features_improved.csv` | Improved feature table: 13 features + label (400 rows) |
| `comparison_results.csv` | Image results: 3 feature sets x 3 models, 5x5 repeated CV |
| `text_count_results.csv` | Spam results with default hyperparameters |
| `text_tuning_results.csv` | Spam results, default vs tuned |
| `text_vocab_limit_results.csv` | Spam results for vocabulary sizes 100 / 300 / 500 / 1000 / 3000 |
| `figures/` | Sample image, grayscale, blur, Canny and preprocessing visualizations |

## How to run

1. Install dependencies: `pip install numpy pandas opencv-python matplotlib scikit-learn scipy`
2. Place the data next to the notebook:
   - `448/Cracks/` and `448/NonCracks/` (asphalt images)
   - `emails.csv` (spam dataset)
3. Run the notebook top to bottom. All random seeds are fixed (`random_state=42`).

---

## Part A - Image feature extraction and classification

**Data.** 400 images (200 crack, 200 non-crack), all 448 x 448 x 3, `uint8`. Images were read with OpenCV, resized to a common 448 x 448 size (already uniform, so this was a no-op) and converted to grayscale.

**Baseline features (9 per image), derived with NumPy/OpenCV only:**
- Intensity statistics: mean brightness, contrast (standard deviation), dark-pixel ratio (< 50), bright-pixel ratio (> 200), min, max and median intensity.
- Canny edge features: Gaussian blur (5 x 5), then `cv2.Canny(100, 200)`; `edge_count` and `edge_density`.

**Classifiers.** Logistic Regression (with `StandardScaler`), Decision Tree (`max_depth=4`, `min_samples_leaf=5`) and Random Forest (200 trees, `max_depth=6`, `min_samples_leaf=3`). Unrestricted trees and forests reached 100% training accuracy (they memorize the 320 training images), so tree depth and leaf size were limited to shrink the train-test gap.

**Hold-out results on the baseline features (stratified 80/20 split, 80 test images):**

| Model | Train Acc | Test Acc | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.900 | 0.925 | 0.905 | 0.950 | 0.927 |
| Decision Tree | 0.956 | 0.925 | 0.947 | 0.900 | 0.923 |
| Random Forest | 0.978 | 0.963 | 0.951 | 0.975 | 0.963 |

Confusion matrices, training time and prediction time are printed in the notebook. One test image is worth 1.25 percentage points, so single-split differences are not conclusive; the comparison in Part C therefore uses repeated cross-validation.

---

## Part C - Improving the representation (images)

**Change made.** Added crack-specific preprocessing and features. Raw brightness and Canny statistics respond to asphalt grain, shadows and uneven lighting as much as to cracks. The new pipeline suppresses those and keeps long, thin dark structures:

1. Remove uneven lighting (subtract a Gaussian-blurred background, sigma = 30)
2. Median filter (5) and Gaussian blur (sigma = 5) to suppress asphalt grain
3. Darken low-intensity regions (< 120, factor 0.4), then invert
4. Threshold at 180 and close small gaps (5 x 5 elliptical kernel)
5. Keep only connected components at least 35 px long

Four features are extracted from the resulting crack mask: `crack_pixel_count`, `crack_pixel_ratio`, `num_components`, `largest_component`.

Class means confirm the mask separates the classes:

| Feature | Non-crack | Crack |
|---|---|---|
| crack_pixel_ratio | 0.033 | 0.123 |
| num_components | 6.6 | 17.9 |
| largest_component (px) | 1,698 | 6,874 |

**Three feature sets were compared** (same models, same 5 x 5 repeated stratified cross-validation, 25 folds):

- **Baseline:** 9 features (intensity + Canny)
- **Improved:** 13 features (baseline + 4 mask features)
- **Improved compact:** 9 features (improved minus `min_intensity`, `max_intensity`, `crack_pixel_count`, `edge_count`). `max_intensity` is almost constant (250-255), `crack_pixel_count` and `edge_count` duplicate their ratio/density columns, and `min_intensity` was dropped to shrink the set.

| Feature set | Features | Extraction (ms/image) | Model | CV Acc | CV Std | CV F1 |
|---|---|---|---|---|---|---|
| Baseline | 9 | 10.79 | Logistic Regression | 0.900 | 0.029 | 0.901 |
| | | | Decision Tree | 0.930 | 0.026 | 0.930 |
| | | | Random Forest | 0.943 | 0.025 | 0.943 |
| Improved | 13 | 27.98 | Logistic Regression | 0.925 | 0.024 | 0.926 |
| | | | Decision Tree | 0.896 | 0.038 | 0.895 |
| | | | Random Forest | 0.945 | 0.023 | 0.945 |
| Improved compact | 9 | 27.98 | Logistic Regression | 0.932 | 0.021 | 0.933 |
| | | | Decision Tree | 0.895 | 0.039 | 0.894 |
| | | | Random Forest | 0.946 | 0.026 | 0.946 |

Fit and prediction times are in `comparison_results.csv`; all are well under one second per fold.

### Trade-off discussion

- **Dimensionality vs performance.** Going from 9 to 13 features did not help more than dropping back to 9: the compact set matched or beat the full improved set for every model, so the extra columns were redundant.
- **Computation.** The mask pipeline raises feature-extraction time from 10.8 to 28.0 ms per image (about 2.6x). Model training and prediction times are negligible by comparison, so extraction is the real cost.
- **Performance.** Logistic Regression benefits most (90.0% to 92.5-93.2%), because the mask features are close to linearly separable. Random Forest is essentially unchanged (94.3% to 94.5-94.6%), well inside the fold-to-fold standard deviation of 2-4 points. The Decision Tree did not improve (93.0% to about 89.5%), although its standard deviation (about 3.8 points) means this is better read as "no gain" than as a real drop.
- **Conclusion.** For tree ensembles the simple intensity + Canny features are nearly as good at about 40% of the extraction cost. The mask features are worth their cost mainly when a simple linear model is required. Most differences between feature sets are smaller than the cross-validation noise and should not be over-interpreted.

---

## Part B - Text vectorization and spam classification

**Dataset note.** `emails.csv` is a pre-vectorized dataset: 5,172 rows, one `Email No.` column, 3,000 word-count columns (the 3,000 most frequent words) and a `Prediction` label (0 = non-spam, 1 = spam). It contains no raw email text, so text cleaning steps such as tokenizing and lowercasing were already done, and `CountVectorizer` cannot be applied. The count matrix is used directly as the count representation.

**Cleaning and inspection.**
- Class distribution: 3,672 non-spam and 1,500 spam (29.0% spam).
- Dropped `Email No.` (identifier, not a feature).
- Found and removed 541 duplicate rows (identical word counts and label) before splitting, so identical emails cannot appear in both train and test. This leaves 4,631 emails (31.5% spam).
- Stratified 80/20 split: 3,704 training and 927 test emails (31.6% / 31.5% spam). The matrix is 5.9% dense, so it is stored as a sparse matrix.

**Models.** Multinomial Naive Bayes and Logistic Regression on the 3,000-feature count matrix. Spam is the positive class.

| Model | Train Acc | Test Acc | Precision | Recall | F1 | Train time (s) |
|---|---|---|---|---|---|---|
| Multinomial NB | 0.947 | 0.951 | 0.895 | 0.959 | 0.926 | 0.02 |
| Logistic Regression | 0.999 | 0.974 | 0.956 | 0.962 | 0.959 | 3.84 |

Confusion matrices (rows = true non-spam / spam): NB `[[602, 33], [12, 280]]`, Logistic Regression `[[622, 13], [11, 281]]`.

**TF-IDF was not used.** The assignment asks for a CountVectorizer vs TF-IDF comparison; this submission uses count vectors only, so there is no representation comparison of the two. The number of features (3,000) is fixed by the dataset.

### Hyperparameter tuning

5-fold stratified `GridSearchCV` on the training split only, scored with F1; the test set was used once at the end.

| Model | Grid | Best | CV F1 | Test Acc | Test F1 |
|---|---|---|---|---|---|
| Multinomial NB | alpha in {0.01, 0.03, 0.1, 0.3, 1, 3} | 0.01 | 0.926 | 0.957 (was 0.951) | 0.933 (was 0.926) |
| Logistic Regression | C in {0.01, 0.1, 1, 10, 100} | 1 | 0.944 | 0.974 (unchanged) | 0.959 (unchanged) |

Tuning gave a small gain for Naive Bayes (less smoothing helps) and none for Logistic Regression, because the default `C=1` was already the best value. The Naive Bayes optimum sits at the smallest alpha tested, so an even smaller value might do slightly better. For Logistic Regression, training F1 is 0.9996 vs 0.944 in cross-validation at `C=1`, so it fits the training data almost perfectly while `C=0.1` scores the same in cross-validation (0.944) with less overfitting (train F1 0.993). The training-time difference between default and tuned Logistic Regression (3.8 s vs 1.6 s) is timing noise; the fitted model is the same.

### Limiting the vocabulary (dimensionality vs performance)

Keeping only the top-k words (k = 100 to 3,000; **TODO: state how the k words were chosen, e.g. highest total frequency**), Test Acc / F1:

| Vocabulary | Multinomial NB | Logistic Regression |
|---|---|---|
| 100 | 0.836 / 0.757 | 0.905 / 0.847 |
| 300 | 0.899 / 0.848 | 0.951 / 0.923 |
| 500 | 0.919 / 0.877 | 0.969 / 0.951 |
| 1,000 | 0.936 / 0.903 | 0.973 / 0.957 |
| 3,000 | 0.951 / 0.926 | 0.975 / 0.961 |

Cutting the vocabulary from 3,000 to 1,000 words (67% fewer features) costs Logistic Regression only about 0.2 points of accuracy and 0.3 points of F1. At 500 words it loses about 0.6 points of accuracy and 1 point of F1, and below 300 words performance falls off quickly. Naive Bayes is more sensitive to vocabulary size and keeps improving up to the full vocabulary. Naive Bayes trains in milliseconds at any size; Logistic Regression takes a few seconds at all sizes, and its time does not fall reliably with fewer features (lbfgs iteration counts vary), so the main saving from a smaller vocabulary is memory and model size rather than training time.

---

## Overall observations

1. **Preprocessing decisions matter as much as model choice.** Removing 541 duplicate emails prevents train/test leakage in the spam data, and limiting tree depth removes the memorization seen with default trees on images.
2. **More engineered features are not automatically better.** The image mask features cost 2.6x more to extract and only clearly helped the linear model; the text vocabulary can be cut by two thirds with almost no loss.
3. **Model families respond differently to representation.** Linear models gain from well-designed features (image mask features); tree ensembles and Logistic Regression on text are strong with simple or redundant representations.
4. **Tuning gains were small.** Defaults were close to optimal for both text models.

## Limitations

- The image preprocessing thresholds (blur sigma, shadow threshold, binary threshold, minimum component length) were chosen by visual inspection of these same images, which may slightly favor the improved features.
- 400 images give noisy estimates: the fold-to-fold standard deviation (2-4 points) is comparable to most gaps between feature sets.
- Text results come from one stratified split (plus cross-validation inside tuning); the differences between Logistic Regression and Naive Bayes are large enough to be reliable, but small differences between tuned and default settings are not.
- The spam dataset contains word counts only, so raw-text cleaning, stop-word removal from text and n-grams were not applicable, and TF-IDF was not used.
