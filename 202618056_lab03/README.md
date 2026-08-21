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

Two preprocessing pipelines were created and evaluated with two classification algorithms:

- Logistic Regression
- Decision Tree Classifier

The same stratified train-test split was used for all experiments to ensure a fair comparison.

---

## 2. Dataset

The project uses the **Hotel Booking Demand Dataset**, which contains booking information for hotels and includes information such as:

- Lead time
- Arrival date
- Number of adults, children, and babies
- Meal type
- Country
- Market segment
- Previous cancellations
- Deposit type
- Room type
- Booking changes
- Customer type
- Average Daily Rate (`adr`)
- Reservation status
- Booking cancellation status

The target variable is:

```text
is_canceled

where:
0 = Booking was not canceled
1 = Booking was canceled

3. Data Loading and Understanding

The dataset was loaded using Pandas and examined using:

head()
shape
info()
describe()
dtypes

The class distribution of is_canceled was also checked to understand the distribution of canceled and non-canceled bookings.

The target variable was separated from the input features:
y = df["is_canceled"]

The remaining usable columns were used to create X.

Numerical and categorical features were identified using their data types.

4. Data Preprocessing and Cleaning
4.1 Target Selection and Leakage Prevention

The target variable was set to:

is_canceled

The following columns were removed because they directly reveal the final booking outcome:

reservation_status
reservation_status_date

Keeping these columns would result in target leakage, because the model could use information that would only be available after the booking outcome was determined.

The company column was also removed because it contained extremely high missingness and therefore provided limited reliable information.
4.2 Missing Value Analysis

Missing values were checked for every column using both counts and percentages.

A heatmap was used to visualize the missing-value pattern across the dataset.

The company column was identified as having very high missingness and was removed.

The remaining missing values were handled through the preprocessing pipelines rather than manually filling them before the train-test split.

4.3 Numerical Missing Values

Missing numerical values were handled using:
KNNImputer(n_neighbors=5)

The KNN imputer estimates missing values using the values of the five nearest observations.

This imputation was included inside the Scikit-learn pipeline so that the imputer was fitted only on the training data.
4.4 Categorical Missing Values

Missing categorical values were handled using:

SimpleImputer(strategy="most_frequent")

The most frequently occurring category in each column was used to replace missing values.

Categorical variables were then converted into numerical features using:

OneHotEncoder(handle_unknown="ignore")

The handle_unknown="ignore" option ensures that unseen categories in the test set do not cause an error during prediction.
5. Outlier Treatment

Selected numerical features were examined for potential outliers using boxplots and the IQR method.

The following features were checked:

lead_time
adr
adults
children
babies

Logically impossible bookings where:

adults = 0
children = 0
babies = 0

were removed.

Unrealistic adr values were also filtered.

A conservative:

3 × IQR

criterion was used to remove only clear and extreme outliers.

The number of rows before and after outlier removal was recorded.

Boxplots and distribution plots were then used to visually check the data after outlier removal.

Some observations may still appear outside the standard 1.5 × IQR range because only extreme outliers were intentionally removed.

6. Train-Test Split

The dataset was split into training and testing sets using:

train_test_split(
    test_size=0.2,
    stratify=y,
    random_state=42
)

The same train-test split was used for all four experiments.

This ensures that the models can be compared fairly under identical conditions.

7. Preprocessing Pipelines

Two different numerical preprocessing approaches were created.

Both pipelines use the same categorical preprocessing.

7.1 Pipeline A — StandardScaler

For numerical features:

KNNImputer(n_neighbors=5)
        ↓
StandardScaler

For categorical features:

SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")

Therefore, Pipeline A uses:

Numerical:
KNNImputer → StandardScaler


Categorical:
SimpleImputer → OneHotEncoder
7.2 Pipeline B — MinMaxScaler

For numerical features:

KNNImputer(n_neighbors=5)
        ↓
MinMaxScaler

For categorical features:

SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")

Therefore, Pipeline B uses:

Numerical:
KNNImputer → MinMaxScaler


Categorical:
SimpleImputer → OneHotEncoder
7.3 ColumnTransformer and Pipeline

ColumnTransformer was used to apply different preprocessing operations to numerical and categorical columns.

Pipeline was used to combine preprocessing and model training.

This ensures that preprocessing operations such as imputation, scaling, and encoding are fitted only on the training data, preventing data leakage from the test set.

8. Classification Models

Two classification algorithms were used.

8.1 Logistic Regression

The Logistic Regression model was configured as:

LogisticRegression(max_iter=1000)

The max_iter=1000 setting provides sufficient iterations for the model to converge.

8.2 Decision Tree Classifier

The Decision Tree model was configured as:

DecisionTreeClassifier(random_state=42)

The random_state=42 ensures reproducible results.

9. Four Model-Pipeline Combinations

Four experiments were performed:

Experiment	Model	Preprocessing
1	Logistic Regression	Pipeline A - StandardScaler
2	Logistic Regression	Pipeline B - MinMaxScaler
3	Decision Tree	Pipeline A - StandardScaler
4	Decision Tree	Pipeline B - MinMaxScaler

The model settings and train-test split were kept unchanged across all experiments.

10. Model Performance Evaluation

Each model was evaluated using:

Training Accuracy
Testing Accuracy
Precision
Recall
F1-Score
Train-Test Accuracy Difference

The train-test difference was calculated as:

Training Accuracy - Testing Accuracy

A large positive difference indicates possible overfitting.

11. Performance Comparison
Model & Pipeline	Train Accuracy	Test Accuracy	Overfitting Gap	Precision	Recall	F1-Score
Logistic Regression + Pipeline A	0.8186	0.8183	0.0003	0.8111	0.6642	0.7304
Logistic Regression + Pipeline B	0.8129	0.8132	-0.0003	0.8085	0.6495	0.7203
Decision Tree + Pipeline A	0.9963	0.8602	0.1361	0.8084	0.8159	0.8122
Decision Tree + Pipeline B	0.9963	0.8600	0.1363	0.8078	0.8164	0.8121
12. Final Observations
12.1 Best Overall Result

The Decision Tree + Pipeline A achieved the best overall performance among the four combinations.

It achieved:

Testing Accuracy: 86.02%
F1-Score: 0.8122

The Decision Tree performed better than both Logistic Regression models and was able to capture non-linear relationships between the booking features.
12.2 Effect of Scaling on Logistic Regression

StandardScaler performed slightly better than MinMaxScaler for Logistic Regression.

Metric	Pipeline A - StandardScaler	Pipeline B - MinMaxScaler
Testing Accuracy	81.83%	81.32%
F1-Score	0.7304	0.7203

Therefore, the choice of scaling method had a small but noticeable effect on Logistic Regression.

StandardScaler produced better testing accuracy, precision, recall, and F1-score than MinMaxScaler in this experiment.

12.3 Effect of Scaling on Decision Tree

Scaling had almost no effect on Decision Tree performance.

Metric	Pipeline A	Pipeline B
Testing Accuracy	86.02%	86.00%
Precision	0.8084	0.8078
Recall	0.8159	0.8164
F1-Score	0.8122	0.8121

The differences between the two pipelines are extremely small.

This is expected because Decision Trees are generally insensitive to feature scaling. They make decisions based on feature thresholds rather than distances between numerical values.

12.4 Overfitting Analysis

Both Decision Tree models show significant signs of overfitting.

The Decision Trees achieved:

Training Accuracy ≈ 99.63%
Testing Accuracy  ≈ 86.00%

The train-test gaps were:

Decision Tree + Pipeline A = 0.1361
Decision Tree + Pipeline B = 0.1363

This corresponds to approximately a 13.6 percentage-point gap.

The large difference indicates that the Decision Trees fit the training data very closely but do not generalize equally well to unseen data.

12.5 Logistic Regression Generalization

Logistic Regression showed much better generalization.

The train-test differences were:

Logistic Regression + Pipeline A = 0.0003
Logistic Regression + Pipeline B = -0.0003

These differences are extremely small.

Therefore, Logistic Regression shows very little evidence of overfitting and generalizes more consistently to the test set.

However, its testing accuracy and F1-score were lower than those of the Decision Tree.

13. Confusion Matrix Analysis

Confusion matrices were created for:

The best Logistic Regression result.
The best Decision Tree result.

The confusion matrices provide a detailed view of:

True Positives
True Negatives
False Positives
False Negatives

They help evaluate how effectively the models distinguish between canceled and non-canceled bookings.

The Decision Tree achieved stronger overall classification performance, while Logistic Regression showed more stable generalization between training and testing data.

14. Overfitting and Feature Reduction

The Decision Tree models showed a large train-test accuracy gap, indicating possible overfitting.

To investigate whether model complexity could be reduced, selected high-cardinality or less useful features were considered for removal.

The reduced-feature Decision Tree models can be compared with the original models using:

Training Accuracy
Testing Accuracy
F1-Score
Train-Test Difference

The goal of feature reduction is not simply to lower training accuracy. A successful reduction should ideally:

Reduce the train-test accuracy gap.
Maintain or improve testing accuracy.
Maintain or improve F1-score.
Improve generalization to unseen data.
15. Conclusion

The experiment compared two preprocessing pipelines and two classification algorithms for predicting hotel booking cancellations.

The Decision Tree + Pipeline A achieved the highest overall performance with a testing accuracy of 86.02% and an F1-score of 0.8122.

However, the Decision Tree models showed significant overfitting, with a train-test accuracy gap of approximately 13.6 percentage points.

Logistic Regression showed much better generalization, with almost no difference between training and testing accuracy, but its overall predictive performance was lower.

The comparison also demonstrated that:

StandardScaler performed slightly better than MinMaxScaler for Logistic Regression.
Scaling had almost no effect on Decision Tree performance.
Decision Trees achieved higher predictive performance but showed greater overfitting.
Logistic Regression generalized more consistently but had lower test performance.

Overall, the experiment demonstrates the importance of evaluating both preprocessing strategies and model types when developing a machine-learning classification system.
