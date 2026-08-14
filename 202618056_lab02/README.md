Lab Assignment - 2
Vectorized Programming with NumPy and Data Wrangling with Pandas

Name: Keval Patodiya
ID: 202618056

Dataset (Part B): Kaggle Titanic dataset (train.csv)

## Project Overview

This project is a practical implementation of **NumPy, Pandas, Matplotlib, and Seaborn** concepts using Python.

The notebook is divided into two major parts:

* **Part A – Vectorized Programming with NumPy**
* **Part B – Data Analysis with Pandas using the Titanic Dataset**

The project demonstrates array manipulation, vectorized operations, linear algebra, probability distributions, data inspection, filtering, aggregation, missing-value handling, feature engineering, correlation analysis, and data visualization.

---

## Objectives

The main objectives of this project are to:

* Understand NumPy arrays and their properties.
* Perform vectorized numerical and matrix operations.
* Work with probability distributions and histograms.
* Load and inspect a real-world dataset using Pandas.
* Select and filter data using `loc`, `iloc`, and Boolean indexing.
* Perform grouping and aggregation using `groupby()`.
* Identify and handle missing values.
* Detect Fare outliers using the IQR method.
* Create new features from existing Titanic variables.
* Build pivot tables for survival analysis.
* Create meaningful visualizations and interpret relationships between variables.

---

## Technologies and Libraries Used

* **Python**
* **NumPy** – numerical computing and array operations
* **Pandas** – data manipulation and analysis
* **Matplotlib** – data visualization
* **Seaborn** – statistical visualization
* **Jupyter Notebook** – development and presentation environment

---

## Project Structure

### Part A – Vectorized Programming with NumPy

### Task 1 – Arrays, Statistics, and Indexing

This section covers:

* Generating random arrays with a fixed random seed.
* Calculating minimum, maximum, median, mean, and standard deviation.
* Creating arrays using `np.arange()`.
* Creating zero and one arrays using `np.zeros()` and `np.ones()`.
* Creating evenly spaced values using `np.linspace()`.
* Creating and inspecting 2D and 3D arrays.
* Array indexing and slicing.
* Reshaping arrays and flattening them back to one dimension.

A random seed of `42` is used to make the generated results reproducible.

### Task 2 – Vectorized Arithmetic and Linear Algebra

This section demonstrates:

* Matrix addition.
* Element-wise multiplication.
* Matrix multiplication using `@`.
* Matrix transpose.
* Determinant calculation.
* Matrix inverse.
* Verification of the inverse using `np.allclose()`.

For example, the notebook verifies that multiplying a matrix by its inverse produces the identity matrix.

### Task 3 – Normal Distribution and Histogram

A normal distribution containing **1,000 values** is generated with:

* Mean = **50**
* Standard deviation = **10**

The generated sample produced:

* Sample mean ≈ **49.7111**
* Sample standard deviation ≈ **9.8872**

The results are close to the selected population parameters, demonstrating expected sampling variation.

---

# Part B – Titanic Dataset Analysis

The second part uses the **Titanic dataset** loaded from `titanic.csv`.

## Task 4 – Load and Inspect Data

The dataset is loaded using Pandas:

```python
df = pd.read_csv('titanic.csv')
```

The notebook demonstrates:

* `head()`
* `tail()`
* `shape`
* `columns`
* `info()`
* `describe()`
* `loc`
* `iloc`

The key difference demonstrated is:

* **`loc`** – label-based selection; can use column names and conditions.
* **`iloc`** – position-based selection; uses integer row and column positions.

---

## Task 5 – Filtering and Querying

Boolean indexing is used to answer several questions about Titanic passengers.

Important results include:

| Analysis                                               |     Result |
| ------------------------------------------------------ | ---------: |
| Male passengers older than 50                          |     **47** |
| Female first-class passengers                          |     **94** |
| Survival rate of female first-class passengers         | **96.81%** |
| Age 20–40, above-median Fare, survived                 |    **104** |
| Alone, age < 30, did not survive                       |    **141** |
| Southampton, Pclass 2/3, above Southampton median Fare |    **193** |

## The overall median Fare used in the analysis is **14.4542**, while the Southampton median Fare is **13.0**.

## Task 6 – GroupBy and Aggregation

The notebook calculates survival rates and summary statistics using `groupby()`.

### Survival Rate by Sex

| Sex    | Survival Rate |
| ------ | ------------: |
| Female |    **74.20%** |
| Male   |    **18.89%** |

### Survival Rate by Passenger Class

| Class | Survival Rate |
| ----- | ------------: |
| 1st   |    **62.96%** |
| 2nd   |    **47.28%** |
| 3rd   |    **24.24%** |

Female passengers had a substantially higher survival rate than male passengers, while first-class passengers had the highest survival rate among the three passenger classes.

---

## Task 7 – Missing Values and Fare Outliers

Missing values were investigated for every column.

### Important Missing-Value Results

| Column        | Missing Values | Percentage |
| ------------- | -------------: | ---------: |
| Age           |            177 | **19.87%** |
| Cabin         |            687 | **77.10%** |
| Embarked      |              2 |  **0.22%** |
| Other columns |              0 |     **0%** |

The `Cabin` column contains the largest proportion of missing values. Missing `Age` values were handled using mean imputation, and other approaches such as median, mode, and random-value imputation were also explored.

Fare outliers were also investigated using:

* Q1
* Q3
* IQR
* Lower and upper 1.5 × IQR bounds
* Outlier count

---

## Task 8 – Feature Engineering and Pivot Table

Two new features were created:

### FamilySize

```python
FamilySize = SibSp + Parch + 1
```

### IsAlone

A passenger is classified as travelling alone when:

```text
FamilySize = 1
```

A pivot table was then created using:

* Rows → `Sex`
* Columns → `Pclass`
* Values → Mean `Survived`

### Highest Survival Group

**Female, 1st Class – 96.81%**

### Lowest Survival Group

**Male, 3rd Class – 13.54%**

## These results are directly reported by the notebook.

## Task 9 – Visualizations and Observations

The project includes several visualizations:

* Missing-value bar chart
* Missing-value heatmap
* Correlation heatmap
* Survival rate by Sex bar chart
* Age vs Fare scatter plot
* Pair plot of numerical Titanic variables

The notebook uses Matplotlib and Seaborn for visualization.

---

# Key Observations

1. **Sex and survival:** Female passengers had a much higher survival rate (**74.20%**) than male passengers (**18.89%**).

2. **Passenger class and survival:** First-class passengers had the highest survival rate (**62.96%**), while third-class passengers had the lowest (**24.24%**).

3. **Sex and class together:** Female first-class passengers had the highest survival rate (**96.81%**), whereas male third-class passengers had the lowest (**13.54%**).

4. **Fare and survival:** Fare and Survived have a positive correlation of approximately **+0.26**, suggesting that passengers paying higher fares tended to have somewhat higher survival rates.

5. **Class and fare:** Pclass and Fare have a negative correlation of approximately **−0.55**, reflecting the higher fares generally associated with first-class passengers.

6. **Age and survival:** Age has a weak negative correlation with Survived of approximately **−0.08**, indicating that age alone has only a weak linear relationship with survival.

7. **Missing data:** The Cabin column has **687 missing values (77.10%)**, while Age has **177 missing values (19.87%)**. Age missing values were handled through mean imputation.

---

## Correlation Highlights

The strongest relationships identified in the correlation analysis are:

* **Strongest positive:** `Fare` vs `Survived` ≈ **+0.26**
* **Strongest negative:** `Pclass` vs `Fare` ≈ **−0.55**
* **Another notable negative:** `Pclass` vs `Survived` ≈ **−0.34**

These correlations suggest that passenger class and fare were associated with survival, although correlation alone does not establish causation.

---

## Conclusion

This project provides hands-on practice with **NumPy and Pandas for data analysis**, progressing from basic array manipulation and vectorized mathematical operations to real-world Titanic dataset analysis.

The Titanic analysis demonstrates how data filtering, grouping, missing-value treatment, feature engineering, pivot tables, correlations, and visualizations can be combined to extract meaningful insights from a dataset. The analysis particularly highlights the strong differences in survival rates based on **sex and passenger class**.

## Limitations

* The analysis is based only on the available Titanic dataset.
* Correlation does not imply causation.
* Missing-value imputation can affect statistical results.
* The `Cabin` column contains a very high percentage of missing values.
* The analysis is descriptive and does not build a predictive machine-learning model.
