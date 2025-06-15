# 📊 Task 1: Exploratory Data Analysis and Preprocessing

## 🔍 Objective

The purpose of this task was to:
- Load and understand the structure of the raw auto insurance dataset.
- Perform preprocessing and cleaning while preserving meaningful information.
- Generate useful exploratory data analysis (EDA) visualizations.
- Prepare a clean dataset to be used in the next stages of risk modeling.

---

## 🗂 Data Loading

- A custom delimiter detection method was implemented to correctly parse the raw CSV file.
- Data was loaded using `pandas.read_csv()` with chunking for efficiency (`chunksize=100000`).
- Successfully loaded over **1,000,000 rows** and **52 columns**.

---

## 🧼 Data Cleaning Strategy

The following steps were performed in `PreProcessData`:

### ✔️ Column Standardization
- Converted all column names to lowercase and replaced spaces with underscores.

### ✔️ Data Type Conversion
- Converted `transactionmonth` to datetime format.

### ✔️ Loss Ratio Calculation
- Added a new column: `loss_ratio = totalclaims / totalpremium`.

### ✔️ Missing Data Handling
- **Dropped rows** only if they had missing values in **at least 3** out of 4 critical columns:
  - `totalclaims`
  - `totalpremium`
  - `gender`
  - `transactionmonth`
- Filled missing values:
  - `customvalueestimate` with 0
  - `bank` with `"Unknown"`
  - `gender` with `"Not specified"`

### ✔️ Outlier and Invalid Value Filtering
- Filtered out rows where:
  - `totalpremium <= 0`
  - `totalclaims < 0`
  - `loss_ratio` was in the top 1% outliers

> ⚠️ Note: These filters were relaxed or adjusted when they caused excessive row loss.

---

## 📊 Exploratory Data Analysis (EDA)

Various univariate and bivariate visualizations were generated using the cleaned data:
- Distribution plots of `totalpremium`, `totalclaims`, and `loss_ratio`.
- Bar plots of loss ratio by `province`, `gender`, and `vehicletype`.
- Time-series trends of `loss_ratio` across months.
- Correlation matrix of numerical features.
- Box plots for claims and custom value estimates.
- Top vehicle makes by average claim amount.

> Only plots with valid and sufficient data were rendered.

---

## 📁 Output

- Cleaned dataset saved to: `../data/cleaned/machine_learning_rating_cleaned.csv`
- EDA visualizations displayed in notebook and console log.

---

## ✅ Summary

Task 1 successfully:
- Loaded and inspected the dataset.
- Applied thoughtful and assignment-aligned cleaning strategies.
- Conducted informative visual analysis using well-prepared data.

Ready to proceed to **Task 2: Statistical Testing and Hypothesis Formulation**.
