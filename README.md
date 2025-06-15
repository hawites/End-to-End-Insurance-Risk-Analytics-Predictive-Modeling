# B5W3: Insurance Risk Analytics - Week 3

## 📁 Project Overview
This project involves performing end-to-end data processing, exploration, and version control for South African auto insurance data. The overall goal is to build a strong foundation for risk analytics and machine learning modeling by ensuring high-quality, reproducible datasets.

---

## ✅ Task 1: Data Cleaning and Exploratory Data Analysis (EDA)

### Objectives:
- Load the raw dataset and explore its structure.
- Clean the data to remove or impute missing/invalid values based on logical thresholds.
- Derive useful metrics like `loss_ratio`.
- Visualize key features for pattern discovery and insights.

### Key Steps:
- Observed data types, missing values, and distributions.
- Cleaned column names and transformed date fields.
- Filtered rows only when at least **3 or more of the critical columns** (`totalclaims`, `totalpremium`, `gender`, `transactionmonth`) were missing.
- Calculated `loss_ratio = totalclaims / totalpremium` where valid.
- Removed outlier rows with unusually high `loss_ratio` (above 99th percentile).
- Saved cleaned data to `data/cleaned/machine_learning_rating_cleaned.csv`.

### Output:
- Cleaned dataset with reliable columns and reduced noise.
- Charts/visuals for numerical and categorical variables.
- Validated temporal patterns, correlation matrix, and boxplots.

---

## ✅ Task 2: Data Version Control (DVC)

### Objectives:
Ensure full reproducibility of data pipelines using DVC to track datasets, clean outputs, and enable auditing or rollback when needed.

### Key Steps:
1. **Initialize DVC:**
   ```bash
   dvc init
   ```

2. **Configure Remote Storage:**
   ```bash
   mkdir ../dvc_storage
   dvc remote add -d localstorage ../dvc_storage
   ```

3. **Track Cleaned Data:**
   ```bash
   dvc add data/cleaned/machine_learning_rating_cleaned.csv
   ```

4. **Commit and Push Changes:**
   ```bash
   git add data/cleaned/machine_learning_rating_cleaned.csv.dvc .gitignore
   git commit -m "Track cleaned dataset with DVC"
   dvc push
   ```

5. *(Optional)* If creating a new version (v2) of the cleaned data:
   ```bash
   cp data/cleaned/machine_learning_rating_cleaned.csv data/cleaned/machine_learning_rating_cleaned_v2.csv
   dvc add data/cleaned/machine_learning_rating_cleaned_v2.csv
   git add .
   git commit -m "Added v2 cleaned dataset"
   dvc push
   ```

---

## 🧠 Notes
- DVC helps ensure that data used in analysis or modeling is always auditable.
- This aligns with industry standards in insurance, especially for risk assessments and regulatory compliance.
- All work is done in branches (`task-1`, `task-2`) and merged via Pull Requests for traceability.