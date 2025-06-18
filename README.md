# Insurance Risk Analytics 

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

## ✅ Task 3: Hypothesis Testing on Risk Drivers

### Objectives:
Statistically validate whether certain customer or regional features significantly influence insurance risk (claim frequency/severity) or profitability (margin), to inform a new risk-based segmentation strategy.

### Hypotheses Tested:
1. **H₀: No risk differences across provinces**  
2. **H₀: No risk differences between zip codes**  
3. **H₀: No significant margin (profit) difference between zip codes**  
4. **H₀: No significant risk difference between Women and Men**

### Key Metrics:
- **Claim Frequency**: Proportion of policies with at least one claim  
- **Claim Severity**: Average claim amount (given a claim occurred)  
- **Margin**: TotalPremium - TotalClaims

### Methodology:
- **Group Segmentation**: Data was split by feature (e.g., province, zip, gender)
- **Statistical Testing**:  
  - T-tests were used for comparing means (claim severity and margin)
  - Chi-squared test was considered for categorical comparisons  
- **Significance Threshold**: α = 0.05

### Key Findings:
- ✅ **Rejected H₀ for provinces**: Statistically significant risk differences were found across provinces (p < 0.001)  
  📌 *Recommendation*: Premiums should be adjusted regionally (e.g., Gauteng had 15% higher loss ratio than Western Cape).

- ❌ **Failed to reject H₀ for zip codes** (risk and margin): No strong evidence of differentiation in claims or profit by zip.

- ❌ **Failed to reject H₀ for gender**: Men and women had statistically similar risk profiles.

### 🧪 How to Run:
In a notebook:
```python
from src.analysis.risk_hypothesis_testing import HypothesisTester
import pandas as pd

df = pd.read_csv("data/cleaned/machine_learning_rating_cleaned.csv")
tester = HypothesisTester(df)
results = tester.run_all_tests()
print(results)
```

### Output:
- Modular Python class `HypothesisTester` created under `src/analysis/risk_hypothesis_testing.py`
- Results returned as dictionary and integrated into notebook for interpretation
- All findings documented with p-values and business context

---

## ✅ Task 4: Predictive Modeling (Risk-Based Pricing)

### Objective:
Predict claim severity (`totalclaims`) using machine learning models.

### Models Implemented:
- Linear Regression
- Random Forest
- XGBoost (best performer)

### Evaluation:
- RMSE and R² scores used
- SHAP used for model explainability

### Conclusion:
XGBoost offered the most accurate predictions, with key drivers identified through SHAP. The output is ready for integration into risk-adjusted pricing models.

---

## 🧠 Notes
- DVC helps ensure that data used in analysis or modeling is always auditable.
- This aligns with industry standards in insurance, especially for risk assessments and regulatory compliance.
- All work is done in branches (`task-1`, `task-2`, `task-3`) and merged via Pull Requests for traceability.