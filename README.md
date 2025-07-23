## Mortality Prediction Using Patient Data 

This project implements a machine learning pipeline to predict in-hospital mortality using clinical, demographic, and physiological data from the SUPPORT2 dataset. This repository focuses on model development using **XGBoost and Random Forest classifiers** as well as **Neural Networks** in the future.

---

###  Dataset

The SUPPORT2 dataset (*Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatments*) includes medical data from 9,105 critically ill hospitalized patients. It is hosted by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/880/support2).

Key data characteristics:

* **Clinical lab values**: albumin, bilirubin, creatinine, BUN, white blood cell count, etc.
* **Physiological metrics**: heart rate, temperature, mean blood pressure
* **Demographic info**: age, sex, race, education
* **Functional scores**: ADLs, coma scale, survival estimates
* **Target variable**: `death`

This dataset presents challenges typical of medical data:

* Numerous missing values
* Mixed data types (numerical + categorical)
* Class imbalance (fewer deaths than survivals)

---

### Pipeline Overview

1. **Data Cleaning**

   * Apply domain-based default values for critical lab variables
   * Drop columns with excessive missingness (e.g., `glucose`, `adlp`)
   * KNN imputation for remaining numerical values
   * Remove rows with missing categorical data

2. **Feature Engineering**

   * Encode binary categoricals (e.g., `sex`, `ca`) with label encoding
   * One-hot encode multi-class categoricals (e.g., `race`, `dnr`)
   * Detect and remove redundant features via correlation and information gain

3. **Modeling (Preview)**

   * Use **XGBoost** and **Random Forest** as main classifiers
   * Address class imbalance with **SMOTE** or undersampling (optional)

---

### Setup and Requirements

To install the environment, make sure you have Python 3.8+ and run:

```bash
git clone https://github.com/JanSmigielski/mortality_prediction.git
cd mortality_prediction
pip install -r requirements.txt
```
