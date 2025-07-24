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

   * Encode binary categoricals (e.g., `sex`) with label encoding
   * One-hot encode multi-class categoricals (e.g., `race`, `dnr`)
   * Detect and remove redundant features via correlation and information gain

3. **Model Training**

   * Training using **XGBoost** and **Random Forest**
   * Grid Search and Random Search for hyperparameter tuning
   * Optionally apply SMOTE to address class imbalance

4. **Evaluation**

   * ROC AUC, F1 score, precision-recall metrics on test set
   * Confusion matrix and classification report for interpretation

---

### Setup and Requirements

 1. Create a new environment (Python 3.10 recommended)

```bash
conda create -n credit-risk-xai python=3.10
```

 2. Activate the enviroment

```bash
conda activate credit-risk-xai
```

 3. Install the required packages from requirements.txt

```bash
pip install -r requirements.txt
```
