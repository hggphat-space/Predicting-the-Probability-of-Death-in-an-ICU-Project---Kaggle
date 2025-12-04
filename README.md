## ICU Mortality Prediction Using Machine Learning

### Overview

This project predicts **in-hospital mortality of ICU patients** using electronic health record (EHR) data derived from the **MIMIC-III** database.  
It was originally developed as part of the **Monash University ETC3250/5250** unit and has been refactored and documented here to showcase an end‑to‑end **machine learning + clinical data** workflow.

I implemented and compared multiple models in **R**:

- **Logistic Regression**
- **Decision Tree** (`rpart`)
- **Neural Network with PCA** (`nnet` + `caret::preProcess(method = "pca")`)
- **XGBoost / Gradient Boosted Trees** (`xgboost`, `caret::train(method = "xgbTree")`)

The primary evaluation metric is **AUC-ROC**, chosen for its suitability in **imbalanced clinical outcomes** and its threshold‑independent nature.

---

### Data

#### MIMIC-III

The project uses features derived from the **MIMIC-III (Medical Information Mart for Intensive Care III)** database, a large, de‑identified ICU dataset.

- **Target variable**
  - `HOSPITAL_EXPIRE_FLAG`  
    - `1`: patient died during hospital stay  
    - `0`: patient survived to discharge

- **Key inputs**
  - **Demographics**
    - `AGE` (cleaned and recomputed from `DOB` and `ADMITTIME`, implausible ages removed)
    - Admission type (e.g., `EMERGENCY`)
    - ICU unit (`FIRST_CAREUNIT`)
  - **Vital signs (summary features)**
    - Heart rate: `HeartRate_Min`, `HeartRate_Mean`, `HeartRate_Max`
    - Blood pressure: `SysBP_*`, `DiasBP_*`, `MeanBP_*`
    - Respiratory rate: `RespRate_*`
    - Temperature: `TempC_*`
    - Oxygen saturation: `SpO2_*`
    - Glucose: `Glucose_*`
  - **Stay-level information**
    - Length of stay in days: `LOS_days` (derived from an hourly difference variable)
  - **Diagnosis information**
    - ICD-9 codes cleaned and grouped by the first 3 digits (e.g., `ICD9_410`, `ICD9_428`), then pivoted to wide **dummy variables**.

> **Note**: Raw MIMIC-III data is **not included** in this repository. To fully reproduce the pipeline, you must obtain access to MIMIC-III and recreate the derived CSVs (`mimic_train_X.csv`, `mimic_train_y.csv`, `mimic_test_X.csv`, `MIMIC_metadata_diagnose.csv`, `MIMIC_diagnoses.csv`).

---

### Feature Engineering

A major focus of this project is **clinically meaningful feature engineering** on top of the extracted MIMIC-III features.

- **Age processing and buckets**
  - Clean `DOB` and `ADMITTIME` with robust parsing and remove invalid ages.
  - Create age groups (`young`, `adult`, `elderly`) and encode them as dummy variables.

- **ICD-9 diagnosis features**
  - Clean ICD-9 codes and group by 3-digit prefix (ICD9 categories).
  - Remove very rare categories and pivot to one-hot dummy variables (`ICD9_XXX`) per ICU stay.
  - Merge ICD-9 features into train and test sets by `HADM_ID` or `SUBJECT_ID`.

- **Physiological outlier flags**
  - Define **clinically plausible ranges** for heart rate, blood pressure, respiratory rate, temperature, SpO₂, and glucose.
  - For each variable, create flags such as:
    - `HeartRate_Mean_too_low`, `HeartRate_Mean_too_high`
    - `SysBP_Min_too_low`, `SysBP_Max_too_high`
    - Similar flags for other vital signs.

- **Engineered clinical indices and interactions**
  - **Hemodynamic indices**
    - `Shock_Index = HeartRate_Mean / SysBP_Mean`
    - `MAP = (2 * DiasBP_Mean + SysBP_Mean) / 3`
    - `Pulse_Pressure = SysBP_Mean - DiasBP_Mean`
    - `HR_BP_Ratio = HeartRate_Mean / MeanBP_Mean`
  - **Instability and ranges**
    - `Glucose_Instability = Glucose_Max - Glucose_Min`
    - `HeartRate_Range`, `MeanBP_Range`, `SpO2_Range`, etc. (max − min)
  - **Risk and contextual features**
    - `O2_Desat_Risk` (e.g., SpO₂ minimum below a critical threshold)
    - `Is_Elderly` (e.g., `AGE >= 70`)
    - `Shock_Severity` (e.g., high Shock Index)
    - `High_risk_unit` (ICU units with the highest mortality in the training data)
    - `Is_emergency` (admission type is emergency)
  - **Composite risk score**
    - Constructed from binary flags (e.g., low blood pressure, hypoxemia, tachycardia, hyperglycemia, hypothermia) with different weights into a single **`Risk_score`**.
  - **Higher-order interactions**
    - `Shock_Risk_Combo = Shock_Index * Risk_score`
    - `Age_Shock_Index = AGE * Shock_Index`
    - `Age_Is_Emergency = AGE * Is_emergency`
    - `Elderly_Shock`, `Elderly_Hypoxia`
    - `MAP_Temp_Interaction`, `MAP_Pulse_Interaction`, `Temp_Glucose_Ratio`, etc.

- **Preprocessing for modeling**
  - Convert all selected features to numeric where appropriate.
  - **Median imputation** for missing values (train/test separately).
  - Remove **zero-variance** predictors.
  - **Standardize** numeric features (`center` + `scale`) using `caret::preProcess`.
  - Remove **highly correlated** features (`|r| > 0.8`) to reduce multicollinearity.
  - Build a final feature set combining:
    - Engineered numeric variables
    - Outlier flags
    - Age and LOS
    - ICD-9 dummies
    - Age-group dummies.

---

### Modeling Approach

All modeling is done in **R**, primarily using `caret`, `xgboost`, `nnet`, `rpart`, and `pROC`.

- **Logistic Regression**
  - `glm(family = "binomial")` on the full engineered feature set.
  - Baseline model and coefficient-based insights into mortality drivers.

- **Decision Tree (`rpart`)**
  - `rpart(HOSPITAL_EXPIRE_FLAG ~ ., method = "class")` with controlled complexity (`cp`).
  - Simple non-linear benchmark and interpretable tree structure.

- **XGBoost (Boosted Trees)**
  - Trained with `xgboost::xgb.train` and via `caret::train(method = "xgbTree")`.
  - Tuned parameters include `nrounds`, `max_depth`, `eta`, `gamma`, `subsample`, `colsample_bytree`, `min_child_weight`.
  - Evaluation through AUC-ROC and feature importance analysis (top 25 features).

- **Neural Network with PCA (`nnet` + `caret`)**
  - PCA (`caret::preProcess(method = "pca")`) on numeric features before training.
  - Neural network trained on PCA components with tuned hidden size and weight decay.
  - Used to explore a simple NN on a lower-dimensional representation and compare against tree-based methods.

---

### Results

> The exact AUC values depend on the final training runs; fill them in from your latest outputs.

- **Logistic Regression**: provides a strong linear baseline with reasonable AUC-ROC.
- **Decision Tree**: captures some non-linearities, with performance similar to or slightly better than Logistic Regression.
- **Neural Network (with PCA)**: competitive AUC but more sensitive to hyperparameters and preprocessing.
- **XGBoost**: **best-performing model**, achieving the highest AUC-ROC and strongest discrimination between survivors and non‑survivors.

Feature importance from XGBoost highlights:

- Demographic/context variables: `AGE`, `Is_Elderly`, `Is_emergency`, `High_risk_unit`, `LOS_days`.
- Engineered indices: `Shock_Index`, `MAP`, `Risk_score`, ranges of vital signs.
- ICD-9 diagnosis dummies associated with high-risk conditions.

These patterns align well with clinical expectations around ICU mortality risk.

---

### How to Run

1. **Install R packages**

   ```r
   install.packages(c(
     "nnet", "readr", "dplyr", "xgboost", "tibble", "lubridate",
     "caret", "pROC", "ggplot2", "gridExtra", "rpart", "tidyr"
   ))
   ```

2. **Prepare data**

- Place the derived CSVs (`mimic_train_X.csv`, `mimic_train_y.csv`, `mimic_test_X.csv`,
  `MIMIC_metadata_diagnose.csv`, `MIMIC_diagnoses.csv`) in the directory expected by the code  
  (the same path used in `Project_Hoang Gia Phat.qmd` via `setwd()`).

3. **Run the analysis**

- Open `Project_Hoang Gia Phat.qmd` in RStudio or another editor.
- Make sure `setwd()` at the top of the file points to the correct project directory.
- Run all chunks or render the document to:
  - Clean and engineer features
  - Train Logistic Regression, Decision Tree, Neural Network, and XGBoost models
  - Compute and plot ROC curves
  - Train a tuned XGBoost model with cross-validation
  - Generate final predictions on the test set (e.g., `submission_boost_cv_the_one.csv`).

---

### Model Evaluation

- **ROC curve (Receiver Operating Characteristic)**:
  - Plots **True Positive Rate (Sensitivity)** vs **False Positive Rate (1 − Specificity)** across thresholds.
- **AUC (Area Under the ROC Curve)**:
  - Measures how well the model ranks patients who die vs those who survive.
  - Ranges from 0.5 (random) to 1.0 (perfect discrimination).

**Why AUC-ROC here**:

- Threshold-independent, which is important when the clinical decision threshold can vary.
- More informative than accuracy for **imbalanced outcomes** like mortality.
- Commonly used and well-understood in clinical prediction model literature.

---

### What This Shows About My Skills

- **End-to-end ML pipeline on real clinical data**  
  From raw MIMIC-derived CSVs → data cleaning → feature engineering → modeling → evaluation and outputs.
- **Clinical feature engineering**  
  Design of indices like **Shock Index**, **MAP**, **Risk_score**, vital sign ranges, and interaction terms aligned with medical reasoning.
- **Handling messy, high-dimensional tabular data**  
  ICD-9 processing, one-hot encoding, missing data handling, standardization, and multicollinearity reduction.
- **Modeling and comparison**  
  Implementation and fair comparison of Logistic Regression, Decision Tree, PCA + Neural Network, and XGBoost using consistent metrics.
- **Explainability and interpretation**  
  Feature importance analysis, interpretable baselines, and clear documentation of findings.

This project is intended to serve as a **portfolio piece** that demonstrates my ability to build and evaluate machine learning models on complex healthcare data and to communicate the work clearly.

---

### Role of Generative AI

I used **ChatGPT** as a coding and ideation assistant to:

- Brainstorm clinically meaningful engineered features (e.g., Shock Index, composite risk scores, interaction terms).
- Draft and refine R code for data cleaning, ICD-9 processing, feature engineering, and model training with `caret` and `xgboost`.
- Help debug issues and clarify modeling options for tabular clinical data.
- Improve the structure and clarity of this README and other documentation.

All AI-suggested ideas and code were **reviewed, tested, and adapted by me**, and I am responsible for the final methodology and conclusions.

---

### References

- Johnson AEW, Pollard TJ, Shen L, et al. *MIMIC-III, a freely accessible critical care database*. Scientific Data. 2016.  
- Chen T, Guestrin C. *XGBoost: A Scalable Tree Boosting System*. KDD, 2016.  
- Le Gall JR, Lemeshow S, Saulnier F. *A new Simplified Acute Physiology Score (SAPS II)*.  
- Knaus WA, Draper EA, Wagner DP, Zimmerman JE. *APACHE II: a severity of disease classification system*.  
- Monash University, **ETC3250/5250** materials on predictive modelling and health analytics.




