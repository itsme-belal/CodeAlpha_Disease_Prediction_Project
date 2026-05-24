# 🏥 Disease Prediction from Medical Data

A machine learning classification project that predicts the likelihood of three diseases — **Heart Disease**, **Diabetes**, and **Breast Cancer** — using structured patient data from the UCI Machine Learning Repository.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Key Findings](#key-findings)
- [Disclaimer](#disclaimer)

---

## Overview

This project applies supervised binary classification to three well-known medical datasets. The full pipeline covers exploratory data analysis, preprocessing, model training, hyperparameter configuration, evaluation with multiple metrics, and visualization — all inside a single Jupyter Notebook.

**Task:** Predict the presence or absence of a disease given patient features such as age, symptoms, and lab test results.  
**Type:** Binary Classification  
**Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, 5-Fold Cross-Validation

---

## Datasets

| Dataset | File | Samples | Features | Target |
|---|---|---|---|---|
| Heart Disease | `Heart_Disease_Prediction.csv` | 270 | 13 | Presence / Absence |
| Diabetes | `diabetes.csv` | 768 | 8 | 1 (Diabetic) / 0 (Not) |
| Breast Cancer | `breast_cancer.csv` | 683 | 9 | 4 (Malignant) / 2 (Benign) |

### Feature Details

**Heart Disease** — Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, Fasting Blood Sugar, EKG Results, Max Heart Rate, Exercise Angina, ST Depression, Slope of ST, Number of Vessels (Fluoro), Thallium

**Diabetes** — Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age

**Breast Cancer** — Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape, Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin, Normal Nucleoli, Mitoses

---

## Project Structure

```
datasets/
│
├── Heart_Disease_Prediction.csv     # Heart disease dataset
├── diabetes.csv                     # Diabetes dataset
├── breast_cancer.csv                # Breast cancer dataset
│
disease-prediction/
│
├── Disease_Prediction_ML.ipynb      # Main Jupyter Notebook (all code)
│
├── final_results.csv                # Generated: full metrics table
│
├── eda_class_distribution.png       # Generated: class balance plots
├── eda_correlations.png             # Generated: correlation heatmaps
├── eda_heart_distributions.png      # Generated: feature distributions
├── eda_diabetes_distributions.png   # Generated: feature distributions
├── cm_heart_disease.png             # Generated: confusion matrices
├── cm_diabetes.png
├── cm_breast_cancer.png
├── roc_heart_disease.png            # Generated: ROC curves
├── roc_diabetes.png
├── roc_breast_cancer.png
├── metrics_heart_disease.png        # Generated: metric comparisons
├── metrics_diabetes.png
├── metrics_breast_cancer.png
├── feature_importance.png           # Generated: RF feature importances
├── cv_stability.png                 # Generated: cross-validation AUC
└── cross_dataset_comparison.png     # Generated: cross-dataset summary
```

---

## Algorithms

| Algorithm | Key Configuration |
|---|---|
| **SVM** | RBF kernel, C=1.0, probability calibration enabled |
| **Logistic Regression** | max_iter=2000, C=1.0, L2 regularization |
| **Random Forest** | 200 estimators, parallel jobs |
| **XGBoost** | 200 estimators, learning_rate=0.1, logloss objective |

> SVM and Logistic Regression use `StandardScaler`-normalized features. Random Forest and XGBoost operate on raw features.

---

## Results

### Heart Disease (270 samples, 13 features)

| Algorithm | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| SVM | 78.69% | 80.49% | 86.84% | 83.54% | 81.35% |
| Logistic Regression | 80.33% | 80.95% | 89.47% | 85.00% | 83.98% |
| **Random Forest** ⭐ | **83.61%** | **85.00%** | **89.47%** | **87.18%** | **87.36%** |
| XGBoost | 81.97% | 84.62% | 86.84% | 85.71% | 86.61% |

### Diabetes (768 samples, 8 features)

| Algorithm | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| SVM | 79.87% | 77.97% | 71.88% | 74.80% | 85.54% |
| Logistic Regression | 75.97% | 72.13% | 68.75% | 70.40% | 82.17% |
| **Random Forest** ⭐ | **83.77%** | **81.97%** | **78.12%** | **80.00%** | **90.14%** |
| XGBoost | 79.87% | 76.19% | 75.00% | 75.59% | 87.15% |

### Breast Cancer (683 samples, 9 features)

| Algorithm | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| **SVM** ⭐ | **98.25%** | **98.61%** | **98.61%** | **98.61%** | **99.50%** |
| Logistic Regression | 98.25% | 98.61% | 98.61% | 98.61% | 99.54% |
| Random Forest | 95.61% | 95.89% | 97.22% | 96.55% | 99.31% |
| XGBoost | 95.61% | 94.67% | 98.61% | 96.60% | 99.07% |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

Or install all at once:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
jupyter>=1.0.0
```

---

## Usage

1. **Clone or download** this repository and place all three CSV files in the same folder as the notebook.

2. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Open** `Disease_Prediction_ML.ipynb` and run all cells from top to bottom (`Kernel → Restart & Run All`).

The notebook will automatically generate all plots and save `final_results.csv`.

---

## Visualizations

The notebook produces the following plots:

- **Class Distribution** — bar charts showing target balance for each dataset
- **Correlation Heatmaps** — pairwise feature correlations per dataset
- **Feature Distributions** — histograms split by diagnosis class
- **Confusion Matrices** — for all 4 algorithms × 3 datasets
- **ROC Curves** — overlaid curves with AUC scores per dataset
- **Metric Comparison** — grouped bar charts (Accuracy, Precision, Recall, F1, AUC)
- **CV Stability** — 5-fold cross-validation AUC with error bars
- **Feature Importance** — top-10 Random Forest feature importances per dataset
- **Cross-Dataset Comparison** — side-by-side algorithm performance across all 3 diseases

---

## Key Findings

- **Random Forest** is the best overall model, achieving the highest AUC on both Heart Disease (87.36%) and Diabetes (90.14%). Ensemble methods handle the mixed numeric features in tabular medical data particularly well.

- **Breast Cancer** features are highly separable — even linear models like Logistic Regression reach 98.25% accuracy and 99.54% AUC, suggesting the UCI features cleanly distinguish benign from malignant tissue.

- **SVM** is highly competitive on well-scaled datasets and achieves the highest AUC of any model on Breast Cancer (99.50%).

- **XGBoost** consistently ranks second and provides the best accuracy-speed tradeoff for larger datasets.

- **Cross-validation** confirms stability: Random Forest's 5-fold CV AUC is 90.14% ± ~1.3% on Diabetes, showing low variance across folds.

**Recommended model per use case:**

| Dataset | Best Model | Primary Metric |
|---|---|---|
| Heart Disease | Random Forest | AUC 87.36% |
| Diabetes | Random Forest | AUC 90.14% |
| Breast Cancer | SVM or Logistic Regression | AUC 99.50–99.54% |

---

## Disclaimer

> This project is for **educational and research purposes only**. The models trained here are not validated for clinical use and should not be used to make real medical decisions. Always consult qualified healthcare professionals for diagnosis and treatment.

---

## Author

**Task 4 — Disease Prediction from Medical Data**  
Machine Learning Classification Project  
Datasets sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
