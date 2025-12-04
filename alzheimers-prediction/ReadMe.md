# Alzheimer's Disease Onset Prediction

## Overview
This project provides a machine learning framework to predict the **future onset of Alzheimer’s Disease** using patient demographic, lifestyle, medical history, and biomarker features. The model **excludes cognitive tests and symptom indicators** to prevent data leakage, ensuring that predictions are based only on predictive features available **before disease onset**.

The model is built using **XGBoost**, with preprocessing steps for missing values and feature scaling already integrated. The pipeline is saved, allowing anyone to input new patient data and obtain predictions immediately.

---

## Features Used
- **Demographics:** Age, Gender, Ethnicity, EducationLevel  
- **Lifestyle Factors:** BMI, Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality  
- **Medical History:** FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, Depression, HeadInjury, Hypertension  
- **Clinical Biomarkers:** SystolicBP, DiastolicBP, CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides  

> **Excluded (Leakage):** MMSE, FunctionalAssessment, MemoryComplaints, BehavioralProblems, ADL, Confusion, Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness

---

## Repository Structure: 

    alzheimers-prediction/
    |
    ├─ README.md
    ├─ requirements.txt
    │
    ├─ models/
    │ └─ alzheimers_pipeline.pkl # saved pipeline with imputer, scaler, and XGBoost model
    │
    ├─ scripts/
    │ └─ predict_new_patients.py # script to predict new patients
    │
    └─ data/
    └─ sample_new_patients.csv # example input file
---

## Installation

1. Clone the repository:
2. run requirements.txt
```bash
pip install -r requirements.txt
```

## Using the Model

1. **Prepare a CSV file** with new patient data, ensuring it contains all the non-leakage features in this exact order:

Age,Gender,Ethnicity,EducationLevel,BMI,Smoking,AlcoholConsumption,PhysicalActivity,DietQuality,SleepQuality,FamilyHistoryAlzheimers,CardiovascularDisease,Diabetes,Depression,HeadInjury,Hypertension,SystolicBP,DiastolicBP,CholesterolTotal,CholesterolLDL,CholesterolHDL,CholesterolTriglycerides

2. **Run the prediction script:**

```bash
python scripts/predict_new_patients.py data/sample_new_patients.csv
```
Output:
- The script prints predictions to the console.
- A CSV file is generated with predictions and probabilities:
```bash
sample_new_patients_predictions.csv
```
### Columns include:
- Prediction: 0 = No Alzheimer’s, 1 = Alzheimer’s
- Probability: Model’s confidence for Alzheimer’s prediction

# Model Details
- Algorithm: XGBoost Classifier

# Preprocessing:
- Missing values imputed with median
- StandardScaler applied to numeric features

# Evaluation:
- Accuracy: ~96%
- ROC-AUC: ~0.99

