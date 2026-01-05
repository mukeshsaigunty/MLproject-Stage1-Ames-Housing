# MLproject Stage-1 – Ames Housing Price Prediction

This project is my **Stage-1 machine learning implementation** for predicting house prices using the **Ames Housing dataset**.

The main goal of this stage is to build a **clean, well-structured ML pipeline** with proper preprocessing, model training, and evaluation.  
This stage focuses more on **correct process and structure** rather than only accuracy.

---

## 📌 What This Project Does

- Loads the Ames Housing dataset from OpenML
- Separates numeric and categorical features
- Applies preprocessing:
  - Missing value handling
  - Feature scaling
  - Encoding categorical variables
- Trains regression models (XGBoost / LightGBM)
- Evaluates performance using **RMSLE**
- Uses **nested cross-validation with Optuna** for reliable evaluation

---

## 🗂️ Project Structure

Each file represents a clear step in the pipeline:

- `imports_and_env_setup.py`  
  Common imports and environment setup

- `config_and_constants.py`  
  Global constants and configuration values

- `load_and_prepare_data.py`  
  Loads dataset and prepares feature columns

- `metrics_and_helpers.py`  
  Evaluation metric (RMSLE) and helper functions

- `custom_sklearn_transformers.py`  
  Custom sklearn transformers used in preprocessing

- `preprocessing_pipeline.py`  
  Complete preprocessing pipeline

- `model_factory.py`  
  Model selection logic (XGBoost / LightGBM)

- `training_and_early_stopping.py`  
  Training logic with early stopping

- `optuna_nested_cross_validation.py`  
  Nested cross-validation using Optuna

- `final_training_and_artifacts.py`  
  Final training and model saving

- `main.py`  
  Entry point to run the full Stage-1 pipeline

---

## ⚙️ Setup Instructions

Create a virtual environment (optional but recommended):

```bash
python -m venv stage
stage\Scripts\activate
