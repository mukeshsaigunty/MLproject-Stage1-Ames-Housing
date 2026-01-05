from load_and_prepare_data import load_ames_housing_data
from optuna_nested_cross_validation import run_nested_cv

X, y, num_cols, cat_cols = load_ames_housing_data()

score = run_nested_cv(X, y, num_cols, cat_cols)

print("✅ Stage-1 completed")
print("Mean RMSLE:", score)
