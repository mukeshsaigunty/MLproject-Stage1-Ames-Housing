from imports_and_env_setup import *
from custom_sklearn_transformers import EnsureDataFrame, RareCategoryGrouper

def build_preprocessing_pipeline(num_cols, cat_cols, params):

    num_pipe = Pipeline([
        ("imp", KNNImputer()),
        ("poly", PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False)),
        ("scale", RobustScaler()),
        ("pca", PCA(n_components=params["pca_var"]))
    ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("to_df", EnsureDataFrame(cat_cols)),
        ("rare", RareCategoryGrouper(params["rare_min_freq"])),
        ("loo", ce.LeaveOneOutEncoder(
            sigma=params["loo_smooth"],
            handle_unknown="value",
            handle_missing="value"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
