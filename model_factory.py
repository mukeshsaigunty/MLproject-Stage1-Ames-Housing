from imports_and_env_setup import *
from config_and_constants import RANDOM_STATE

def build_regression_model(params):
    if params["model_type"] == "XGB":
        return xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["xgb_max_depth"],
            subsample=params["xgb_subsample"],
            colsample_bytree=params["xgb_colsample"],
            reg_lambda=params["xgb_reg_lambda"],
            min_child_weight=params["xgb_min_child_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist"
        )

    return lgb.LGBMRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["lgb_num_leaves"],
        subsample=params["lgb_subsample"],
        colsample_bytree=params["lgb_colsample"],
        reg_lambda=params["lgb_reg_lambda"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )
