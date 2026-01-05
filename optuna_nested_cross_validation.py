from imports_and_env_setup import *
from metrics_and_helpers import rmsle
from training_and_early_stopping import refit_on_full_training
from config_and_constants import *

def suggest_params(trial):
    params = {
        "model_type": trial.suggest_categorical("model_type", ["XGB", "LGBM"]),
        "pca_var": trial.suggest_float("pca_var", 0.85, 0.98),
        "loo_smooth": trial.suggest_float("loo_smooth", 1.0, 5.0),
        "rare_min_freq": trial.suggest_float("rare_min_freq", 0.005, 0.03),
        "n_estimators": trial.suggest_int("n_estimators", 600, 2400, step=600),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "xgb_max_depth": 5,
        "xgb_subsample": 0.8,
        "xgb_colsample": 0.8,
        "xgb_reg_lambda": 1.0,
        "xgb_min_child_weight": 1.0,
        "lgb_num_leaves": 31,
        "lgb_subsample": 0.8,
        "lgb_colsample": 0.8,
        "lgb_reg_lambda": 1.0
    }
    return params


def run_nested_cv(X, y, num_cols, cat_cols):

    outer_cv = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda t: rmsle(
                y_te,
                np.expm1(
                    refit_on_full_training(
                        X_tr, y_tr, num_cols, cat_cols, suggest_params(t)
                    )["model"].predict(
                        refit_on_full_training(
                            X_tr, y_tr, num_cols, cat_cols, suggest_params(t)
                        )["pre"].transform(X_te)
                    )
                )
            ),
            n_trials=N_TRIALS_INNER
        )

        scores.append(study.best_value)

    return np.mean(scores)
