from imports_and_env_setup import *
from preprocessing_pipeline import build_preprocessing_pipeline
from model_factory import build_regression_model
from config_and_constants import RANDOM_STATE, ES_SPLIT

def refit_on_full_training(X, y, num_cols, cat_cols, params):

    iso = IsolationForest(contamination=0.01, random_state=RANDOM_STATE)
    mask = iso.fit_predict(
        X[num_cols].fillna(X[num_cols].median(numeric_only=True))
    )

    X = X.loc[mask == 1]
    y = y.loc[mask == 1]

    X_tr, X_es, y_tr, y_es = train_test_split(
        X, y, test_size=ES_SPLIT, random_state=RANDOM_STATE
    )

    pre = build_preprocessing_pipeline(num_cols, cat_cols, params)
    X_tr_t = pre.fit_transform(X_tr, y_tr)
    X_es_t = pre.transform(X_es)

    model = build_regression_model(params)
    model.fit(X_tr_t, np.log1p(y_tr))

    return {"pre": pre, "model": model}
