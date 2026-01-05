import os


RANDOM_STATE = 41

OUTER_SPLITS = 4
INNER_SPLITS = 3
N_TRIALS_INNER = 10

EARLY_STOPPING_ROUNDS = 50
ES_SPLIT = 0.15

BASE_PATH = "artifacts"
os.makedirs(BASE_PATH, exist_ok=True)

ARTIFACT_MODEL  = f"{BASE_PATH}/production_model.joblib"
ARTIFACT_REPORT = f"{BASE_PATH}/nestedcv_report.json"
ARTIFACT_CARD   = f"{BASE_PATH}/model_card.json"
