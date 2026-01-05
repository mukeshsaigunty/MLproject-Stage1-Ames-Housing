from imports_and_env_setup import *
from config_and_constants import *

def save_artifacts(model, report, card):

    joblib.dump(model, ARTIFACT_MODEL)

    with open(ARTIFACT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    with open(ARTIFACT_CARD, "w") as f:
        json.dump(card, f, indent=2)
