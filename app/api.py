from fastapi import FastAPI, status, HTTPException, Depends
from pathlib import Path
import os
from . import schemas
import pandas as pd

from churn import preprocess
from churn.predict import get_prediction

# Define project base directory
def get_project_root() -> Path:
    return Path(__file__).absolute().parent.parent

PROJECT_BASE = get_project_root()

app = FastAPI(title="Survival API")

artifact_path = Path.joinpath(PROJECT_BASE, "artifact/model.zip")
data_path = Path.joinpath(PROJECT_BASE, "data/customer_churn.csv")

@app.get('/', status_code=status.HTTP_200_OK)
def home():
    return {"Message": "Pysurvival - Churn Prediction Model"}


@app.post('/predict', status_code=status.HTTP_200_OK)
def predict(survival_payload: schemas.SurvivalPayload):
    payload_df = pd.json_normalize(survival_payload.dict())
    # data = pd.DataFrame(data)
    encoded_data = preprocess.one_hot_encoder(payload_df)
    print(encoded_data)
    prediction = get_prediction(encoded_data, artifact_path)
    print(prediction)
    return prediction.to_dict(orient='records')
