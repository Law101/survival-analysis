from fastapi import FastAPI, status, HTTPException, Depends
from pathlib import Path
import pandas as pd

from churn import preprocess
from churn.predict import get_prediction


app = FastAPI()

artifact_path = Path('/mnt/c/Users/Lawrence/Downloads/Lawrence/survival-analysis/artifact/model.zip')
data_path = Path('/mnt/c/Users/Lawrence/Downloads/Lawrence/survival-analysis/data/customer_churn.csv')

@app.get('/', status_code=status.HTTP_200_OK)
def home():
    return {"Message": "Pysurvival - Churn Prediction Model"}


@app.post('/predict', status_code=status.HTTP_200_OK)
def predict(data_path: str):
    data = pd.read_csv(data_path)
    # data = pd.DataFrame(data)
    encoded_data = preprocess.one_hot_encoder(data)
    print(encoded_data)
    prediction = get_prediction(encoded_data, artifact_path)
    print(prediction)
    return prediction.to_dict()