from fastapi import FastAPI, status, HTTPException, Depends

app = FastAPI()

@app.get('/', status_code=status.HTTP_200_OK)
def home():
    return {"Message": "Pysurvival - Churn Prediction Model"}