

from dill import load
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os

with open('rfr_model_best.pkl', 'rb') as f:
    rfr_model = load(f)

with open('abr_model_best.pkl', 'rb') as f:
    abr_model = load(f)

with open('gbr_model_best.pkl', 'rb') as f:
    gbr_model = load(f)

rfr_model = rfr_model[0]
abr_model = abr_model[0]
gbr_model = gbr_model[0]

app = FastAPI()

class Payload(BaseModel):
    Platform: str
    Year_of_Release: float
    Genre: str
    Publisher: str
    NA_Sales: float
    EU_Sales: float
    JP_Sales: float
    Other_Sales: float
    User_Score: float
    User_Count: float
    Rating: str

@app.post("/inference")
def predict(payload: Payload):
    df = pd.DataFrame([payload.model_dump().values()], columns=payload.model_dump().keys())
    y_pred_rfr = rfr_model.predict(df)
    y_pred_abr = abr_model.predict(df)
    y_pred_gbr = gbr_model.predict(df)
    response = {
        'prediction_rfr': y_pred_rfr[0],
        'prediction_abr': y_pred_abr[0],
        'prediction_gbr': y_pred_gbr[0],
        'model_name_rfr': 'Random Forest Regressor',
        'model_name_abr': 'AdaBoost Regressor',
        'model_name_gbr': 'Gradient Boosting Regressor'
    }
    return response
