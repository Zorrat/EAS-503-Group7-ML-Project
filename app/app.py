

from dill import load
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os

print("-->",os.getcwd())
print("-->",os.listdir())

with open('rfr_model.pkl', 'rb') as f:
    rfr_model = load(f)


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
    y_pred = rfr_model.predict(df)
    response = {
        'prediction': y_pred[0],
        'model_name': 'rfr_model_v1',
        'model_last_updated': '2024-05-16',
    }
    return response
