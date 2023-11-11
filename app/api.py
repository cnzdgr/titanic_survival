import os
import sys
import json
import numpy as np
import pandas as pd

# Adding the main folder to sys.path
d = os.getcwd()
par = os.path.dirname(d)
sys.path.append(par)

from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from app.model.predict import make_prediction
from app.schemas import predict_schema as schemas


api_router = APIRouter()

@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleTitanicDataInputs) -> Any:
    '''
    POST request to the API, response needs to be matching with the schema and contains
    (i) errors, (ii) version, (iii) predictions
    '''
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    logger.info(f"Predicting using inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))
    
    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Model Predictions: {results.get('predictions')}")

    return results

