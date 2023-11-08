# Adding the main folder to sys.path
import sys
import os
d = os.getcwd()
sys.path.append(os.path.dirname(d))

from typing import Any
from fastapi import APIRouter
from app.model.predict import make_prediction


api_router = APIRouter()

@api_router.post("/predict", status_code=200)
async def predict() -> Any:
    return make_prediction()