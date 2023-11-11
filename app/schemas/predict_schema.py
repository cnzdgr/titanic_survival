'''
Schema for the API request and response
'''

from typing import Any, List, Optional
from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class TitanicInputSchema(BaseModel):
    Cabin: Optional[str]
    Fare: Optional[float]
    Sex: Optional[str]
    Age: Optional[float]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Cabin": "AAA",
                        "Fare": 120,
                        "Sex": "male",
                        "Age": 28,
                    }
                ]
            }
        }
