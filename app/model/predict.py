'''
Uses the saved model and makes prediction
'''
import sys
import os 
import typing as t
import pandas as pd
from pathlib import Path

d = os.getcwd()
par = os.path.dirname(d)
par_par = os.path.dirname(par)
sys.path.append(par_par)
    
from app.model.model_config import config
from app.model.processing.data_manager import load_model
from app.model.processing.validation import validate_inputs

model_file_name = f"{config.a_config.model_save_file}{config.a_config.version}.pkl"
_titanic_pipe = load_model(file_name=model_file_name)


def make_prediction(*,input_data: t.Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": config.a_config.version, "errors": errors}

    if not errors:
        predictions = _titanic_pipe.predict(
            X=validated_data[config.m_config.features]
        )
        results = {
            "predictions": predictions,
            "version": config.a_config.version,
            "errors": errors,
        }
        
    return results

'''
# Use if you need to check the function make_prediction
from processing.data_manager import load_dataset
data = load_dataset(file_name=config.a_config.test_data_file)
new_preds = make_prediction(input_data=data)
result = pd.DataFrame(new_preds['predictions'], columns=["Survived"])
result.to_csv('result.csv', index=False)
'''