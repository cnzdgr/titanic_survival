'''
Module responsible from
(i) Reading dataset, with or without pre-pipeline cleaning
(ii) Save/Load trained model (output of the sklearn Pipeline)
'''
# Adding the ROOT path 
from pathlib import Path
from typing import List
import sys
import os
d = os.getcwd()
par_3 = os.path.dirname(os.path.dirname(os.path.dirname(d)))
sys.path.append(par_3)

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from app.model.model_config import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset_raw(*, file_name: str) -> pd.DataFrame:
    '''Load the dataset as raw, without any cleaning'''
    df = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))    
    return df


def pre_pipeline_cleaning(*, df: pd.DataFrame) -> pd.DataFrame:
    '''Preliminary cleaning of the dataset, such as replacing weird values, etc '''
    return df


def load_dataset(*, file_name: str) -> pd.DataFrame:
    '''Load the dataset after pre-pipeline-cleaning''' 
    df = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    clean_df = pre_pipeline_cleaning(df=df)

    return clean_df


def save_model(*, model_to_keep: Pipeline) -> None:
    '''
    Saves the model (output of the pipeline) and removes previous models
    By this way, we will be sure there is only 1 model 
    '''   

    save_file_name = f"{config.a_config.model_save_file}{config.a_config.version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_model(files_to_keep=[save_file_name])
    
    joblib.dump(model_to_keep, save_path)
    

def load_model(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    return joblib.load(filename=file_path)


def remove_old_model(*, files_to_keep: List[str]) -> None:
    '''
    To remove old model/s (output of the pipeline).
    To be sure there is one-to-one mapping between package version
    and model version to be used in other parts
    '''
    
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in files_to_keep:
            model_file.unlink()
