import numpy as np
import pandas as pd
import sys
import os

d = os.getcwd()
par_3 = os.path.dirname(os.path.dirname(os.path.dirname(d)))
sys.path.append(par_3)

from pydantic import BaseModel, ValidationError
from typing import List, Optional, Tuple
from app.model.model_config import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    '''
    There are NA values that we expect in our training dataset
    But if there is NA in other cells, drop corresponding rows
    To keep the pipeline safe
    Expected behaviour is: there shouldn't be any dropped values
    '''
    validated_df = input_data.copy()
    new_vars_with_na = [
        var for var in config.m_config.features
        if var
        not in config.m_config.numerical_vars_with_na
        and validated_df[var].isnull().sum() > 0
    ]
    validated_df.dropna(subset=new_vars_with_na, inplace=True)

    return validated_df


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    '''Check inputs and be sure that values can be processed'''

    # convert syntax error field names (beginning with numbers)
    relevant_data = input_data[config.m_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TitanicInputSchema(BaseModel):
    Cabin: Optional[str]
    Fare: Optional[float]
    Sex: Optional[str]
    Age: Optional[float]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicInputSchema]