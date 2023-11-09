'''
Defining the feature engineering and model training pipeline
'''

import sys
import os
from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer
from feature_engine.transformation import LogTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

d = os.getcwd()
par_2 = os.path.dirname(os.path.dirname(d))
sys.path.append(par_2)
print(par_2)

from app.model.model_config import config
from app.model.processing import custom_preprocess as cp


survival_pipe = Pipeline([
    # -- IMPUTATION --
    (
     'missing_imputation', 
     cp.MissingBinaryTransformer(variables=config.m_config.vars_to_check_existance)
     ),
    (
     'missing_indicator',
     AddMissingIndicator(variables=config.m_config.numerical_vars_with_na)
     ),
    (
     'mean_imputation',
     MeanMedianImputer(imputation_method='mean', variables=config.m_config.numerical_vars_with_na)
     ),

    # -- TRANSFORMATION --
    (
     'gender_binarize',
     cp.GenderBinaryTransformer(variables=config.m_config.vars_to_binarize)
     ),
    (
     'fare_nonzero',
     cp.NonZeroTransformer(variables=config.m_config.vars_to_log_transform)
     ),
    (
     'log',
     LogTransformer(variables=config.m_config.vars_to_log_transform)
     ),

    # -- SCALING AND PREDICTION --
    (
     'scaler',
     MinMaxScaler()
     ),
    (
     'Logit',
     LogisticRegression(penalty=config.m_config.penalty,
                        solver=config.m_config.solver,
                        random_state=config.m_config.random_state)
     ),
])
