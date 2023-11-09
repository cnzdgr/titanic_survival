'''
Module responsible from model training
'''
import sys
import os
d = os.getcwd()
par_2 = os.path.dirname(os.path.dirname(d))
sys.path.append(par_2)

from sklearn.model_selection import train_test_split
from app.model.model_config import config
from pipeline import survival_pipe
from app.model.processing.data_manager import load_dataset
from app.model.processing.data_manager import save_model


def run_training() -> None:
    data = load_dataset(file_name=config.a_config.training_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.m_config.features],  # predictors
        data[config.m_config.target],
        test_size=config.m_config.test_size,
        random_state=config.m_config.random_state,
    )

    survival_pipe.fit(X_train, y_train)

    save_model(model_to_keep=survival_pipe)


if __name__ == "__main__":
    run_training()