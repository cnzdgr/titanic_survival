'''
This file is to aggregate necessary directories
And combine/validate all model configuations
'''

from pathlib import Path

from pydantic import BaseModel
from strictyaml import YAML, load

# Project Directories
MODEL_ROOT = Path(__file__).resolve().parent
ROOT = MODEL_ROOT.parent.parent
YAML_FILE_PATH = MODEL_ROOT / "model_details.yaml"
DATASET_DIR = MODEL_ROOT / "datasets"
TRAINED_MODEL_DIR = MODEL_ROOT / "trained_models"


'''Validating configuration by object type'''
class AppConfig(BaseModel):
    # High level configuaration
    training_data_file: str
    test_data_file: str
    model_save_file: str


class ModelConfig(BaseModel):
    # Model-dependent configuration
    target: str
    features: list[str]
    test_size: float
    random_state: int
    penalty: str
    solver: str
    vars_to_check_existance: list[str]
    numerical_vars_with_na: list[str]
    vars_to_log_transform: list[str]
    vars_to_binarize: list[str]



class Config(BaseModel):
    '''Main configuration object'''

    a_config: AppConfig
    m_config: ModelConfig


def find_yaml_file() -> Path:
    '''Locates the model configuration file'''

    if YAML_FILE_PATH.is_file():
        return YAML_FILE_PATH
    raise Exception(f".yml not found at specified directory: {YAML_FILE_PATH!r}")


def fetch_config_from_yaml(yml_path: Path = None) -> YAML:
    '''Parsing .yml file'''

    if not yml_path:
        yml_path = find_yaml_file()

    if yml_path:
        with open(yml_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {yml_path}")


def validate_config(parsed_config: YAML = None) -> Config:
    '''Validate all config values from the .yml file'''

    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()


    # specify the data attribute from the strictyaml YAML type.
    _config = Config(a_config = AppConfig(**parsed_config.data), 
                     m_config = ModelConfig(**parsed_config.data),
                     )

    return _config
