'''
This file is to aggregate necessary directories
And combine/validate all configuation
'''

from pathlib import Path

from pydantic import BaseModel
from strictyaml import YAML, load

# Project Directories

MODEL_ROOT = Path(__file__).resolve().parent
ROOT = MODEL_ROOT.parent.parent
YAML_FILE_PATH = MODEL_ROOT / "config.yml"
DATASET_DIR = MODEL_ROOT / "datasets"
TRAINED_MODEL_DIR = MODEL_ROOT / "trained_models"

