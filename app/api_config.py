import logging
from typing import List
from pydantic import AnyHttpUrl, BaseModel

class LoggingSettings(BaseModel):
    LOGGING_LEVEL: int = logging.INFO
    

class Settings(BaseModel):
    API_V1_STR: str = "/api/v1"

    logging: LoggingSettings = LoggingSettings()

    # BACKEND_CORS_ORIGINS is a comma-separated list of origins
    # e.g: http://localhost,http://localhost:4200,http://localhost:3000
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # type: ignore
        "http://localhost:8000",  # type: ignore
        "https://localhost:3000",  # type: ignore
        "https://localhost:8000",  # type: ignore
    ]

    PROJECT_NAME: str = "Titanic Prediction API"

    class Config:
        case_sensitive = True
        

settings = Settings()

