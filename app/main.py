from typing import Any
import sys
import os

# Adding the main folder to sys.path
d = os.getcwd()
sys.path.append(os.path.dirname(d))


from loguru import logger
from fastapi.middleware.cors import CORSMiddleware
from app.api_config import settings

from app.api import api_router
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse


root_router = APIRouter()

app = FastAPI(
    title="/api/titanic_survival_predictor")

@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)


app.include_router(api_router)
app.include_router(root_router)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    import uvicorn
    logger.warning("Running in development mode.")

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")