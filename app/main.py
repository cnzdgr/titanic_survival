# Adding the main folder to sys.path
import sys
import os
d = os.getcwd()
sys.path.append(os.path.dirname(d))
from typing import Any

from app.api import api_router
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse


root_router = APIRouter()

app = FastAPI(
    title="/api/v1")

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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")