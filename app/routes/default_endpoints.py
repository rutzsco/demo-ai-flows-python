from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader
from typing import Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Define API Key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

@router.get("/")
async def redirect_to_docs():
    """Redirect root to Swagger documentation"""
    return RedirectResponse(url="/docs")

@router.get("/status")
async def status(api_key: Optional[str] = Depends(api_key_header)):
    logger.info("Status Endpoint")
    return {"message": "Hello World"}
