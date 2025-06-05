from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from typing import Optional
from app.routes.auth import get_api_key
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def redirect_to_docs():
    """Redirect root to Swagger documentation"""
    return RedirectResponse(url="/docs")

@router.get("/status")
async def status(api_key: Optional[str] = Depends(get_api_key)):
    logger.info("Status Endpoint")
    return {"message": "Hello World"}

@router.get("/test-auth")
async def test_auth(api_key: Optional[str] = Depends(get_api_key)):
    """Test endpoint to verify authentication"""
    return {"message": "Authentication working!", "api_key_received": api_key is not None}
