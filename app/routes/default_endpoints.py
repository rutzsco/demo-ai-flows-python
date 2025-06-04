from fastapi import APIRouter
from fastapi.responses import RedirectResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def redirect_to_docs():
    """Redirect root to Swagger documentation"""
    return RedirectResponse(url="/docs")

@router.get("/status")
async def status():
    logger.info("Status Endpoint")
    return {"message": "Hello World"}
