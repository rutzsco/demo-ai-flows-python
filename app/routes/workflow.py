from fastapi import APIRouter
from pydantic import BaseModel
from app.models.api_models import ChatRequest
from app.services.sk import SemanticKernelService
import asyncio
router = APIRouter()

sk_service = SemanticKernelService()

class WorkflowInput(BaseModel):
    data: str

@router.post("/workflow")
async def run_workflow(input_data: WorkflowInput):
    """
    POST endpoint for executing a Semantic Kernel workflow.
    """
    result = await sk_service.run_workflow(input_data.data)
    return {"result": result}

@router.post("/weather")
async def run_weather_workflow(input_data: ChatRequest):
    """
    POST endpoint for executing a weather workflow.
    """
    result = await sk_service.run_weather(input_data)
    return {"result": result}

@router.post("/agent/weather")
async def run_weather_workflow(input_data: ChatRequest):
    """
    POST endpoint for executing a weather workflow.
    """
    result = await sk_service.run_weather_agent(input_data)
    return {"result": result}
