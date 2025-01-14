# app/routes/workflow.py

from fastapi import APIRouter
from pydantic import BaseModel
from ..services.sk import SemanticKernelService
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
