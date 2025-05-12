from fastapi import APIRouter
from pydantic import BaseModel
from app.models.api_models import ChatRequest, ChatThreadRequest, AgentCreateRequest
from app.services.sk import SemanticKernelService
from app.services.weather_agent_service import WeatherAgentService
from app.services.chat_agent_service import ChatAgentService
from app.services.azure_ai_agent_factory import AzureAIAgentFactory
import asyncio
router = APIRouter()

sk_service = SemanticKernelService()
weather_service = WeatherAgentService()
chat_agent_service = ChatAgentService()
azure_ai_agent_factory = AzureAIAgentFactory()

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
    result = await weather_service.run_weather(input_data)
    return {"result": result}

@router.post("/agent/weather")
async def run_weather_workflow(input_data: ChatThreadRequest):
    """
    POST endpoint for executing a weather workflow.
    """
    result = await weather_service.run_weather_agent(input_data)
    return {"result": result}

@router.post("/agent/chat")
async def run_weather_workflow(input_data: ChatThreadRequest):
    """
    POST endpoint for executing a weather workflow.
    """
    result = await chat_agent_service.run_chat_sk(input_data)
    return {"result": result}

@router.post("/agent/chat-direct")
async def run_weather_workflow(input_data: ChatThreadRequest):
    """
    POST endpoint for executing a weather workflow.
    """
    result = await chat_agent_service.run_chat_direct(input_data)
    return {"result": result}

@router.post("/agent/chat/create")
async def run_weather_workflow(input_data: AgentCreateRequest):
    """
    POST endpoint for executing a weather workflow.
    """
    result = await azure_ai_agent_factory.run_create_azure_ai_agent(input_data)
    return {"result": result}
