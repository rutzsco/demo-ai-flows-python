import os
from dotenv import load_dotenv
from app.models.api_models import AgentCreateRequest
from app.prompts.file_service import FileService

from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings

from azure.identity import DefaultAzureCredential


class AzureAIAgentFactory:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
 
        self.ai_agent_settings = AzureAIAgentSettings.create()

        pass


    async def run_create_azure_ai_agent(self, request: AgentCreateRequest) -> str:

            creds = DefaultAzureCredential()
            async with (AzureAIAgent.create_client(credential=creds) as client,):
                agent_definition = await client.agents.create_agent(model=request.model, name=request.name, instructions=request.instructions)
                
            return agent_definition

