from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings

from app.models.api_models import AgentCreateRequest


class AzureAIAgentFactory:
    def __init__(self):
        # Load environment variables from .env file
        # Semantic kernel loads the .env file automatically, so we don't need to do it here.
        self.ai_agent_settings = AzureAIAgentSettings.create()

    async def run_create_azure_ai_agent(self, request: AgentCreateRequest) -> str:
        creds = DefaultAzureCredential()
        async with (AzureAIAgent.create_client(credential=creds) as client,):
            # First, check if an agent with this name already exists
            try:
                # List all agents and check if one with the requested name exists
                agents_pager = client.agents.list_agents()
                async for agent in agents_pager:
                    if agent.name == request.name:
                        # Agent with this name already exists, return it
                        return agent
            except Exception as e:
                # If listing fails, continue with creation
                print(f"Warning: Could not list existing agents: {e}")
            
            # No existing agent found, create a new one
            agent_definition = await client.agents.create_agent(model=request.model, name=request.name, instructions=request.instructions)
            
        return agent_definition