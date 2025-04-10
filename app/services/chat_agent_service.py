import os
from typing import List

import semantic_kernel as sk
from dotenv import load_dotenv
from opentelemetry import trace

from app.models.api_models import ChatRequest, ExecutionDiagnostics, RequestResult
from app.prompts.file_service import FileService
from app.services.weather_plugin import WeatherPlugin

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.contents import  ChatMessageContent, FunctionCallContent
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from azure.ai.projects.models import BingGroundingTool 

class ChatAgentService:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Retrieve configuration from environment variables
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        
        if not api_key or not endpoint or not deployment_name:
            raise ValueError("Missing required environment variables for OpenAI configuration.")
    
        self.ai_agent_settings = AzureAIAgentSettings.create()
        self.file_service = FileService()

        pass


    async def run_chat(self, request: ChatRequest) -> str:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("Agent: Chat") as current_span:
            # Validate the request object
            if not request.messages:
                raise ValueError("No messages found in request.")
            user_message = request.messages[-1].content

            async with (DefaultAzureCredential() as creds, AzureAIAgent.create_client(credential=creds) as client,):
                
                # 1. Create an agent on the Azure AI agent service
                #agent_definition = await client.agents.create_agent(
                #    model=self.ai_agent_settings.model_deployment_name,
                #    name="Assistant",
                #    instructions="Answer the user's questions."
                #)
                agent_definition = await client.agents.get_agent(agent_id='asst_g8KPAp9JzA8LpqJOIiwIisL8')
                # 2. Create a Semantic Kernel agent for the Azure AI agent
                agent = AzureAIAgent(
                    client=client,
                    definition=agent_definition,
                )

                # 3. Create a thread for the agent
                # If no thread is provided, a new thread will be
                # created and returned with the initial response
                thread: AzureAIAgentThread = None

                try:
                    # 4. Invoke the agent with the specified message for response
                    response = await agent.get_response(messages=user_message, thread=thread)
                    thread = response.thread
                finally:
                    # 5. Cleanup: Delete the thread and agent
                    await thread.delete() if thread else None
 #                   await client.agents.delete_agent(agent.id) if agent else None   

            request_result = RequestResult(
                content=f"{response}")

            return request_result