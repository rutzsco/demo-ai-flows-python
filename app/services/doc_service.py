import os, time
from typing import List
from dotenv import load_dotenv
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from app.models.api_models import ChatThreadRequest, ExecutionDiagnostics, RequestResult, Source
from app.prompts.file_service import FileService


from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import AnnotationContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent

from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from app.services.doc_plugin import DocPlugin

import pandas as pd
from azure.storage.blob import BlobServiceClient
import uuid
from typing import Annotated
from semantic_kernel.functions import kernel_function
    
intermediate_steps: list[ChatMessageContent] = []

async def handle_intermediate_steps(message: ChatMessageContent) -> None:
    intermediate_steps.append(message)
    
class DocService:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()


    async def run(self, request: ChatThreadRequest) -> str:
        # Validate the request object
        if not request.message:
            raise ValueError("No messages found in request.")
            
        ai_agent_settings = AzureAIAgentSettings.create()

        async with (
            DefaultAzureCredential() as creds,
            AzureAIAgent.create_client(
                credential=creds,
                conn_str=ai_agent_settings.project_connection_string.get_secret_value(),
            ) as client,
        ):
            AGENT_NAME = "DocService"
            AGENT_INSTRUCTIONS = "Tell me the row name and column names so I can create an empty excel file for you."

            # Create agent definition
            agent_definition = await client.agents.create_agent(
                model=ai_agent_settings.model_deployment_name,
                name=AGENT_NAME,
                instructions=AGENT_INSTRUCTIONS,
            )

            # Create the AzureAI Agent
            agent = AzureAIAgent(
                client=client,
                definition=agent_definition,
                plugins=[DocPlugin()],  # add the sample plugin to the agent
            )

            # Create a thread for the agent
            # If no thread is provided, a new thread will be
            # created and returned with the initial response
            thread: AzureAIAgentThread = None

            #user_inputs = [
            #    "Please create an empty Excel file with row names: ['A', 'B', 'C'] and column names: ['X', 'Y', 'Z'].",
            #    "Thank you",
            #]

            try:
                for user_input in [request.message]:
                    print(f"> User: '{user_input}'")
                    async for response in agent.invoke(
                        messages=user_input,
                        thread=thread,
                        on_intermediate_message=handle_intermediate_steps,
                    ):
                        print(f"> Agent: {response}")
                        thread = response.thread
            finally:
                # Cleanup: Delete the thread and agent
                await thread.delete() if thread else None
                await client.agents.delete_agent(agent.id)

            # Print the intermediate steps
            print("\nIntermediate Steps:")
            return_str = ""
            for msg in intermediate_steps:
                if any(isinstance(item, FunctionResultContent) for item in msg.items):
                    for fr in msg.items:
                        if isinstance(fr, FunctionResultContent):
                            print(f"Function Result:> {fr.result} for function: {fr.name}")
                elif any(isinstance(item, FunctionCallContent) for item in msg.items):
                    for fcc in msg.items:
                        if isinstance(fcc, FunctionCallContent):
                            print(f"Function Call:> {fcc.name} with arguments: {fcc.arguments}")
                else:
                    print(f"{msg.role}: {msg.content}")
                    if msg.role == "assistant":
                        return_str += f"{msg.content}"
        return return_str