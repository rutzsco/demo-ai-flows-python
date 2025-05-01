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
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent, StreamingChatMessageContent, StreamingAnnotationContent
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread

import pandas as pd
from azure.storage.blob import BlobServiceClient
import uuid
from typing import Annotated
from semantic_kernel.functions import kernel_function

class DocPlugin:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Retrieve configuration from environment variables
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
        
        if not api_key or not endpoint or not deployment_name:
            raise ValueError("Missing required environment variables for OpenAI configuration.")
        if not blob_connection_string or not blob_container_name:
            raise ValueError("Missing required environment variables for Azure Blob Storage.")

        self.ai_agent_settings = AzureAIAgentSettings.create()
        self.file_service = FileService()
        self.blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        self.blob_container_name = blob_container_name
        
    @kernel_function(
        name="create_empty_excel_file",
        description="Create an empty Excel file based on the given row names and column names."
    )
    def create_empty_excel_file(
        self,
        row_names: Annotated[List[str], "A list of row names"],
        col_names: Annotated[List[str], "A list of column names"]
    ) -> Annotated[str, "Return the file name. The file has been uploaded to Azure Blob Storage."]:
        # Generate a unique file name using UUID
        file_name = f"{uuid.uuid4()}.xlsx"
        file_path = file_name

        # Create a DataFrame with the given row and column names
        df = pd.DataFrame(index=row_names, columns=col_names)
        df.to_excel(file_path, index=True)  # Save row names as the index in the Excel file

        # Upload the file to Azure Blob Storage
        blob_client = self.blob_service_client.get_blob_client(container=self.blob_container_name, blob=file_name)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        # delete local file
        os.remove(file_path)
        
        # Get the full URL of the uploaded file
        file_url = blob_client.url

        return f"File uploaded to Azure Blob Storage: {file_url}"