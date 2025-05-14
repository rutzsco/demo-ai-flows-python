import os
import uuid
import base64
import mimetypes
from typing import Tuple, Optional, Any

from azure.storage.blob import BlobServiceClient
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import FilePurpose
from azure.identity import DefaultAzureCredential

from semantic_kernel.contents import ChatMessageContent, ImageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

async def download_and_process_file(blob_service_client: BlobServiceClient, file_name: str) -> Tuple[Optional[bytes], Any]:
    """
    Downloads a file from blob storage and processes it for AI Project service.
    
    Args:
        blob_service_client: BlobServiceClient instance
        file_name: Name of the file to download and process
        
    Returns:
        tuple: (file_content, ai_project_file) where file_content is the binary content
              and ai_project_file is the reference to the uploaded file in AI Project service
    """
    file_content = None
    ai_project_file = None
    temp_file_path = None
    
    try:
        # Get the blob container name from environment variables
        blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
        blob_client = blob_service_client.get_blob_client(container=blob_container_name, blob=file_name)
        download_stream = blob_client.download_blob()
        file_content = download_stream.readall()
        print(f"Downloaded file '{file_name}' from blob storage")
        
        # Create a temporary file to upload to AI Project service
        temp_file_path = f"./temp_{uuid.uuid4()}{os.path.splitext(file_name)[1]}"
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Create an AI Project client and upload the file
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(), 
            conn_str=os.environ["AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"]
        )
        ai_project_file = project_client.agents.upload_file_and_poll(
            file_path=temp_file_path, 
            purpose=FilePurpose.AGENTS
        )
        print(f"Uploaded file to AI Project service with ID: {ai_project_file.id}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        # Continue without the file if there's an error
    finally:
        # Clean up the temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return file_content, ai_project_file

def create_chat_message_content(user_message: str, file_content=None, file_name=None, ai_project_file=None) -> ChatMessageContent:
    """
    Creates a ChatMessageContent object based on the user message and optional file content.
    
    Args:
        user_message: The text message from the user
        file_content: Binary content of the file (if any)
        file_name: Name of the file (if any)
        ai_project_file: Reference to the file uploaded to AI Project service (if any)
        
    Returns:
        ChatMessageContent: The formatted chat message content
    """
    # If we have an image file, include it in the ChatMessageContent
    if file_content and file_name and any(file_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
        # When file is an image, create a chat message with image content
        # Determine mime type based on file extension
        mime_type, _ = mimetypes.guess_type(file_name)
        
        # Create a data URL with the base64-encoded image
        base64_image = base64.b64encode(file_content).decode('utf-8')
        data_url = f"data:{mime_type};base64,{base64_image}"
        
        # Create an ImageContent object with the data URL
        image_content = ImageContent(data_uri=data_url)
        
        # Create the chat message with both text and image content
        return ChatMessageContent(role=AuthorRole.USER, content=user_message, items=[image_content])
    else:
        # Regular text-only message
        return ChatMessageContent(role=AuthorRole.USER, content=user_message)
