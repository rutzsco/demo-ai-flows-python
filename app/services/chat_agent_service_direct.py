import shutil
import os, time
from typing import List
from dotenv import load_dotenv
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from app.models.api_models import ChatThreadRequest, ExecutionDiagnostics, RequestResult, Source, FileReference
from app.prompts.file_service import FileService

from azure.storage.blob import BlobServiceClient

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import AnnotationContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent, StreamingChatMessageContent, StreamingAnnotationContent, StreamingFileReferenceContent, ImageContent, FileReferenceContent
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from azure.ai.agents.models import CodeInterpreterTool, FileSearchTool, FilePurpose, FileSearchTool, CodeInterpreterTool

from azure.identity import DefaultAzureCredential
from pathlib import Path
import json
import uuid  

class ChatAgentServiceDirect:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        self.agent_id = os.getenv("AZURE_AI_AGENT_ID")
        blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        self.blob_service_client = None
        if blob_connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        pass

    
    async def run_chat_direct(self, request: ChatThreadRequest) -> str:

        project_client = AIProjectClient.from_connection_string(credential=DefaultAzureCredential(), conn_str=os.environ["AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"],)
        print(f"Created project client, project ID: {project_client}")
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("Agent: Chat") as current_span:
            with project_client:
                try:
                    user_message = request.message + " Save the result to a file."
                    print(f"User message: {user_message}")
                    
                    code_interpreter = CodeInterpreterTool()
                    
                    # create agent with code interpreter tool and tools_resources
                    agent = project_client.agents.create_agent(
                        model=os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"],
                        name="agent_run_chat_direct",
                        instructions="You are helpful agent.",
                        tools=code_interpreter.definitions,
                        tool_resources=code_interpreter.resources,
                    )
                    print(f"Created agent, agent ID: {agent.id}")

                    # create a thread
                    thread = project_client.agents.create_thread()
                    print(f"Created thread, thread ID: {thread.id}")

                    # create a message
                    message = project_client.agents.create_message(
                        thread_id=thread.id,
                        role="user",
                        content=user_message,
                    )
                    print(f"Created message, message ID: {message.id}")

                    # create and execute a run
                    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
                    print(f"Run finished with status: {run.status}")

                    if run.status == "failed":
                        # Check if you got "Rate limit is exceeded.", then you want to get more quota
                        print(f"Run failed: {run.last_error}")
                        return f"Run failed: {run.last_error}"

                    # print the messages from the agent
                    messages = project_client.agents.list_messages(thread_id=thread.id)
                    print(f"Messages: {messages}")
                    
                    # get the most recent message from the assistant
                    last_msg = messages.get_last_text_message_by_role("assistant")
                    if last_msg:
                        print(f"Last Message: {last_msg.text.value}")
    
                    # Access the attributes of the annotation directly
                    try:
                        annotation = last_msg.text.annotations[0]
                        print(annotation)
                    except Exception as e:
                        print(f"Error: {e}")
                        return f"annotation error: {e}"

                    # If you need to convert the annotation to a dictionary
                    annotation_dict = {
                        "type": annotation.type,
                        "text": annotation.text,
                        "file_path": annotation.file_path.file_id if annotation.file_path else None,
                    }
                    print(json.dumps(annotation_dict, indent=2))

                    root, extension = os.path.splitext(annotation_dict["text"])
                    file_name = str(uuid.uuid4()) + extension  # Convert UUID to string
                    file_id = annotation_dict["file_path"]
                    print(f"File name: {file_name}, File ID: {file_id}")
                    
                    project_client.agents.save_file(file_id=file_id, file_name=file_name)
                    print(f"Saved the file to: {file_name}") 

                    # save the newly created file
                    
                    blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
                    blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
                    if not blob_connection_string or not blob_container_name:
                        raise ValueError("Missing required environment variables for Azure Blob Storage.")

                    blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
                    blob_client = blob_service_client.get_blob_client(container=blob_container_name, blob=file_name)
                    with open(file_name, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)

                    # Get the full URL of the uploaded file
                    file_url = blob_client.url
                    
                    # delete local copies of the file
                    project_client.agents.delete_file(file_id)
                    os.remove(file_name)
                    if agent:
                        project_client.agents.delete_agent(agent.id)
                        
                    print("Done. You can now access the file from the following URL:")
                    print(file_url)   
                   
                    return(f"{last_msg.text.value} \nA copy in [cloud]({file_url})")
                    
                except Exception as e:
                    print(f"Error: {e}")
                    return f"Error: {e}"
                    

    async def run_chat_docs(self, query:str, temp_dir: str) -> str:
        print(f"Query: {query}")    
        print(f"Temp dir: {temp_dir}")
        
        project_client = AIProjectClient.from_connection_string(credential=DefaultAzureCredential(), conn_str=os.environ["AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"],)
        print(f"Created project client, project ID: {project_client}")
            
        with project_client:
            try:
                user_message = query
                print(f"User message: {user_message}")
                
                # create agent with code interpreter tool and tools_resources
                agent = project_client.agents.create_agent(
                    model=os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"],
                    name="agent_run_chat_docs",
                    instructions="You are helpful agent.",
                )
                print(f"Created agent, agent ID: {agent.id}")

                # read in files from the temp directory
                file_ids=[]
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    print(f"File path: {file_path}")
                    # upload the file
                    file = project_client.agents.upload_file_and_poll(file_path=file_path, purpose=FilePurpose.AGENTS)
                    file_ids.append(file.id)
                    print(f"Uploaded file, file ID: {file.id}")

                # create a vector store with the file you uploaded
                vector_store = project_client.agents.create_vector_store_and_poll(file_ids=[file.id], name="my_vectorstore")
                print(f"Created vector store, vector store ID: {vector_store.id}")
                
                # create a file search tool
                file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
                
                thread = project_client.agents.create_thread(
                    tool_resources=file_search_tool.resources
                )
                print(f"Created thread, thread ID: {thread.id}")

                message = project_client.agents.create_message(
                    thread_id=thread.id, role="user", content=user_message
                )
                print(f"Created message, message ID: {message.id}")

                run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)

                messages = project_client.agents.list_messages(thread_id=thread.id)
                print(f"Messages: {messages}")
                
                for file_id in file_ids:
                    if file_id:
                        project_client.agents.delete_file(file_id)
                if vector_store:
                    project_client.agents.delete_vector_store(vector_store.id)
                if agent:
                    project_client.agents.delete_agent(agent.id)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    
                last_msg = messages.get_last_text_message_by_role("assistant")
                if last_msg:
                    print(f"Last Message: {last_msg.text.value}")
                    return(f"{last_msg.text.value}")
                else:
                    return(f"No response from the assistant")
            
            except Exception as e:
                        print(f"Error: {e}")
                        return f"Error: {e}"