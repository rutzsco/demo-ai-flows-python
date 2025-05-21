import shutil
import os, time
from typing import List, Optional, Any
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
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent, StreamingChatMessageContent, StreamingAnnotationContent, StreamingFileReferenceContent
from azure.identity.aio import DefaultAzureCredential
#from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    CodeInterpreterTool,
    FileSearchTool,
    FilePurpose,
    ListSortOrder,
    MessageRole,
    BingGroundingTool,
    FunctionTool, 
    ToolSet,
    MessageAttachment,
    MessageTextContent,
    AsyncAgentEventHandler,
    MessageDeltaChunk,
    ThreadMessage,
    ThreadRun,
    RunStep,

)
from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from pathlib import Path
import json
import uuid  # Import the uuid module

import asyncio
from azure.ai.agents.aio import AgentsClient as AsyncAgentsClient

class MyEventHandler(AsyncAgentEventHandler[str]):

    async def on_message_delta(self, delta: "MessageDeltaChunk") -> Optional[str]:
        return f"Text delta received: {delta.text}"

    async def on_thread_message(self, message: "ThreadMessage") -> Optional[str]:
        return f"ThreadMessage created. ID: {message.id}, Status: {message.status}"

    async def on_thread_run(self, run: "ThreadRun") -> Optional[str]:
        return f"ThreadRun status: {run.status}"

    async def on_run_step(self, step: "RunStep") -> Optional[str]:
        return f"RunStep type: {step.type}, Status: {step.status}"

    async def on_error(self, data: str) -> Optional[str]:
        return f"An error occurred. Data: {data}"

    async def on_done(self) -> Optional[str]:
        return "Stream completed."

    async def on_unhandled_event(self, event_type: str, event_data: Any) -> Optional[str]:
        return f"Unhandled Event Type: {event_type}, Data: {event_data}"
    
class ChatAgentService:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        self.agent_id = os.getenv("AZURE_AI_AGENT_ID")
        blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        self.blob_service_client = None
        if blob_connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        pass


    async def run_chat_sk(self, request: ChatThreadRequest) -> str:

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("Agent: run_chat_sk") as current_span:
            # Validate the request object
            if not request.message:
                raise ValueError("No messages found in request.")
            user_message = request.message

            # Check if a file was specified in the request
            file_content = None
            ai_project_file = None
            temp_file_path = None
            if request.file:
                try:
                    # Get the blob container name from environment variables
                    blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
                    blob_client = self.blob_service_client.get_blob_client(container=blob_container_name, blob=request.file)
                    download_stream = blob_client.download_blob()
                    file_content = download_stream.readall()
                    print(f"Downloaded file '{request.file}' from blob storage")
                    
                    # Create a temporary file to upload to AI Project service
                    temp_file_path = f"./temp_{uuid.uuid4()}{os.path.splitext(request.file)[1]}"
                    with open(temp_file_path, "wb") as f:
                        f.write(file_content)
                    
                except Exception as e:
                    print(f"Error processing file: {e}")
                    raise e
                
            # Create an Azure AI agent client
            async with AsyncDefaultAzureCredential() as creds:
                async with AsyncAgentsClient(
                    endpoint=os.environ["AZURE_AI_AGENT_PROJECT_ENDPOINT"],
                    credential=creds,
                ) as agents_client:
                    file = await agents_client.files.upload_and_poll(file_path=temp_file_path, purpose=FilePurpose.AGENTS)
                    print(f"Uploaded file to AI Project service.")
                    
                    # Clean up the temporary file if it was created
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    
                    code_interpreter = CodeInterpreterTool()
                    print(f"Code interpreter tool: {code_interpreter}")
                    
                    # Create agent
                    agent_name="agent_run_chat_sk"
                    agent = await agents_client.create_agent(
                        model=os.environ["MODEL_DEPLOYMENT_NAME"],
                        name=agent_name,
                        instructions="You are helpful agent",
                        tools=code_interpreter.definitions,
                        tool_resources=code_interpreter.resources,
                    )
                    print(f"Created agent, agent ID: {agent.id}")
                
                    thread = await agents_client.threads.create()
                    print(f"Created thread, thread ID: {thread.id}")
                    
                    # Create a message with the file search attachment
                    attachment = MessageAttachment(file_id=file.id, tools=FileSearchTool().definitions)
                    message = await agents_client.messages.create(
                        thread_id=thread.id,
                        role="user",
                        content=user_message,
                        attachments=[attachment],
                    )
                    print(f"Created message, message ID: {message.id}")
                    
                    async with await agents_client.runs.stream(
                        thread_id=thread.id, agent_id=agent.id, event_handler=MyEventHandler()
                    ) as stream:
                        async for event_type, event_data, func_return in stream:
                            print(f"Received data.")
                            print(f"Streaming receive Event Type: {event_type}")
                            print(f"Event Data: {str(event_data)[:100]}...")
                            print(f"Event Function return: {func_return}\n")

                    await agents_client.delete_agent(agent.id)
                    print("Deleted agent")
                    
                    # Get the last message from the agent
                    last_msg=""
                    file_urls = []
                    messages = agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
                    async for msg in messages:
                        print(f"{msg.role}: {msg.content}")
                        for ann in msg.file_path_annotations:
                            print("File Paths:")
                            print(f"  Type: {ann.type}")
                            print(f"  Text: {ann.text}")
                            print(f"  File ID: {ann.file_path.file_id}")
                            print(f"  Start Index: {ann.start_index}")
                            print(f"  End Index: {ann.end_index}")
                        
                            filename=os.path.basename(ann.text)
                            #file_name = str(uuid.uuid4()) + extension  # Convert UUID to string
                            file_id = ann.file_path.file_id
                            print(f"File name: {filename}, File ID: {file_id}")
                            
                            await agents_client.files.save(file_id=file_id, file_name=filename)
                            print(f"Saved the file to: {filename}") 
                            
                            # save the newly created file to blob storage
                            blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
                            blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
                            if not blob_connection_string or not blob_container_name:
                                raise ValueError("Missing required environment variables for Azure Blob Storage.")

                            blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
                            blob_client = blob_service_client.get_blob_client(container=blob_container_name, blob=filename)
                            with open(filename, "rb") as data:
                                blob_client.upload_blob(data, overwrite=True)
                                
                            # Clean up the temporary file if it was created
                            if filename and os.path.exists(filename):
                                os.remove(filename)
                                
                            # Get the full URL of the uploaded file
                            file_urls.append({"name": filename, "url": blob_client.url})
                            
                        if msg.role == MessageRole.AGENT:
                            last_part = msg.content[-1]
                            if isinstance(last_part, MessageTextContent):
                                print(f"{msg.role}: {last_part.text.value}")
                                last_msg =last_part.text.value
                                
                    # Clean up the temporary file if it was created
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    if filename and os.path.exists(filename):
                        os.remove(filename)
                                        
                    if len(file_urls)>0:
                        file_url = ', '.join([f"[{file['name']}]({file['url']})" for file in file_urls])
                        return(f"{last_msg} \nBlob copy: {file_url}")
                    else:
                        return(f"{last_msg}")

    async def run_chat_direct(self, request: ChatThreadRequest) -> str:
        agent_name="agent_run_chat_direct"
        agents_client = AgentsClient(credential=DefaultAzureCredential(), endpoint=os.environ["AZURE_AI_AGENT_PROJECT_ENDPOINT"])
        print(f"Created project client, project ID: {agents_client}")
        
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("Agent: run_chat_direct") as current_span:
            with agents_client:
                try:
                    user_message = request.message + " Save the result to a file."
                    print(f"User message: {user_message}")
                    
                    code_interpreter = CodeInterpreterTool()
                    print(f"Code interpreter tool: {code_interpreter}")
                    
                    # create agent with code interpreter tool and tools_resources
                    agent = agents_client.create_agent(
                        model=os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"],
                        name=agent_name,
                        instructions="You are helpful agent.",
                        tools=code_interpreter.definitions,
                        tool_resources=code_interpreter.resources,
                    )
                    print(f"Created agent, agent ID: {agent.id}")

                    # create a thread
                    thread = agents_client.threads.create()
                    print(f"Created thread, thread ID: {thread.id}")

                    # create a message
                    message = agents_client.messages.create(    
                        thread_id=thread.id,
                        role="user",
                        content=user_message,
                    )
                    print(f"Created message, message ID: {message.id}")

                    # create and execute a run
                    run = agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
                    print(f"Run finished with status: {run.status}")

                    if run.status == "failed":
                        # Check if you got "Rate limit is exceeded.", then you want to get more quota
                        print(f"Run failed: {run.last_error}")
                        return f"Run failed: {run.last_error}"
                    
                    # get messages
                    messages = agents_client.messages.list(thread_id=thread.id)
                    print(f"Messages: {messages}")
    
                    file_urls = []
                    for msg in messages:
                        # Print details of every file-path annotation
                        for ann in msg.file_path_annotations:
                            print("File Paths:")
                            print(f"  Type: {ann.type}")
                            print(f"  Text: {ann.text}")
                            print(f"  File ID: {ann.file_path.file_id}")
                            print(f"  Start Index: {ann.start_index}")
                            print(f"  End Index: {ann.end_index}")
                            
                            filename=os.path.basename(ann.text)
                            #file_name = str(uuid.uuid4()) + extension  # Convert UUID to string
                            file_id = ann.file_path.file_id
                            print(f"File name: {filename}, File ID: {file_id}")
                            
                            agents_client.files.save(file_id=file_id, file_name=filename)
                            print(f"Saved the file to: {filename}") 
                            
                            # save the newly created file to blob storage
                            blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
                            blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
                            if not blob_connection_string or not blob_container_name:
                                raise ValueError("Missing required environment variables for Azure Blob Storage.")

                            blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
                            blob_client = blob_service_client.get_blob_client(container=blob_container_name, blob=filename)
                            with open(filename, "rb") as data:
                                blob_client.upload_blob(data, overwrite=True)
                                
                            # Get the full URL of the uploaded file
                            file_urls.append({"name": filename, "url": blob_client.url})

                            # delete local copies of the file
                            os.remove(filename)
                    
                    last_msg = agents_client.messages.get_last_message_text_by_role(thread_id=thread.id, role=MessageRole.AGENT)
                    if last_msg:
                        print(f"Last Message: {last_msg.text.value}")
        
                    # delete agent
                    agents_client.delete_agent(agent.id)
                    print("Deleted agent")
                     
                    if len(file_urls)>0:
                        file_url = ', '.join([f"[{file['name']}]({file['url']})" for file in file_urls])
                        return(f"{last_msg.text.value} \nBlob copy: {file_url}")
                    else:
                        return(f"{last_msg.text.value}")
                    
                except Exception as e:
                    print(f"Error: {e}")
                    return f"Error: {e}"
                    

    async def run_chat_docs(self, query:str, temp_dir: str) -> str:
        print(f"Query: {query}")    
        print(f"Temp dir: {temp_dir}")
        
        agent_name="agent_run_chat_docs"
        agents_client = AgentsClient(credential=DefaultAzureCredential(), endpoint=os.environ["AZURE_AI_AGENT_PROJECT_ENDPOINT"])
        print(f"Created project client, project ID: {agents_client}")
            
        with agents_client:
            try:
                user_message = query
                print(f"User message: {user_message}")
                
                # read in files from the temp directory
                file_ids=[]
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    print(f"File path: {file_path}")
                    # upload the file
                    file = agents_client.files.upload_and_poll(file_path=file_path, purpose=FilePurpose.AGENTS)
                    file_ids.append(file.id)
                    print(f"Uploaded file, file ID: {file.id}")

                # create a vector store with the file you uploaded
                vector_store = agents_client.vector_stores.create_and_poll(file_ids=[file.id], name=agent_name+"_vectorstore")
                print(f"Created vector store, vector store ID: {vector_store.id}")
                
                # create a file search tool
                file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
                print(f"File search tool: {file_search_tool}")
                
                # Initialize agent bing tool and add the connection id
                conn_id = os.environ["AZURE_AI_AGENT_BING_CONNECTION_NAME"]
                bing = BingGroundingTool(connection_id=conn_id)
                print(f"Bing grounding tool: {bing}")
    
                toolset = ToolSet()
                toolset.add(file_search_tool)
                toolset.add(bing)

                # create agent with code interpreter tool and tools_resources
                agent = agents_client.create_agent(
                    model=os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"],
                    name=agent_name,
                    instructions="You are helpful agent.",
                )
                print(f"Created agent, agent ID: {agent.id}")

                # Create thread for communication
                thread = agents_client.threads.create(
                    tool_resources=toolset.resources,
                )
                print(f"Created thread, ID: {thread.id}")
                
                # create a message
                message = agents_client.messages.create(    
                    thread_id=thread.id,
                    role="user",
                    content=user_message + ".Verify the information with citations"
                )

                # Create and process agent run in thread with tools
                run = agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
                print(f"Run finished with status: {run.status}")
                
                if run.status == "failed":
                    # Check if you got "Rate limit is exceeded.", then you want to get more quota
                    print(f"Run failed: {run.last_error}")
                    return f"Run failed: {run.last_error}"

                
                # [START teardown]
                # Delete the file when done
                agents_client.vector_stores.delete(vector_store.id)
                print("Deleted vector store")

                agents_client.files.delete(file_id=file.id)
                print("Deleted file")

                # Delete the agent when done
                agents_client.delete_agent(agent.id)
                print("Deleted agent")
                # [END teardown]
                
                last_msg = agents_client.messages.get_last_message_text_by_role(thread_id=thread.id, role=MessageRole.AGENT)
                if last_msg:
                    print(f"Last Message: {last_msg.text.value}")
                    return(f"{last_msg.text.value}")
                else:
                    return(f"")
            
            except Exception as e:
                        print(f"Error: {e}")
                        return f"Error: {e}"