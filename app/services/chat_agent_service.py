import os
import uuid
from dotenv import load_dotenv
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from app.models.api_models import ChatThreadRequest, RequestResult, Source, FileReference

from azure.storage.blob import BlobServiceClient

from semantic_kernel.contents import ChatMessageContent, FunctionCallContent, StreamingChatMessageContent, StreamingAnnotationContent, StreamingFileReferenceContent, ImageContent, FileReferenceContent
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread

from azure.identity.aio import DefaultAzureCredential
from azure.ai.agents.models import FileSearchTool, FileSearchTool
from azure.identity import DefaultAzureCredential

from app.utils.file_utils import download_and_process_file, create_chat_message_content

class ChatAgentService:
    def __init__(self):

        load_dotenv()

        self.agent_id = os.getenv("AZURE_AI_AGENT_ID")
        blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        self.blob_service_client = None
        if blob_connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        pass

    async def run_chat_sk(self, request: ChatThreadRequest) -> str:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("Agent: Chat") as current_span:
            # Validate the request object
            if not request.message:
                raise ValueError("No messages found in request.")
            user_message = request.message

            # Check if a file was specified in the request
            file_content = None
            ai_project_file = None
            if request.file and self.blob_service_client:
                file_content, ai_project_file = await download_and_process_file(self.blob_service_client, request.file)

            # Define a list to hold callback message content
            intermediate_steps: list[str] = []
            async def handle_intermediate_steps(message: ChatMessageContent) -> None:
                print("handle_intermediate_steps")
                if any(isinstance(item, FunctionCallContent) for item in message.items):
                    for fcc in message.items:
                        if isinstance(fcc, FunctionCallContent):
                            intermediate_steps.append(f"Function Call: {fcc.name} with arguments: {fcc.arguments}")
                        else:
                            print(f"{message.role}: {message.content}")
                else:
                    print(f"{message.role}: {message.content}")

            creds = DefaultAzureCredential()
            async with (AzureAIAgent.create_client(credential=creds) as client,):
                  # Create an agent on the Azure AI agent service. Create a Semantic Kernel agent for the Azure AI agent
                agent_definition = await client.agents.get_agent(agent_id=self.agent_id)
                agent = AzureAIAgent(client=client, definition=agent_definition)
                thread: AzureAIAgentThread  = None
                if request.thread_id:
                    thread = AzureAIAgentThread(client=client, thread_id=request.thread_id)               
                if ai_project_file:
                    try:
                        project_client = AIProjectClient(credential=DefaultAzureCredential(), endpoint=os.environ["AZURE_AI_AGENT_ENDPOINT"])
                        with project_client:
                            # Check if we need to create a thread with vector store functionality
                            thread_id = request.thread_id
                            if not thread and not request.thread_id:
                                # Create a vector store first
                                print(f"Creating new vector store with file ID: {ai_project_file.id}")
                                vector_store = project_client.agents.vector_stores.create_and_poll(file_ids=[ai_project_file.id], name=f"rutzsco_paif_vs_{uuid.uuid4()}")
                                print(f"Created vector store with ID: {vector_store.id}")
                                
                                # Create file search tool with the vector store
                                file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
                                
                                # Create thread with the file search tool resources
                                print("Creating new thread with vector store attachment")
                                thread_response = project_client.agents.threads.create(tool_resources=file_search_tool.resources)
                                thread_id = thread_response.id
                                thread = AzureAIAgentThread(client=client, thread_id=thread_id)
                                print(f"Created new thread with ID: {thread_id} and vector store {vector_store.id}")
                            elif thread_id:
                                # Check if the existing thread already has a vector store
                                vector_store_id = None
                                try:
                                    thread_details = project_client.agents.threads.get(thread_id)
                                    if (hasattr(thread_details, 'tool_resources') and 
                                        thread_details.tool_resources and
                                        hasattr(thread_details.tool_resources, 'file_search') and
                                        thread_details.tool_resources.file_search and
                                        hasattr(thread_details.tool_resources.file_search, 'vector_store_ids')):
                                        vector_store_ids = thread_details.tool_resources.file_search.vector_store_ids
                                        if vector_store_ids:
                                            vector_store_id = vector_store_ids[0]
                                            print(f"Found existing vector store ID: {vector_store_id}")
                                except Exception as e:
                                    print(f"Could not get thread details: {e}")
                                
                                if vector_store_id:
                                    # Add the file to the existing vector store
                                    print(f"Adding file {ai_project_file.id} to existing vector store {vector_store_id}")
                                    project_client.agents.vector_store_files.create_and_poll(vector_store_id=vector_store_id, file_id=ai_project_file.id)
                                    print(f"Added file to existing vector store {vector_store_id}")
                                else:
                                    # Create a new vector store and update the thread
                                    print(f"Creating new vector store with file ID: {ai_project_file.id}")
                                    vector_store = project_client.agents.vector_stores.create_and_poll(file_ids=[ai_project_file.id], name=f"rutzsco_paif_vs_{uuid.uuid4()}")
                                    print(f"Created vector store with ID: {vector_store.id}")
                                    
                                    # Update the existing thread with file search tool resources
                                    file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
                                    project_client.agents.threads.update(thread_id=thread_id, tool_resources=file_search_tool.resources)
                                    print(f"Updated thread {thread_id} with vector store {vector_store.id}")
                            
                    except Exception as e:
                        print(f"Error setting up vector store: {e}")

                annotations: list[StreamingAnnotationContent] = []
                files: list[StreamingFileReferenceContent] = []
                sources = []
                file_references = []
                responseContent = ''
                code_output_content = ''
                try:
                    # Create the appropriate ChatMessageContent based on whether we have a file
                    cmc = create_chat_message_content(
                        user_message=user_message, 
                        #file_content=file_content, 
                        #file_name=request.file, 
                        #ai_project_file=ai_project_file
                    )
                    
                    async for result in agent.invoke_stream(messages=cmc, thread=thread, on_intermediate_message=handle_intermediate_steps):
                        response = result
                        
                        annotations.extend([item for item in result.items if isinstance(item, StreamingAnnotationContent)])
                        files.extend([item for item in result.items if isinstance(item, StreamingFileReferenceContent)])
                        if isinstance(result.message, StreamingChatMessageContent):
                            responseContent += result.message.content
                        else:
                            print(f"{result}")

                        # Check for code in metadata
                        if hasattr(result, 'metadata') and result.metadata and result.metadata.get("code") is True:
                            if isinstance(result.message, StreamingChatMessageContent) and result.message.content:
                                code_output_content += result.message.content

                        thread = response.thread
                    
                    # Extract annotations from the ChatMessageContent response
                    for item in annotations:
                        source = Source(
                            quote=item.quote if hasattr(item, 'quote') else '',
                            title=item.title if hasattr(item, 'title') else '',
                            url=item.url if hasattr(item, 'url') else '',
                            start_index=item.start_index if hasattr(item, 'start_index') else '',
                            end_index=item.end_index if hasattr(item, 'end_index') else ''
                        )
                        sources.append(source)

                    for item in files:
                        fr = FileReference(id=item.file_id if hasattr(item, 'file_id') else '')
                        file_references.append(fr)
                
                finally:
                    print("Completed agent invocation")  

            request_result = RequestResult(
                content=responseContent,
                sources=sources,
                files=file_references,
                intermediate_steps=intermediate_steps,
                thread_id=thread.id,
                code_content=code_output_content.strip() # Add code_output_content to RequestResult
            )

            return request_result