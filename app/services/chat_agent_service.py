import os, time
from typing import List
from dotenv import load_dotenv
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from app.models.api_models import ChatThreadRequest, ExecutionDiagnostics, RequestResult, Source, FileReference
from app.prompts.file_service import FileService


from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import AnnotationContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent, StreamingChatMessageContent, StreamingAnnotationContent, StreamingFileReferenceContent
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeInterpreterTool
from azure.ai.projects.models import FilePurpose
from azure.ai.projects.models import MessageRole, BingGroundingTool
from azure.identity import DefaultAzureCredential
from pathlib import Path
import json
import uuid  # Import the uuid module

class ChatAgentService:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        self.agent_id = os.getenv("AZURE_AI_AGENT_ID")

        pass


    async def run_chat_sk(self, request: ChatThreadRequest) -> str:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("Agent: Chat") as current_span:
            # Validate the request object
            if not request.message:
                raise ValueError("No messages found in request.")
            user_message = request.message

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

                # Create a thread for the agent. If no thread is provided, a new thread will be created and returned with the initial response
                thread: AzureAIAgentThread  = None
                if request.thread_id:
                    thread = AzureAIAgentThread(client=client, thread_id=request.thread_id) 

                annotations: list[StreamingAnnotationContent] = []
                files: list[StreamingFileReferenceContent] = []
                sources = []
                file_references = []
                responseContent = ''
                try:
                    # 4. Invoke the agent with the specified message for response
                    async for result in agent.invoke_stream(messages=user_message, thread=thread, on_intermediate_message=handle_intermediate_steps):
                        response = result
                        annotations.extend([item for item in result.items if isinstance(item, StreamingAnnotationContent)])
                        files.extend([item for item in result.items if isinstance(item, StreamingFileReferenceContent)])
                        if isinstance(result.message, StreamingChatMessageContent):
                            responseContent += result.message.content
                        else:
                            print(f"{result}")

                        thread = response.thread
                    
                    # Extract annotations from the ChatMessageContent response
                    for item in annotations:
                        source = Source(
                            quote=item.quote if hasattr(item, 'quote') else '',
                            title=item.title if hasattr(item, 'title') else '',
                            url=item.url if hasattr(item, 'url') else ''
                        )
                        sources.append(source)

                    for item in files:
                        fr = FileReference(
                            id=item.file_id if hasattr(item, 'file_id') else ''
                        )
                        file_references.append(fr)
                
                finally:
                    print("Completed agent invocation")  

            request_result = RequestResult(
                content=responseContent,
                sources=sources,
                files=file_references,
                intermediate_steps=intermediate_steps,
                thread_id=thread.id
            )

            return request_result
    
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
                        model="gpt-4",
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
                    from azure.storage.blob import BlobServiceClient
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
                    
                    print("Done. You can now access the file from the following URL:")
                    print(file_url)   
                   
                    return(f"{last_msg.text.value} \nA copy in [cloud]({file_url})")
                    
                except Exception as e:
                    print(f"Error: {e}")
                    return f"Error: {e}"
                finally:
                    project_client.agents.delete_thread(thread.id)
                    project_client.agents.delete_agent(agent.id)
                    