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

            async with (DefaultAzureCredential() as creds, AzureAIAgent.create_client(credential=creds) as client,):
                
                # 1. Create an agent on the Azure AI agent service
                agent_definition = await client.agents.get_agent(agent_id=self.agent_id)
                # 2. Create a Semantic Kernel agent for the Azure AI agent
                agent = AzureAIAgent(
                    client=client,
                    definition=agent_definition,
                )

                # 3. Create a thread for the agent
                # If no thread is provided, a new thread will be
                # created and returned with the initial response
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
                    # 5. Cleanup: Delete the thread and agent
                    #await thread.delete() if thread else None
                    #await client.agents.delete_agent(agent.id) if agent else None   

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
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("Agent: Chat") as current_span:
            with project_client:

                user_message = request.messages[-1].content
                agent = project_client.agents.get_agent(agent_id='asst_g8KPAp9JzA8LpqJOIiwIisL8')
                print(f"Created agent, agent ID: {agent.id}")
                thread = project_client.agents.create_thread()
                print(f"Created thread, thread ID: {thread.id}")
                message = project_client.agents.create_message(thread_id=thread.id, role="user", content="Hello, tell me a hilarious joke")
                print(f"Created message, message ID: {message.id}")

                run = project_client.agents.create_run(thread_id=thread.id, agent_id=agent.id)

                # Poll the run as long as run status is queued or in progress
                while run.status in ["queued", "in_progress", "requires_action"]:
                    # Wait for a second
                    time.sleep(1)
                    run = project_client.agents.get_run(thread_id=thread.id, run_id=run.id)

                    print(f"Run status: {run.status}")

                project_client.agents.delete_agent(agent.id)
                print("Deleted agent")

                messages = project_client.agents.list_messages(thread_id=thread.id)
                print(f"messages: {messages}")