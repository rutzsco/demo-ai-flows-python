from app.services.weather_plugin import WeatherPlugin
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from dotenv import load_dotenv
import os
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from opentelemetry import trace
from app.models.api_models import ChatRequest, ExecutionStep, ExecutionDiagnostics, RequestResult
from typing import List

class SemanticKernelService:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Retrieve configuration from environment variables
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        
        if not api_key or not endpoint or not deployment_name:
            raise ValueError("Missing required environment variables for OpenAI configuration.")
    
        self.kernel = sk.Kernel()
        self.kernel.add_service(AzureChatCompletion(
            api_key=api_key,
            endpoint=endpoint,
            deployment_name=deployment_name
        ))
        self.kernel.add_plugin(WeatherPlugin(self.kernel), plugin_name="weather")
        pass

    async def run_workflow(self, data: str) -> str:
        result = await self.kernel.invoke_prompt(data)
        return f"Processed with Semantic Kernel: {result}"
    
    async def run_weather(self, request: ChatRequest) -> str:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("Agent: Weather") as current_span:
            kernel_arguments = KernelArguments(
            settings=PromptExecutionSettings(
                function_choice_behavior=FunctionChoiceBehavior.Auto(filters={"included_plugins": ["weather"]}),
                )
            )

            kernel_arguments ["diagnostics"] = []
            if not request.messages:
                raise ValueError("No messages found in request.")
            data = request.messages[-1].content
            result = await self.kernel.invoke_prompt(data, arguments=kernel_arguments)

            request_result = RequestResult(
                content=f"{result}",
                execution_diagnostics=ExecutionDiagnostics(steps=kernel_arguments ["diagnostics"]))

            return request_result

