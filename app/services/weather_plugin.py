import requests
import json
import datetime
from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from dataclasses import dataclass
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions.kernel_arguments import KernelArguments
from app.models.api_models import ExecutionStep

@dataclass
class LocationPoint:
    Latitude: float
    Longitude: float

class WeatherPlugin:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
  
    @kernel_function(name="get_weather_for_latitude_longitude", description="get the weather for a latitude and longitude GeoPoint")
    async def get_weather_for_latitude_longitude(self, arguments: Annotated[KernelArguments, {"include_in_function_choices": False}], latitude: Annotated[str, "The location GeoPoint latitude"], longitude: Annotated[str, "The location GeoPoint longitude"]) -> Annotated[str, "The output is a string"]:
        start_time = datetime.datetime.now().isoformat()
        url = f"https://api.weather.gov/points/{latitude},{longitude}"
        headers = {"User-Agent": "app"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        json_response = response.json()
        forecast_url = json_response["properties"]["forecast"]

        forecast_response = requests.get(forecast_url, headers=headers)
        forecast_response.raise_for_status()
        forecast_response_body = forecast_response.text

        end_time = datetime.datetime.now().isoformat()
        # Add the diagnostic result to the arguments
        diagnostic_result = ExecutionStep(name="get_weather_for_latitude_longitude", content=forecast_response_body, start_time=start_time, end_time=end_time)
        arguments["diagnostics"].append(diagnostic_result)

        return forecast_response_body
            
    @kernel_function(name="get_lat_long", description="Get a latitude and longitude GeoPoint for the provided city or postal code.")
    async def determine_lat_long_async(self, arguments: Annotated[KernelArguments, {"include_in_function_choices": False}], location: Annotated[str, "A location string as a city and state or postal code"]) -> Annotated[LocationPoint, "The location GeoPoint"]:
        start_time = datetime.datetime.now().isoformat()
        # Use the LLM get the latitude and longitude
        result = await self.kernel.invoke_prompt(f"What is the geopoint for: {location}. Return the result as a JSON object with Latitude and Longitude properties: {{\"Latitude\": 0.0, \"Longitude\": 0.0}}. Only return the JSON.", max_tokens=100)
        
        # Parse the result to extract the JSON object
        json_result  = f"{result}".strip("'")
        json_data = json.loads(json_result)
        location = LocationPoint(
            Latitude=json_data["Latitude"],
            Longitude=json_data["Longitude"]
        )

        end_time = datetime.datetime.now().isoformat()
        # Add the diagnostic result to the arguments
        diagnostic_result = ExecutionStep(name="get_lat_long", content=json_data, start_time=start_time, end_time=end_time)
        arguments["diagnostics"].append(diagnostic_result)

        return location