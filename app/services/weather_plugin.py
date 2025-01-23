import requests
from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from dataclasses import dataclass
from semantic_kernel.kernel import Kernel

@dataclass
class LocationPoint:
    Latitude: float
    Longitude: float

class WeatherPlugin:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
  
    @kernel_function(name="get_weather_for_latitude_longitude", description="get the weather for a latitude and longitude GeoPoint")
    async def get_weather_for_latitude_longitude(self, latitude: Annotated[str, "The location GeoPoint latitude"], longitude: Annotated[str, "The location GeoPoint longitude"]) -> Annotated[str, "The output is a string"]:
        url = f"https://api.weather.gov/points/{latitude},{longitude}"
        headers = {"User-Agent": "app"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        json_response = response.json()
        forecast_url = json_response["properties"]["forecast"]

        forecast_response = requests.get(forecast_url, headers=headers)
        forecast_response.raise_for_status()
        forecast_response_body = forecast_response.text

        return forecast_response_body
            
    #@kernel_function(name="determine_lat_long", description="Get a latitude and longitude GeoPoint for the provided city or postal code.")
    async def determine_lat_long_async(self, location: Annotated[str, "A location string as a city and state or postal code"]) -> Annotated[LocationPoint, "The location GeoPoint"]:
        search_answer = await self.kernel.invoke_prompt(f"What is the geopoint for: {location}")
        search_answer = search_answer.strip('"')  # Remove any surrounding quotes
        parts = [part.strip() for part in search_answer.split(',')]
        lp = LocationPoint(Latitude=float(parts[0]), Longitude=float(parts[1]))
        return lp