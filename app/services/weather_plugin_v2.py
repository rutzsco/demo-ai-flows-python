import asyncio
import aiohttp
from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function


class WeatherPluginV2:
  
    @kernel_function(name="get_weather_for_latitude_longitude", description="get the weather for a latitude and longitude GeoPoint")
    async def get_weather_for_latitude_longitude(self, latitude: Annotated[str, "The location GeoPoint latitude"], longitude: Annotated[str, "The location GeoPoint longitude"]) -> Annotated[str, "The output is a string"]:
        url = f"https://api.weather.gov/points/{latitude},{longitude}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": "app"}) as response:
                response.raise_for_status()
                response_body = await response.text()

            json_response = await response.json()
            forecast_url = json_response["properties"]["forecast"]

            async with session.get(forecast_url) as forecast_response:
                forecast_response.raise_for_status()
                forecast_response_body = await forecast_response.text()
                #arguments["WeatherForcast"] = forecast_response_body
                return forecast_response_body

    async def determine_lat_long_async(self, location: Annotated[str, "The location GeoPoint latitude"]) -> Annotated[str, "The output is a string"]:


        chat_history = ChatHistory("WeatherLatLongSystemPrompt")  # Replace with actual prompt fetching logic
        chat_history.add_user_message(weather_location)

        search_answer = await chat_gpt.get_chat_message_content_async(chat_history, None, kernel)

        parts = [part.strip() for part in search_answer.split(',')]
        lp = LocationPoint(Latitude=float(parts[0]), Longitude=float(parts[1]))
        arguments["LocationPoint"] = lp

        return lp