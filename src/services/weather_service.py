from typing import Dict, List
from datetime import datetime
import requests
from functools import lru_cache
import os

class WeatherService:
    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.base_url = "https://api.weatherapi.com/v1"

    def get_historical_weather(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get historical weather data for a specific location and date"""
        endpoint = f"{self.base_url}/history.json"
        params = {
            "key": self.api_key,
            "q": f"{lat},{lon}",
            "dt": date.strftime("%Y-%m-%d"),
            "hour": date.hour
        }
        
        response = requests.get(endpoint, params=params)
        if response.status_code != 200:
            raise Exception(f"Weather API error: {response.text}")
            
        return response.json()

    def _safe_get(self, data_list: List, index: int):
        """Safely get an item from a list with bounds checking"""
        try:
            return data_list[index]
        except (IndexError, TypeError):
            return None 