"""
Weather Service Module
Handles weather data retrieval and validation using Open-Meteo API with caching.
Independent of UI frameworks for clean modular architecture.
"""

import logging
from datetime import datetime
from typing import Dict, List
import os

# Set up logging
logger = logging.getLogger(__name__)

class WeatherService:
    """Weather service for validation using Open-Meteo API"""
    
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Set up caching for weather data requests
        try:
            import requests_cache
            self.session = requests_cache.CachedSession(
                'weather_cache', 
                expire_after=86400,  # Cache for 24 hours
                allowable_methods=('GET', 'POST')
            )
            logger.info("✅ Weather API caching enabled")
        except ImportError:
            import requests
            self.session = requests
            logger.warning("⚠️ requests_cache not installed. Weather API calls will not be cached.")
        
    def get_historical_weather(self, lat: float, lon: float, date: datetime) -> Dict:
        """
        Fetch weather data for validation
        
        Args:
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate  
            date (datetime): Date and time for weather data
            
        Returns:
            Dict: Weather data including temperature, precipitation, wind data, etc.
                 Returns error information if API call fails.
        """
        try:
            date_str = date.strftime("%Y-%m-%d")
            
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": date_str,
                "end_date": date_str,
                "hourly": "temperature_2m,precipitation,windspeed_10m,windgusts_10m,snowfall",
                "timezone": "America/Chicago"
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            hourly_data = data.get('hourly', {})
            
            # Get data for the specific hour
            target_hour = min(date.hour, len(hourly_data.get('temperature_2m', [])) - 1) if hourly_data.get('temperature_2m') else 0
            
            return {
                'timestamp': date.isoformat(),
                'coordinates': {'lat': lat, 'lon': lon},
                'temperature': self._safe_get(hourly_data.get('temperature_2m', []), target_hour),
                'precipitation': self._safe_get(hourly_data.get('precipitation', []), target_hour),
                'wind_speed': self._safe_get(hourly_data.get('windspeed_10m', []), target_hour),
                'wind_gusts': self._safe_get(hourly_data.get('windgusts_10m', []), target_hour),
                'snowfall': self._safe_get(hourly_data.get('snowfall', []), target_hour),
                'api_status': 'success'
            }
            
        except Exception as e:
            # Import requests here to handle both cached and non-cached sessions
            try:
                import requests
                if isinstance(e, requests.exceptions.RequestException):
                    logger.error(f"⚠️  Weather API request failed: {str(e)}")
                    return {'error': f'Weather API error: {str(e)}', 'timestamp': date.isoformat(), 'api_status': 'failed'}
            except ImportError:
                pass
                
            logger.error(f"⚠️  Weather service error: {str(e)}")
            return {'error': f'Weather service error: {str(e)}', 'timestamp': date.isoformat(), 'api_status': 'failed'}
    
    def _safe_get(self, data_list: List, index: int):
        """
        Safely get data from list with bounds checking
        
        Args:
            data_list (List): List to extract data from
            index (int): Index to access
            
        Returns:
            The value at the index, or None if index is out of bounds or list is empty
        """
        if not data_list or index >= len(data_list) or index < 0:
            return None
        return data_list[index]

    def validate_weather_data(self, weather_data: Dict) -> bool:
        """
        Validate if weather data contains required fields and is properly formatted
        
        Args:
            weather_data (Dict): Weather data dictionary to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if not isinstance(weather_data, dict):
            return False
            
        if weather_data.get('api_status') == 'failed':
            return False
            
        required_fields = ['timestamp', 'coordinates']
        for field in required_fields:
            if field not in weather_data:
                return False
                
        # Check if coordinates are properly formatted
        coords = weather_data.get('coordinates', {})
        if not isinstance(coords, dict) or 'lat' not in coords or 'lon' not in coords:
            return False
            
        return True

    def format_weather_summary(self, weather_data: Dict) -> str:
        """
        Format weather data into human-readable summary
        
        Args:
            weather_data (Dict): Weather data to format
            
        Returns:
            str: Formatted weather summary string
        """
        if not self.validate_weather_data(weather_data):
            return "Invalid weather data"
            
        if weather_data.get('api_status') == 'failed':
            return f"Weather data unavailable: {weather_data.get('error', 'Unknown error')}"
            
        parts = []
        
        if weather_data.get('temperature') is not None:
            parts.append(f"Temperature: {weather_data['temperature']:.1f}°C")
            
        if weather_data.get('wind_speed') is not None:
            parts.append(f"Wind: {weather_data['wind_speed']:.1f} km/h")
            
        if weather_data.get('wind_gusts') is not None and weather_data['wind_gusts'] > 0:
            parts.append(f"Gusts: {weather_data['wind_gusts']:.1f} km/h")
            
        if weather_data.get('precipitation') is not None and weather_data['precipitation'] > 0:
            parts.append(f"Precipitation: {weather_data['precipitation']:.1f} mm")
            
        if weather_data.get('snowfall') is not None and weather_data['snowfall'] > 0:
            parts.append(f"Snowfall: {weather_data['snowfall']:.1f} mm")
            
        return ", ".join(parts) if parts else "No weather data available"

    def is_severe_weather(self, weather_data: Dict, wind_threshold: float = 25.0, 
                         gust_threshold: float = 35.0, precip_threshold: float = 5.0) -> bool:
        """
        Determine if weather conditions are severe enough to cause power outages
        
        Args:
            weather_data (Dict): Weather data to analyze
            wind_threshold (float): Wind speed threshold in km/h for severe conditions
            gust_threshold (float): Wind gust threshold in km/h for severe conditions  
            precip_threshold (float): Precipitation threshold in mm for severe conditions
            
        Returns:
            bool: True if conditions are severe, False otherwise
        """
        if not self.validate_weather_data(weather_data) or weather_data.get('api_status') == 'failed':
            return False
            
        # Check wind speed
        wind_speed = weather_data.get('wind_speed', 0) or 0
        if wind_speed >= wind_threshold:
            return True
            
        # Check wind gusts
        wind_gusts = weather_data.get('wind_gusts', 0) or 0
        if wind_gusts >= gust_threshold:
            return True
            
        # Check precipitation
        precipitation = weather_data.get('precipitation', 0) or 0
        if precipitation >= precip_threshold:
            return True
            
        # Check for snowfall (any significant snowfall can cause issues)
        snowfall = weather_data.get('snowfall', 0) or 0
        if snowfall >= 2.0:  # 2mm or more of snowfall
            return True
            
        return False