import pytest
from src.services.llm_manager import LLMManager
from src.services.weather_service import WeatherService
from datetime import datetime

def test_llm_manager_initialization():
    """Test LLM manager initialization"""
    llm_manager = LLMManager()
    assert llm_manager is not None
    assert llm_manager.get_llm() is not None

def test_weather_service_initialization():
    """Test weather service initialization"""
    weather_service = WeatherService()
    assert weather_service is not None
    assert weather_service.api_key is not None

def test_weather_service_get_historical_weather():
    """Test getting historical weather data"""
    weather_service = WeatherService()
    test_date = datetime.now()
    test_lat = 40.7128
    test_lon = -74.0060
    
    with pytest.raises(Exception):  # Should raise exception without valid API key
        weather_service.get_historical_weather(test_lat, test_lon, test_date) 