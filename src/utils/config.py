import os
from typing import Dict, Any
from dotenv import load_dotenv

class Config:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        
        # Application Settings
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Database Settings
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
        
        # Cache Settings
        self.cache_expiry = int(os.getenv("CACHE_EXPIRY", "3600"))
        
        # Validation Settings
        self.validation_delay = float(os.getenv("VALIDATION_DELAY", "0.5"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        
        # LLM Settings
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        
        # Weather Settings
        self.weather_api_url = os.getenv("WEATHER_API_URL", "https://api.weatherapi.com/v1")
        
        # Validate required settings
        self._validate_settings()
    
    def _validate_settings(self):
        """Validate required settings are present"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.weather_api_key:
            raise ValueError("WEATHER_API_KEY is required")
    
    def get_settings(self) -> Dict[str, Any]:
        """Get all settings as a dictionary"""
        return {
            "debug": self.debug,
            "log_level": self.log_level,
            "vector_db_path": self.vector_db_path,
            "cache_expiry": self.cache_expiry,
            "validation_delay": self.validation_delay,
            "max_retries": self.max_retries,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "weather_api_url": self.weather_api_url
        }

# Create global config instance
config = Config() 