"""
Configuration management for the Power Agent application.
Centralizes all configuration, environment variables, and settings.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Settings:
    """Application configuration management"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.cache_dir = self.project_root / "cache"
        self.data_dir = self.project_root / "data"
        
        # Ensure directories exist
        self.cache_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Load prompts configuration
        self.prompts = self._load_prompts()
        
        # Configure logging
        self._setup_logging()
    
    # ==================== API Keys and External Services ====================
    
    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Claude API key from environment"""
        return os.getenv('ANTHROPIC_API_KEY')
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI API key from environment"""
        return os.getenv('OPENAI_API_KEY')
    
    @property
    def weather_api_url(self) -> str:
        """Open-Meteo API base URL"""
        return "https://archive-api.open-meteo.com/v1/archive"
    
    # ==================== LLM Configuration ====================
    
    @property
    def default_llm_provider(self) -> str:
        """Default LLM provider (claude or openai)"""
        return os.getenv('DEFAULT_LLM_PROVIDER', 'claude')
    
    @property
    def claude_model(self) -> str:
        """Claude model name"""
        return os.getenv('CLAUDE_MODEL', 'claude-3-sonnet-20240229')
    
    @property
    def openai_model(self) -> str:
        """OpenAI model name"""
        return os.getenv('OPENAI_MODEL', 'gpt-4')
    
    @property
    def llm_temperature(self) -> float:
        """LLM temperature setting"""
        return float(os.getenv('LLM_TEMPERATURE', '0.0'))
    
    @property
    def llm_max_tokens(self) -> int:
        """Maximum tokens for LLM responses"""
        return int(os.getenv('LLM_MAX_TOKENS', '4000'))
    
    # ==================== Vector Database Configuration ====================
    
    @property
    def vector_db_path(self) -> str:
        """ChromaDB persistence directory"""
        return str(self.cache_dir / "chroma_db")
    
    @property
    def vector_collection_name(self) -> str:
        """Default vector collection name"""
        return os.getenv('VECTOR_COLLECTION_NAME', 'outage_reports')
    
    # ==================== Cache Configuration ====================
    
    @property
    def geocoding_cache_file(self) -> str:
        """Geocoding cache file path"""
        return str(self.cache_dir / "geocoding_cache.joblib")
    
    @property
    def weather_cache_file(self) -> str:
        """Weather cache SQLite file path"""
        return str(self.cache_dir / "weather_cache.sqlite")
    
    @property
    def outage_summary_cache_file(self) -> str:
        """Outage summary cache file path"""
        return str(self.cache_dir / "outage_data_summary.joblib")
    
    @property
    def validation_results_cache_file(self) -> str:
        """Validation results cache file path"""
        return str(self.cache_dir / "validation_results.joblib")
    
    @property
    def vector_db_status_file(self) -> str:
        """Vector DB status file path"""
        return str(self.cache_dir / "vector_db_status.json")
    
    @property
    def cache_expiry_hours(self) -> int:
        """Cache expiry time in hours"""
        return int(os.getenv('CACHE_EXPIRY_HOURS', '24'))
    
    # ==================== Data Configuration ====================
    
    @property
    def default_data_file(self) -> str:
        """Default outage data CSV file"""
        return str(self.data_dir / "raw_data.csv")
    
    @property
    def required_columns(self) -> List[str]:
        """Required columns in outage data CSV"""
        return ['DATETIME', 'LATITUDE', 'LONGITUDE', 'CUSTOMERS']
    
    # ==================== UI Configuration ====================
    
    @property
    def default_ui_framework(self) -> str:
        """Default UI framework (streamlit, fastapi, etc.)"""
        return os.getenv('DEFAULT_UI_FRAMEWORK', 'streamlit')
    
    @property
    def streamlit_config(self) -> Dict[str, Any]:
        """Streamlit-specific configuration"""
        return {
            'page_title': 'LLM-Powered Outage Analysis Agent',
            'page_icon': '⚡',
            'layout': 'wide',
            'initial_sidebar_state': 'expanded',
            'watcher_warning_threshold': int(os.getenv('STREAMLIT_WATCHER_WARNING_THRESHOLD', '300')),
            'file_watcher_type': os.getenv('STREAMLIT_SERVER_FILE_WATCHER_TYPE', 'none')
        }
    
    # ==================== Logging Configuration ====================
    
    @property
    def log_level(self) -> str:
        """Logging level"""
        return os.getenv('LOG_LEVEL', 'INFO')
    
    @property
    def log_format(self) -> str:
        """Logging format"""
        return '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    def _setup_logging(self) -> None:
        """Configure application logging"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=self.log_format
        )
        
        # Configure Streamlit watcher settings
        os.environ["STREAMLIT_WATCHER_WARNING_THRESHOLD"] = str(self.streamlit_config['watcher_warning_threshold'])
        os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = self.streamlit_config['file_watcher_type']
    
    # ==================== Prompt Management ====================
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from JSON file"""
        prompts_file = self.project_root / "prompts.json"
        if prompts_file.exists():
            try:
                with open(prompts_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load prompts.json: {e}")
                return self._get_default_prompts()
        else:
            logging.warning("prompts.json not found, using default prompts")
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict[str, str]:
        """Default prompt templates if JSON file is not available"""
        return {
            "system_prompt": "You are an expert power outage analyst...",
            "validation_prompt": "Analyze this outage report for legitimacy...",
            "chat_prompt": "Answer questions about the outage analysis results...",
            "comprehensive_report_prompt": "Generate a comprehensive analysis report...",
            "exhaustive_report_prompt": "Create an exhaustive technical report...",
            "time_window_prompt": "Analyze outages in this specific time window..."
        }
    
    def get_prompt(self, prompt_name: str) -> str:
        """Get a specific prompt template"""
        return self.prompts.get(prompt_name, f"Prompt '{prompt_name}' not found")
    
    def reload_prompts(self) -> None:
        """Reload prompts from file"""
        self.prompts = self._load_prompts()
    
    # ==================== Weather Service Configuration ====================
    
    @property
    def weather_cache_duration_hours(self) -> int:
        """Weather data cache duration in hours"""
        return int(os.getenv('WEATHER_CACHE_DURATION_HOURS', '24'))
    
    @property
    def weather_severe_thresholds(self) -> Dict[str, float]:
        """Thresholds for severe weather conditions"""
        return {
            'wind_speed_kmh': float(os.getenv('SEVERE_WIND_SPEED', '50.0')),
            'precipitation_mm': float(os.getenv('SEVERE_PRECIPITATION', '10.0')),
            'temperature_low_c': float(os.getenv('SEVERE_TEMP_LOW', '-10.0')),
            'temperature_high_c': float(os.getenv('SEVERE_TEMP_HIGH', '35.0'))
        }
    
    # ==================== Geocoding Configuration ====================
    
    @property
    def geocoding_user_agent(self) -> str:
        """User agent for geocoding requests"""
        return os.getenv('GEOCODING_USER_AGENT', 'power-outage-analysis')
    
    @property
    def geocoding_timeout_seconds(self) -> int:
        """Timeout for geocoding requests"""
        return int(os.getenv('GEOCODING_TIMEOUT', '10'))
    
    # ==================== Validation Configuration ====================
    
    @property
    def validation_confidence_threshold(self) -> float:
        """Minimum confidence threshold for validation"""
        return float(os.getenv('VALIDATION_CONFIDENCE_THRESHOLD', '0.7'))
    
    @property
    def validation_batch_size(self) -> int:
        """Batch size for validation processing"""
        return int(os.getenv('VALIDATION_BATCH_SIZE', '10'))
    
    # ==================== Report Configuration ====================
    
    @property
    def report_output_dir(self) -> str:
        """Directory for generated reports"""
        report_dir = self.project_root / "reports"
        report_dir.mkdir(exist_ok=True)
        return str(report_dir)
    
    @property
    def report_include_maps(self) -> bool:
        """Include maps in generated reports"""
        return os.getenv('REPORT_INCLUDE_MAPS', 'true').lower() == 'true'
    
    # ==================== Development Configuration ====================
    
    @property
    def debug_mode(self) -> bool:
        """Enable debug mode"""
        return os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    @property
    def enable_mcp(self) -> bool:
        """Enable Model Context Protocol features"""
        return os.getenv('ENABLE_MCP', 'true').lower() == 'true'
    
    # ==================== Utility Methods ====================
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate required environment configuration"""
        checks = {
            'anthropic_api_key': bool(self.anthropic_api_key),
            'openai_api_key': bool(self.openai_api_key),
            'cache_dir_exists': self.cache_dir.exists(),
            'data_dir_exists': self.data_dir.exists(),
            'prompts_loaded': bool(self.prompts),
            'default_data_file_exists': Path(self.default_data_file).exists()
        }
        return checks
    
    def get_all_cache_files(self) -> List[str]:
        """Get list of all cache file paths"""
        return [
            self.geocoding_cache_file,
            self.weather_cache_file,
            self.outage_summary_cache_file,
            self.validation_results_cache_file,
            self.vector_db_status_file
        ]
    
    def clear_all_caches(self) -> None:
        """Clear all cache files"""
        for cache_file in self.get_all_cache_files():
            cache_path = Path(cache_file)
            if cache_path.exists():
                cache_path.unlink()
                logging.info(f"Cleared cache file: {cache_file}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all settings as dictionary"""
        return {
            'api_keys': {
                'anthropic_api_key': '***' if self.anthropic_api_key else None,
                'openai_api_key': '***' if self.openai_api_key else None,
            },
            'llm': {
                'default_provider': self.default_llm_provider,
                'claude_model': self.claude_model,
                'openai_model': self.openai_model,
                'temperature': self.llm_temperature,
                'max_tokens': self.llm_max_tokens
            },
            'paths': {
                'project_root': str(self.project_root),
                'cache_dir': str(self.cache_dir),
                'data_dir': str(self.data_dir),
                'vector_db_path': self.vector_db_path,
                'report_output_dir': self.report_output_dir
            },
            'cache': {
                'expiry_hours': self.cache_expiry_hours,
                'weather_cache_duration_hours': self.weather_cache_duration_hours
            },
            'validation': {
                'confidence_threshold': self.validation_confidence_threshold,
                'batch_size': self.validation_batch_size
            },
            'ui': {
                'default_framework': self.default_ui_framework,
                'streamlit_config': self.streamlit_config
            },
            'features': {
                'debug_mode': self.debug_mode,
                'enable_mcp': self.enable_mcp,
                'report_include_maps': self.report_include_maps
            }
        }


# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings

def reload_settings() -> Settings:
    """Reload settings and return new instance"""
    global settings
    settings = Settings()
    return settings