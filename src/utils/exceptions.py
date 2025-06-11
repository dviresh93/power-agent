class PowerAgentError(Exception):
    """Base exception for Power Agent application"""
    pass

class ValidationError(PowerAgentError):
    """Raised when validation fails"""
    pass

class WeatherAPIError(PowerAgentError):
    """Raised when weather API calls fail"""
    pass

class GeocodingError(PowerAgentError):
    """Raised when geocoding operations fail"""
    pass

class VectorDBError(PowerAgentError):
    """Raised when vector database operations fail"""
    pass

class LLMError(PowerAgentError):
    """Raised when LLM operations fail"""
    pass

class ConfigurationError(PowerAgentError):
    """Raised when configuration is invalid"""
    pass

class CacheError(PowerAgentError):
    """Raised when cache operations fail"""
    pass

class ReportGenerationError(PowerAgentError):
    """Raised when report generation fails"""
    pass 