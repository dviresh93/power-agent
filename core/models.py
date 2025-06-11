"""
Core data models for the Power Agent application.
These models define the structure of data flowing through the system.
"""

from typing import Dict, List, Optional, Tuple, TypedDict, Any
from datetime import datetime
import pandas as pd

try:
    from pydantic import BaseModel, Field
except ImportError:
    try:
        from pydantic.v1 import BaseModel, Field
    except ImportError:
        class BaseModel:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        def Field(*args, **kwargs):
            return None


class OutageAnalysisState(TypedDict):
    """Core state management for outage analysis"""
    dataset_loaded: bool
    validation_complete: bool
    raw_dataset_summary: Dict
    validation_results: Dict
    filtered_summary: Dict
    current_window_analysis: Optional[Dict]
    chat_context: Dict


class OutageRecord(BaseModel):
    """Individual outage record"""
    datetime: datetime
    latitude: float
    longitude: float
    customers: int
    location_name: Optional[str] = None
    validation_status: Optional[str] = None
    weather_data: Optional[Dict] = None


class ValidationResult(BaseModel):
    """Result of outage validation analysis"""
    record_id: str
    is_valid: bool
    confidence_score: float
    weather_correlation: Dict
    llm_analysis: str
    classification: str  # 'real_outage' or 'false_positive'


class WeatherData(BaseModel):
    """Weather information for a specific location and time"""
    datetime: datetime
    latitude: float
    longitude: float
    temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    precipitation: Optional[float] = None
    weather_code: Optional[int] = None
    conditions: Optional[str] = None


class ReportData(BaseModel):
    """Data structure for report generation"""
    title: str
    summary: Dict
    validation_results: List[ValidationResult]
    raw_data_summary: Dict
    filtered_data_summary: Dict
    analysis_window: Optional[Dict] = None
    generated_at: datetime


class LocationInfo(BaseModel):
    """Geographic location information"""
    latitude: float
    longitude: float
    city: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None


class AnalysisRequest(BaseModel):
    """Request structure for analysis operations"""
    operation: str  # 'validate', 'report', 'chat', 'window_analysis'
    parameters: Dict[str, Any]
    context: Optional[Dict] = None


class UIState(BaseModel):
    """Abstract UI state - framework independent"""
    current_view: str
    data_loaded: bool
    analysis_complete: bool
    selected_records: List[str] = []
    filters: Dict[str, Any] = {}
    map_center: Optional[Tuple[float, float]] = None
    map_zoom: int = 10