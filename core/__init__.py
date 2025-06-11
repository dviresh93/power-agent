"""
Core module for Power Agent application.

This module contains the core business logic and data models.
"""

from .models import (
    OutageAnalysisState,
    OutageRecord,
    ValidationResult,
    WeatherData,
    ReportData,
    LocationInfo,
    AnalysisRequest,
    UIState
)

from .validation_engine import (
    ValidationEngine,
    validate_outage_report,
    generate_comprehensive_report,
    generate_exhaustive_report,
    chat_about_results,
    generate_map_data_summary,
    generate_map_section_for_report
)

__all__ = [
    # Data models
    'OutageAnalysisState',
    'OutageRecord', 
    'ValidationResult',
    'WeatherData',
    'ReportData',
    'LocationInfo',
    'AnalysisRequest',
    'UIState',
    
    # Validation engine
    'ValidationEngine',
    'validate_outage_report',
    'generate_comprehensive_report',
    'generate_exhaustive_report',
    'chat_about_results',
    'generate_map_data_summary',
    'generate_map_section_for_report'
]