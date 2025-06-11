import streamlit as st
from services.llm_manager import LLMManager
from services.weather_service import WeatherService
from services.geocoding_service import GeocodingService
from services.vector_db import OutageVectorDB
from models.state import OutageAnalysisState
from ui.streamlit_components import (
    display_validation_results,
    display_map_visualization,
    display_chat_interface
)
from reports.generator import generate_and_download_report

def initialize_services():
    """Initialize all required services"""
    return {
        'llm': LLMManager(),
        'weather': WeatherService(),
        'geocoding': GeocodingService(),
        'vector_db': OutageVectorDB()
    }

def initialize_state():
    """Initialize application state"""
    if 'analysis_state' not in st.session_state:
        st.session_state.analysis_state = OutageAnalysisState(
            dataset_loaded=False,
            validation_complete=False,
            raw_dataset_summary={},
            validation_results={},
            filtered_summary={},
            current_window_analysis=None,
            chat_context={},
            errors=[]
        )

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Power Outage Analysis",
        page_icon="⚡",
        layout="wide"
    )
    
    initialize_state()
    services = initialize_services()
    
    st.title("Power Outage Analysis Dashboard")
    
    # Main navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Validation Results", "Map Visualization", "Chat Interface"]
    )
    
    if page == "Validation Results":
        display_validation_results()
    elif page == "Map Visualization":
        display_map_visualization()
    else:
        display_chat_interface()

if __name__ == "__main__":
    main() 