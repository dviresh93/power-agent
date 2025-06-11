"""
Streamlit adapter implementing the abstract UI interface.
This allows the core application to work with Streamlit through a clean interface.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, date, time
import contextlib

from interfaces.ui_interface import AbstractUI


class StreamlitAdapter(AbstractUI):
    """Streamlit implementation of the abstract UI interface"""
    
    def __init__(self):
        super().__init__()
        self._setup_streamlit_config()
    
    def _setup_streamlit_config(self):
        """Configure Streamlit page settings"""
        try:
            st.set_page_config(
                page_title="LLM-Powered Outage Analysis Agent",
                page_icon="⚡",
                layout="wide",
                initial_sidebar_state="expanded"
            )
        except:
            # Page config already set
            pass
    
    # ==================== Core UI Components ====================
    
    def display_title(self, title: str) -> None:
        st.title(title)
    
    def display_text(self, text: str, style: str = "normal") -> None:
        if style == "header":
            st.header(text)
        elif style == "subheader":
            st.subheader(text)
        elif style == "error":
            st.error(text)
        elif style == "success":
            st.success(text)
        elif style == "warning":
            st.warning(text)
        elif style == "info":
            st.info(text)
        else:
            st.write(text)
    
    def display_metric(self, label: str, value: Any, delta: Optional[str] = None) -> None:
        st.metric(label, value, delta)
    
    def display_progress(self, progress: float, text: str = "") -> None:
        st.progress(progress, text)
    
    # ==================== Input Components ====================
    
    def button(self, label: str, key: Optional[str] = None) -> bool:
        return st.button(label, key=key)
    
    def text_input(self, label: str, value: str = "", key: Optional[str] = None) -> str:
        return st.text_input(label, value, key=key)
    
    def number_input(self, label: str, value: float = 0.0, min_value: Optional[float] = None, 
                    max_value: Optional[float] = None, step: float = 1.0, key: Optional[str] = None) -> float:
        return st.number_input(label, value=value, min_value=min_value, 
                             max_value=max_value, step=step, key=key)
    
    def selectbox(self, label: str, options: List[Any], index: int = 0, key: Optional[str] = None) -> Any:
        return st.selectbox(label, options, index=index, key=key)
    
    def multiselect(self, label: str, options: List[Any], default: List[Any] = None, key: Optional[str] = None) -> List[Any]:
        return st.multiselect(label, options, default=default, key=key)
    
    def checkbox(self, label: str, value: bool = False, key: Optional[str] = None) -> bool:
        return st.checkbox(label, value=value, key=key)
    
    def date_input(self, label: str, value: Optional[date] = None, key: Optional[str] = None) -> date:
        return st.date_input(label, value=value, key=key)
    
    def time_input(self, label: str, value: Optional[time] = None, key: Optional[str] = None) -> time:
        return st.time_input(label, value=value, key=key)
    
    def slider(self, label: str, min_value: float, max_value: float, value: float, step: float = 1.0, key: Optional[str] = None) -> float:
        return st.slider(label, min_value=min_value, max_value=max_value, value=value, step=step, key=key)
    
    def file_uploader(self, label: str, type: List[str] = None, key: Optional[str] = None) -> Optional[Any]:
        return st.file_uploader(label, type=type, key=key)
    
    # ==================== Layout Components ====================
    
    def columns(self, ratios: List[float]) -> List[Any]:
        return st.columns(ratios)
    
    def sidebar(self) -> Any:
        return st.sidebar
    
    def container(self) -> Any:
        return st.container()
    
    def expander(self, label: str, expanded: bool = False) -> Any:
        return st.expander(label, expanded=expanded)
    
    def tabs(self, labels: List[str]) -> List[Any]:
        return st.tabs(labels)
    
    # ==================== Data Display Components ====================
    
    def display_dataframe(self, df: pd.DataFrame, key: Optional[str] = None, height: Optional[int] = None) -> None:
        st.dataframe(df, key=key, height=height, use_container_width=True)
    
    def display_table(self, data: Dict[str, List], key: Optional[str] = None) -> None:
        df = pd.DataFrame(data)
        st.table(df)
    
    def display_chart(self, chart_type: str, data: Any, config: Dict = None) -> None:
        if chart_type == "line":
            st.line_chart(data)
        elif chart_type == "bar":
            st.bar_chart(data)
        elif chart_type == "area":
            st.area_chart(data)
        elif chart_type == "scatter":
            st.scatter_chart(data)
        else:
            st.write(f"Chart type '{chart_type}' not supported")
    
    def display_map(self, center_lat: float, center_lon: float, zoom: int = 10, 
                   markers: List[Dict] = None, key: Optional[str] = None) -> Dict:
        """Display interactive map with markers"""
        # Create folium map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
        
        # Add markers if provided
        if markers:
            for marker in markers:
                folium.Marker(
                    [marker['lat'], marker['lon']],
                    popup=marker.get('popup', ''),
                    icon=folium.Icon(color=marker.get('color', 'blue'))
                ).add_to(m)
        
        # Display map and return interaction data
        map_data = st_folium(m, key=key, width=700, height=500)
        return map_data
    
    def display_json(self, data: Dict, expanded: bool = False) -> None:
        st.json(data, expanded=expanded)
    
    # ==================== Status and Feedback ====================
    
    def success(self, message: str) -> None:
        st.success(message)
    
    def error(self, message: str) -> None:
        st.error(message)
    
    def warning(self, message: str) -> None:
        st.warning(message)
    
    def info(self, message: str) -> None:
        st.info(message)
    
    def spinner(self, message: str = "Loading...") -> Any:
        return st.spinner(message)
    
    # ==================== State Management ====================
    
    def get_session_state(self, key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)
    
    def set_session_state(self, key: str, value: Any) -> None:
        st.session_state[key] = value
    
    def clear_session_state(self) -> None:
        st.session_state.clear()
    
    # ==================== Streamlit-Specific Enhancements ====================
    
    def display_validation_results(self, validation_results: Dict, 
                                 filtered_summary: Dict, raw_summary: Dict) -> None:
        """Enhanced validation results display for Streamlit"""
        if not validation_results:
            self.warning("No validation results to display.")
            return
        
        # Display key metrics
        col1, col2, col3, col4 = self.columns([1, 1, 1, 1])
        
        total_records = raw_summary.get('total_records', 0)
        real_outages = validation_results.get('real_outages', 0)
        false_positives = validation_results.get('false_positives', 0)
        
        with col1:
            self.display_metric("Total Records", total_records)
        with col2:
            self.display_metric("Real Outages", real_outages)
        with col3:
            self.display_metric("False Positives", false_positives)
        with col4:
            if total_records > 0:
                accuracy = real_outages / total_records * 100
                self.display_metric("Accuracy", f"{accuracy:.1f}%")
        
        # Display tables in expandable sections
        if 'real_outage_details' in validation_results and not validation_results['real_outage_details'].empty:
            with self.expander("✅ Real Outages", expanded=True):
                self.display_dataframe(validation_results['real_outage_details'])
        
        if 'false_positive_details' in validation_results and not validation_results['false_positive_details'].empty:
            with self.expander("❌ False Positives", expanded=False):
                self.display_dataframe(validation_results['false_positive_details'])
        
        # Display summary statistics
        with self.expander("📊 Detailed Statistics", expanded=False):
            stats_data = {
                "Metric": ["Total Records", "Real Outages", "False Positives", "Validation Rate"],
                "Value": [
                    total_records,
                    real_outages,
                    false_positives,
                    f"{(real_outages + false_positives) / max(total_records, 1) * 100:.1f}%"
                ]
            }
            self.display_table(stats_data)
    
    def display_chat_interface(self, messages: List[Dict], on_send: Callable[[str], None]) -> None:
        """Enhanced chat interface for Streamlit"""
        self.display_text("💬 Chat with Analysis Results", "subheader")
        
        # Display chat history
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                with st.chat_message("user"):
                    st.write(content)
            else:
                with st.chat_message("assistant"):
                    st.write(content)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the analysis..."):
            # Add user message to display
            with st.chat_message("user"):
                st.write(prompt)
            
            # Call the callback
            on_send(prompt)
    
    def display_time_window_analysis(self, on_analyze: Callable[[datetime, datetime], None]) -> None:
        """Enhanced time window analysis interface"""
        self.display_text("🕒 Time Window Analysis", "subheader")
        
        with self.container():
            col1, col2 = self.columns([1, 1])
            
            with col1:
                self.display_text("Start Date & Time", "text")
                start_date = self.date_input("Start Date", key="start_date")
                start_time = self.time_input("Start Time", key="start_time")
            
            with col2:
                self.display_text("End Date & Time", "text")
                end_date = self.date_input("End Date", key="end_date")
                end_time = self.time_input("End Time", key="end_time")
            
            if self.button("🔍 Analyze Time Window", key="analyze_window"):
                start_datetime = datetime.combine(start_date, start_time)
                end_datetime = datetime.combine(end_date, end_time)
                
                if start_datetime >= end_datetime:
                    self.error("End date/time must be after start date/time")
                else:
                    on_analyze(start_datetime, end_datetime)
    
    # ==================== Additional Streamlit Features ====================
    
    def display_sidebar_status(self, llm_manager, cache_status: Dict) -> None:
        """Display status information in sidebar"""
        with self.sidebar():
            self.display_text("⚡ System Status", "subheader")
            
            # LLM Status
            if llm_manager:
                provider_info = llm_manager.get_provider_info()
                self.display_text(f"🤖 LLM: {provider_info['provider']}", "text")
                self.display_text(f"Model: {provider_info['model']}", "text")
                
                if llm_manager.is_mcp_available():
                    self.display_text("🔗 MCP: Available", "text")
            
            # Cache Status
            self.display_text("💾 Cache Status", "text")
            for cache_name, status in cache_status.items():
                icon = "✅" if status else "❌"
                self.display_text(f"{icon} {cache_name}", "text")
    
    def display_error_boundary(self, error: Exception, context: str = "") -> None:
        """Display error information with context"""
        with self.container():
            self.error(f"An error occurred{': ' + context if context else ''}")
            
            with self.expander("Error Details", expanded=False):
                self.display_text(f"**Error Type:** {type(error).__name__}")
                self.display_text(f"**Error Message:** {str(error)}")
                
                # In debug mode, show full traceback
                if self.get_session_state('debug_mode', False):
                    import traceback
                    self.display_text("**Traceback:**")
                    st.code(traceback.format_exc())
    
    def display_loading_screen(self, message: str, progress: Optional[float] = None) -> None:
        """Display loading screen with optional progress"""
        with self.container():
            self.display_text(f"⏳ {message}", "info")
            if progress is not None:
                self.display_progress(progress)
    
    def file_download_button(self, data: Any, filename: str, mime_type: str = "text/plain") -> bool:
        """Create download button for files"""
        return st.download_button(
            label=f"📥 Download {filename}",
            data=data,
            file_name=filename,
            mime=mime_type
        )