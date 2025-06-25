"""
Abstract UI interface for framework independence.
This allows switching between Streamlit, FastAPI, Flask, Gradio, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, date, time
import pandas as pd

from core.models import ValidationResult, OutageAnalysisState, UIState, ReportData


class AbstractUI(ABC):
    """Abstract base class for UI framework adapters"""
    
    def __init__(self):
        self.state = UIState(current_view="main", data_loaded=False, analysis_complete=False)
    
    # ==================== Core UI Components ====================
    
    @abstractmethod
    def display_title(self, title: str) -> None:
        """Display main application title"""
        pass
    
    @abstractmethod
    def display_text(self, text: str, style: str = "normal") -> None:
        """Display text with optional styling (normal, header, subheader, error, success)"""
        pass
    
    @abstractmethod
    def display_metric(self, label: str, value: Any, delta: Optional[str] = None) -> None:
        """Display a metric with optional delta"""
        pass
    
    @abstractmethod
    def display_progress(self, progress: float, text: str = "") -> None:
        """Display progress bar (0.0 to 1.0)"""
        pass
    
    # ==================== Input Components ====================
    
    @abstractmethod
    def button(self, label: str, key: Optional[str] = None) -> bool:
        """Display button, return True if clicked"""
        pass
    
    @abstractmethod
    def text_input(self, label: str, value: str = "", key: Optional[str] = None) -> str:
        """Text input field"""
        pass
    
    @abstractmethod
    def number_input(self, label: str, value: float = 0.0, min_value: Optional[float] = None, 
                    max_value: Optional[float] = None, step: float = 1.0, key: Optional[str] = None) -> float:
        """Number input field"""
        pass
    
    @abstractmethod
    def selectbox(self, label: str, options: List[Any], index: int = 0, key: Optional[str] = None) -> Any:
        """Dropdown selection"""
        pass
    
    @abstractmethod
    def multiselect(self, label: str, options: List[Any], default: List[Any] = None, key: Optional[str] = None) -> List[Any]:
        """Multi-selection widget"""
        pass
    
    @abstractmethod
    def checkbox(self, label: str, value: bool = False, key: Optional[str] = None) -> bool:
        """Checkbox input"""
        pass
    
    @abstractmethod
    def date_input(self, label: str, value: Optional[date] = None, key: Optional[str] = None) -> date:
        """Date picker"""
        pass
    
    @abstractmethod
    def time_input(self, label: str, value: Optional[time] = None, key: Optional[str] = None) -> time:
        """Time picker"""
        pass
    
    @abstractmethod
    def slider(self, label: str, min_value: float, max_value: float, value: float, step: float = 1.0, key: Optional[str] = None) -> float:
        """Slider input"""
        pass
    
    @abstractmethod
    def file_uploader(self, label: str, type: List[str] = None, key: Optional[str] = None) -> Optional[Any]:
        """File upload widget"""
        pass
    
    # ==================== Layout Components ====================
    
    @abstractmethod
    def columns(self, ratios: List[float]) -> List[Any]:
        """Create column layout, return column objects"""
        pass
    
    @abstractmethod
    def sidebar(self) -> Any:
        """Return sidebar container"""
        pass
    
    @abstractmethod
    def container(self) -> Any:
        """Return generic container"""
        pass
    
    @abstractmethod
    def expander(self, label: str, expanded: bool = False) -> Any:
        """Expandable section"""
        pass
    
    @abstractmethod
    def tabs(self, labels: List[str]) -> List[Any]:
        """Tab layout, return tab objects"""
        pass
    
    # ==================== Data Display Components ====================
    
    @abstractmethod
    def display_dataframe(self, df: pd.DataFrame, key: Optional[str] = None, height: Optional[int] = None) -> None:
        """Display pandas DataFrame"""
        pass
    
    @abstractmethod
    def display_table(self, data: Dict[str, List], key: Optional[str] = None) -> None:
        """Display simple table"""
        pass
    
    @abstractmethod
    def display_chart(self, chart_type: str, data: Any, config: Dict = None) -> None:
        """Display chart (line, bar, scatter, etc.)"""
        pass
    
    @abstractmethod
    def display_map(self, center_lat: float, center_lon: float, zoom: int = 10, 
                   markers: List[Dict] = None, key: Optional[str] = None) -> Dict:
        """Display interactive map with markers, return interaction data"""
        pass
    
    @abstractmethod
    def display_json(self, data: Dict, expanded: bool = False) -> None:
        """Display formatted JSON"""
        pass
    
    # ==================== Status and Feedback ====================
    
    @abstractmethod
    def success(self, message: str) -> None:
        """Display success message"""
        pass
    
    @abstractmethod
    def error(self, message: str) -> None:
        """Display error message"""
        pass
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """Display warning message"""
        pass
    
    @abstractmethod
    def info(self, message: str) -> None:
        """Display info message"""
        pass
    
    @abstractmethod
    def spinner(self, message: str = "Loading...") -> Any:
        """Display loading spinner, return context manager"""
        pass
    
    # ==================== State Management ====================
    
    @abstractmethod
    def get_session_state(self, key: str, default: Any = None) -> Any:
        """Get value from session state"""
        pass
    
    @abstractmethod
    def set_session_state(self, key: str, value: Any) -> None:
        """Set value in session state"""
        pass
    
    @abstractmethod
    def clear_session_state(self) -> None:
        """Clear all session state"""
        pass
    
    # ==================== App-Specific Methods ====================
    
    def display_validation_results(self, validation_results: Dict, 
                                 filtered_summary: Dict, raw_summary: Dict) -> None:
        """Display validation results with metrics and tables"""
        # Default implementation using abstract methods
        total_records = raw_summary.get('total_records', 0)
        real_outages = validation_results.get('real_outages', 0)
        false_positives = validation_results.get('false_positives', 0)
        
        # Display metrics
        cols = self.columns([1, 1, 1])
        with cols[0]:
            self.display_metric("Total Records", total_records)
        with cols[1]:
            self.display_metric("Real Outages", real_outages)
        with cols[2]:
            self.display_metric("False Positives", false_positives)
        
        # Display tables
        if 'real_outage_details' in validation_results:
            self.display_text("Real Outages", "subheader")
            self.display_dataframe(validation_results['real_outage_details'])
        
        if 'false_positive_details' in validation_results:
            self.display_text("False Positives", "subheader")
            self.display_dataframe(validation_results['false_positive_details'])
    
    def display_chat_interface(self, messages: List[Dict], on_send: Callable[[str], None]) -> None:
        """Display chat interface for Q&A"""
        # Display message history
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            style = "normal" if role == 'user' else "info"
            self.display_text(f"**{role.title()}**: {content}", style)
        
        # Input for new message
        new_message = self.text_input("Ask a question about the results:", key="chat_input")
        if self.button("Send") and new_message:
            on_send(new_message)
    
    def display_time_window_analysis(self, on_analyze: Callable[[datetime, datetime], None]) -> None:
        """Display time window analysis interface"""
        self.display_text("Time Window Analysis", "subheader")
        
        cols = self.columns([1, 1])
        with cols[0]:
            start_date = self.date_input("Start Date")
            start_time = self.time_input("Start Time")
        with cols[1]:
            end_date = self.date_input("End Date") 
            end_time = self.time_input("End Time")
        
        if self.button("Analyze Time Window"):
            start_datetime = datetime.combine(start_date, start_time)
            end_datetime = datetime.combine(end_date, end_time)
            on_analyze(start_datetime, end_datetime)


class UIComponent(ABC):
    """Base class for reusable UI components"""
    
    def __init__(self, ui: AbstractUI):
        self.ui = ui
    
    @abstractmethod
    def render(self, **kwargs) -> Any:
        """Render the component"""
        pass


class ValidationResultsComponent(UIComponent):
    """Reusable component for displaying validation results"""
    
    def render(self, validation_results: Dict, filtered_summary: Dict, raw_summary: Dict) -> None:
        self.ui.display_validation_results(validation_results, filtered_summary, raw_summary)


class MapComponent(UIComponent):
    """Reusable component for map visualization"""
    
    def render(self, outage_data: pd.DataFrame, center: Tuple[float, float] = None) -> Dict:
        if center is None:
            center = (outage_data['LATITUDE'].mean(), outage_data['LONGITUDE'].mean())
        
        markers = []
        for _, row in outage_data.iterrows():
            markers.append({
                'lat': row['LATITUDE'],
                'lon': row['LONGITUDE'],
                'popup': f"Customers: {row['CUSTOMERS']}\nTime: {row['DATETIME']}",
                'color': 'red' if row.get('validation_status') == 'real_outage' else 'blue'
            })
        
        return self.ui.display_map(center[0], center[1], zoom=10, markers=markers)


class ChatComponent(UIComponent):
    """Reusable component for chat interface"""
    
    def render(self, messages: List[Dict], on_send: Callable[[str], None]) -> None:
        self.ui.display_chat_interface(messages, on_send)