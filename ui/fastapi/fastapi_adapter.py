"""
FastAPI adapter implementing the abstract UI interface.
Demonstrates how easy it is to switch UI frameworks with the modular architecture.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, date, time
import pandas as pd
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json

from interfaces.ui_interface import AbstractUI


class FastAPIAdapter(AbstractUI):
    """FastAPI implementation of the abstract UI interface"""
    
    def __init__(self, app: FastAPI = None):
        super().__init__()
        self.app = app or FastAPI(title="Power Agent", description="LLM-Powered Outage Analysis")
        self.templates = Jinja2Templates(directory="ui/fastapi/templates")
        self.session_data = {}  # In production, use proper session management
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})
        
        @self.app.get("/api/status")
        async def get_status():
            return {"status": "running", "framework": "fastapi"}
        
        @self.app.post("/api/validate")
        async def validate_data():
            # Validation endpoint
            return {"message": "Validation started"}
        
        @self.app.post("/api/chat")
        async def chat(message: str = Form(...)):
            # Chat endpoint
            return {"response": f"Echo: {message}"}
    
    # ==================== Core UI Components ====================
    # FastAPI doesn't have direct UI components, so these return data structures
    # that would be rendered by JavaScript frontend
    
    def display_title(self, title: str) -> None:
        return {"type": "title", "content": title}
    
    def display_text(self, text: str, style: str = "normal") -> None:
        return {"type": "text", "content": text, "style": style}
    
    def display_metric(self, label: str, value: Any, delta: Optional[str] = None) -> None:
        return {"type": "metric", "label": label, "value": value, "delta": delta}
    
    def display_progress(self, progress: float, text: str = "") -> None:
        return {"type": "progress", "value": progress, "text": text}
    
    # ==================== Input Components ====================
    
    def button(self, label: str, key: Optional[str] = None) -> bool:
        # In FastAPI, buttons would be handled via POST requests
        return {"type": "button", "label": label, "key": key}
    
    def text_input(self, label: str, value: str = "", key: Optional[str] = None) -> str:
        return {"type": "text_input", "label": label, "value": value, "key": key}
    
    def number_input(self, label: str, value: float = 0.0, min_value: Optional[float] = None, 
                    max_value: Optional[float] = None, step: float = 1.0, key: Optional[str] = None) -> float:
        return {
            "type": "number_input", 
            "label": label, 
            "value": value, 
            "min_value": min_value,
            "max_value": max_value,
            "step": step,
            "key": key
        }
    
    def selectbox(self, label: str, options: List[Any], index: int = 0, key: Optional[str] = None) -> Any:
        return {"type": "selectbox", "label": label, "options": options, "index": index, "key": key}
    
    def multiselect(self, label: str, options: List[Any], default: List[Any] = None, key: Optional[str] = None) -> List[Any]:
        return {"type": "multiselect", "label": label, "options": options, "default": default or [], "key": key}
    
    def checkbox(self, label: str, value: bool = False, key: Optional[str] = None) -> bool:
        return {"type": "checkbox", "label": label, "value": value, "key": key}
    
    def date_input(self, label: str, value: Optional[date] = None, key: Optional[str] = None) -> date:
        return {"type": "date_input", "label": label, "value": value.isoformat() if value else None, "key": key}
    
    def time_input(self, label: str, value: Optional[time] = None, key: Optional[str] = None) -> time:
        return {"type": "time_input", "label": label, "value": value.isoformat() if value else None, "key": key}
    
    def slider(self, label: str, min_value: float, max_value: float, value: float, step: float = 1.0, key: Optional[str] = None) -> float:
        return {
            "type": "slider",
            "label": label,
            "min_value": min_value,
            "max_value": max_value,
            "value": value,
            "step": step,
            "key": key
        }
    
    def file_uploader(self, label: str, type: List[str] = None, key: Optional[str] = None) -> Optional[Any]:
        return {"type": "file_uploader", "label": label, "accepted_types": type, "key": key}
    
    # ==================== Layout Components ====================
    
    def columns(self, ratios: List[float]) -> List[Any]:
        return {"type": "columns", "ratios": ratios}
    
    def sidebar(self) -> Any:
        return {"type": "sidebar"}
    
    def container(self) -> Any:
        return {"type": "container"}
    
    def expander(self, label: str, expanded: bool = False) -> Any:
        return {"type": "expander", "label": label, "expanded": expanded}
    
    def tabs(self, labels: List[str]) -> List[Any]:
        return {"type": "tabs", "labels": labels}
    
    # ==================== Data Display Components ====================
    
    def display_dataframe(self, df: pd.DataFrame, key: Optional[str] = None, height: Optional[int] = None) -> None:
        return {
            "type": "dataframe",
            "data": df.to_dict('records'),
            "columns": df.columns.tolist(),
            "key": key,
            "height": height
        }
    
    def display_table(self, data: Dict[str, List], key: Optional[str] = None) -> None:
        return {"type": "table", "data": data, "key": key}
    
    def display_chart(self, chart_type: str, data: Any, config: Dict = None) -> None:
        return {"type": "chart", "chart_type": chart_type, "data": data, "config": config or {}}
    
    def display_map(self, center_lat: float, center_lon: float, zoom: int = 10, 
                   markers: List[Dict] = None, key: Optional[str] = None) -> Dict:
        return {
            "type": "map",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": zoom,
            "markers": markers or [],
            "key": key
        }
    
    def display_json(self, data: Dict, expanded: bool = False) -> None:
        return {"type": "json", "data": data, "expanded": expanded}
    
    # ==================== Status and Feedback ====================
    
    def success(self, message: str) -> None:
        return {"type": "alert", "level": "success", "message": message}
    
    def error(self, message: str) -> None:
        return {"type": "alert", "level": "error", "message": message}
    
    def warning(self, message: str) -> None:
        return {"type": "alert", "level": "warning", "message": message}
    
    def info(self, message: str) -> None:
        return {"type": "alert", "level": "info", "message": message}
    
    def spinner(self, message: str = "Loading...") -> Any:
        return {"type": "spinner", "message": message}
    
    # ==================== State Management ====================
    
    def get_session_state(self, key: str, default: Any = None) -> Any:
        return self.session_data.get(key, default)
    
    def set_session_state(self, key: str, value: Any) -> None:
        self.session_data[key] = value
    
    def clear_session_state(self) -> None:
        self.session_data.clear()
    
    # ==================== FastAPI-Specific Methods ====================
    
    def render_component(self, component_data: Dict) -> str:
        """Render component data as HTML"""
        component_type = component_data.get("type")
        
        if component_type == "title":
            return f"<h1>{component_data['content']}</h1>"
        elif component_type == "text":
            style_class = f"text-{component_data.get('style', 'normal')}"
            return f"<p class='{style_class}'>{component_data['content']}</p>"
        elif component_type == "metric":
            return f"""
            <div class="metric">
                <div class="metric-label">{component_data['label']}</div>
                <div class="metric-value">{component_data['value']}</div>
                {f"<div class='metric-delta'>{component_data['delta']}</div>" if component_data.get('delta') else ""}
            </div>
            """
        elif component_type == "button":
            return f"<button class='btn' id='{component_data.get('key', '')}'>{component_data['label']}</button>"
        elif component_type == "dataframe":
            # Create HTML table from dataframe data
            columns = component_data['columns']
            rows = component_data['data']
            
            html = "<table class='dataframe'><thead><tr>"
            for col in columns:
                html += f"<th>{col}</th>"
            html += "</tr></thead><tbody>"
            
            for row in rows:
                html += "<tr>"
                for col in columns:
                    html += f"<td>{row.get(col, '')}</td>"
                html += "</tr>"
            
            html += "</tbody></table>"
            return html
        
        return f"<div>Unsupported component: {component_type}</div>"
    
    def create_page_layout(self, components: List[Dict]) -> str:
        """Create full page layout from components"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Power Agent - FastAPI</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric { padding: 10px; border: 1px solid #ddd; margin: 10px; display: inline-block; }
                .metric-label { font-size: 12px; color: #666; }
                .metric-value { font-size: 24px; font-weight: bold; }
                .btn { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
                .btn:hover { background: #0056b3; }
                .dataframe { border-collapse: collapse; width: 100%; }
                .dataframe th, .dataframe td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .dataframe th { background-color: #f2f2f2; }
                .text-error { color: red; }
                .text-success { color: green; }
                .text-warning { color: orange; }
                .text-info { color: blue; }
            </style>
        </head>
        <body>
        """
        
        for component in components:
            html += self.render_component(component)
        
        html += """
        </body>
        </html>
        """
        
        return html


def create_fastapi_app():
    """Factory function to create FastAPI app with adapter"""
    adapter = FastAPIAdapter()
    
    @adapter.app.get("/demo", response_class=HTMLResponse)
    async def demo_page(request: Request):
        """Demo page showing the adapter in action"""
        components = [
            adapter.display_title("⚡ Power Agent - FastAPI Demo"),
            adapter.display_text("This demonstrates the same functionality with a different UI framework", "info"),
            adapter.display_metric("Total Records", "1,234", "+5%"),
            adapter.display_metric("Real Outages", "856", "+2%"),
            adapter.display_metric("False Positives", "378", "-1%"),
            adapter.button("Start Validation", "validate_btn"),
            adapter.display_dataframe(
                pd.DataFrame({
                    'DATETIME': ['2023-01-01 12:00:00', '2023-01-02 14:30:00'],
                    'LATITUDE': [40.7128, 34.0522],
                    'LONGITUDE': [-74.0060, -118.2437],
                    'CUSTOMERS': [150, 200]
                })
            )
        ]
        
        return HTMLResponse(adapter.create_page_layout(components))
    
    return adapter.app


if __name__ == "__main__":
    import uvicorn
    app = create_fastapi_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)