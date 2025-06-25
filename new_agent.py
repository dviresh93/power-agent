"""
Power Outage Analysis Agent - LangGraph Implementation
Compatible with LangGraph Studio
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import folium
from typing import Dict, List, Optional, TypedDict, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
import joblib
from functools import lru_cache
import time
import asyncio

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Import existing services - fail fast if missing
from services.llm_service import LLMManager
from services.weather_service import WeatherService
from services.geocoding_service import GeocodingService  
from services.vector_db_service import OutageVectorDB as VectorDBService
from services.langsmith_service import LangSmithMonitor
from services.usage_tracker import LLMUsageTracker as UsageTracker
from cost_analyzer import CostAnalyzer

# Load environment variables
load_dotenv()

# Configure centralized logging
from config.logging_config import setup_power_agent_logging, get_progress_logger
setup_power_agent_logging()
logger = logging.getLogger(__name__)
progress_logger = get_progress_logger("processing")

# ==================== PROMPT MANAGER ====================
def load_prompts() -> Dict:
    """Load prompts from prompts.json file"""
    try:
        with open('prompts.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("prompts.json file not found!")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing prompts.json: {e}")
        raise

# Load prompts globally
PROMPTS = load_prompts()
logger.info("✅ Prompts loaded from prompts.json")

# ==================== SERVICE MANAGER ====================
class PowerAgentServices:
    """Centralized service manager for all agent services"""
    
    def __init__(self):
        # Initialize all services - fail fast if any are broken
        self.llm_manager = LLMManager()
        self.weather_service = WeatherService()
        self.geocoding_service = GeocodingService()
        self.vector_db_service = VectorDBService()
        self.langsmith_monitor = LangSmithMonitor()
        self.usage_tracker = UsageTracker()
        self.cost_analyzer = CostAnalyzer()
        
        logger.info("✅ All services initialized successfully")
    
    def get_llm(self):
        """Get configured LLM instance - shared across all tools to avoid callback duplication"""
        return self.llm_manager.get_llm()
    
    def get_cost_stats(self) -> Dict:
        """Get current cost and usage statistics"""
        # Get cost data from usage tracking
        try:
            usage_data = self.cost_analyzer.get_usage_data(days=1)  # Last day
            total_cost = usage_data['total_cost'].sum() if not usage_data.empty else 0.0
        except:
            total_cost = 0.0
            
        return {
            "total_cost": total_cost,
            "api_calls": {},  # Will be populated by usage tracker if available
            "cache_stats": {}
        }

# Initialize global services instance
services = PowerAgentServices()

# ==================== ENHANCED STATE DEFINITION ====================
class OutageAnalysisState(TypedDict):
    """Enhanced LangGraph State for Power Outage Analysis - Matches Original App"""
    # Core data state
    dataset_loaded: bool
    validation_complete: bool
    raw_dataset_summary: Dict
    validation_results: Dict
    filtered_summary: Dict
    current_window_analysis: Optional[Dict]
    
    # Chat and interaction state
    chat_context: Dict
    messages: List[Dict]
    current_step: str
    user_input: Optional[str]
    
    # Enhanced state from original application
    uploaded_file_data: Optional[Dict]  # File upload information
    dataset_path: Optional[str]  # Path to loaded dataset
    total_records: int  # Total number of records processed
    processed_count: int  # Number of records processed so far
    max_records_to_process: Optional[int]  # Limit number of records for testing
    
    # Caching and performance
    cache_status: Dict  # Cache hit/miss statistics
    api_usage_stats: Dict  # API call tracking and costs
    processing_time: Dict  # Time tracking for operations
    
    # Geographic and weather data
    geocoding_cache: Dict  # Cached geocoding results
    weather_cache: Dict  # Cached weather data
    unique_locations: List[Dict]  # Unique geographic locations
    
    # UI and visualization
    map_data: Optional[Dict]  # Map visualization data
    selected_time_window: Optional[Dict]  # Time window filter
    ui_filters: Dict  # Active UI filters and settings
    
    # Advanced analysis
    detailed_validation_logs: List[Dict]  # Detailed validation decisions
    statistical_summary: Dict  # Advanced statistical analysis
    geographic_clustering: Dict  # Geographic clustering results
    
    # Report generation
    report_data: Dict  # Generated report content
    report_formats: List[str]  # Available report formats
    
    # Error handling and logging
    errors: List[str]
    warnings: List[str]
    debug_info: Dict  # Debug information for troubleshooting
    
    # LLM and cost tracking
    llm_provider: str  # Currently selected LLM provider
    cost_breakdown: Dict  # Detailed cost analysis
    usage_history: List[Dict]  # Historical usage patterns

# ==================== TOOLS ====================
@tool
def load_csv_dataset(file_path: str = "data/raw_data.csv") -> dict:
    """Load and validate CSV dataset with real data format"""
    logger.info(f"Loading dataset from {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load CSV data
    df = pd.read_csv(file_path)
    
    # Validate required columns (real format)
    required_columns = ['DATETIME', 'CUSTOMERS', 'LATITUDE', 'LONGITUDE']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}. Found columns: {list(df.columns)}")
    
    # Data preprocessing and validation
    try:
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        # Convert back to string for vector DB compatibility
        df['DATETIME_STR'] = df['DATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        raise ValueError(f"Invalid DATETIME format in dataset: {str(e)}")
    
    # Validate numeric columns
    for col in ['CUSTOMERS', 'LATITUDE', 'LONGITUDE']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric")
    
    # Remove any rows with missing data
    initial_count = len(df)
    df = df.dropna(subset=required_columns)
    if len(df) != initial_count:
        logger.warning(f"Removed {initial_count - len(df)} rows with missing data")
    
    if len(df) == 0:
        raise ValueError("No valid data rows found after cleaning")
    
    # Generate summary statistics
    summary = {
        "status": "loaded",
        "total_records": len(df),
        "date_range": {
            "start": df['DATETIME'].min().isoformat(),
            "end": df['DATETIME'].max().isoformat()
        },
        "geographic_bounds": {
            "lat_min": float(df['LATITUDE'].min()),
            "lat_max": float(df['LATITUDE'].max()),
            "lon_min": float(df['LONGITUDE'].min()),
            "lon_max": float(df['LONGITUDE'].max())
        },
        "customer_stats": {
            "total_affected": int(df['CUSTOMERS'].sum()),
            "avg_per_outage": float(df['CUSTOMERS'].mean()),
            "max_single_outage": int(df['CUSTOMERS'].max())
        },
        "data_sample": df.head(3).to_dict('records')  # Include sample for verification
    }
    
    # Store in vector DB - force reload to bypass cache and ensure data is stored
    # Create a copy with string datetime for vector DB compatibility
    df_for_vectordb = df.copy()
    df_for_vectordb['DATETIME'] = df_for_vectordb['DATETIME_STR']
    services.vector_db_service.load_outage_data(df_for_vectordb, force_reload=True)
    
    return summary

@tool
def get_weather_data(latitude: float, longitude: float, datetime_str: str) -> dict:
    """Get historical weather data for validation"""
    # Convert datetime string to proper format for weather service
    try:
        if isinstance(datetime_str, str):
            dt = pd.to_datetime(datetime_str)
        else:
            dt = datetime_str
    except Exception as e:
        raise ValueError(f"Invalid datetime format '{datetime_str}': {str(e)}")
    
    # Get weather data from service
    weather_data = services.weather_service.get_historical_weather(latitude, longitude, dt)
    return weather_data

@tool
def validate_outage_report(outage_report: dict, weather_data: dict) -> dict:
    """Enhanced outage validation with detailed analysis"""
    try:
        # Use false_positive_detection prompt from prompts.json
        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPTS["false_positive_detection"]["system"]),
            ("human", PROMPTS["false_positive_detection"]["human"])
        ])
        
        # Format outage report for the prompt
        outage_report_formatted = f"""Time: {outage_report.get('datetime', 'Unknown')}
Location: {outage_report.get('latitude', 'Unknown')}, {outage_report.get('longitude', 'Unknown')}
Customers Affected: {outage_report.get('customers', 'Unknown')}"""
        
        # Format weather data for the prompt
        if weather_data.get('api_status') == 'failed':
            weather_data_formatted = f"Weather data unavailable: {weather_data.get('error', 'Unknown error')}"
        else:
            weather_data_formatted = f"""Temperature: {weather_data.get('temperature', 'N/A')}°C
Precipitation: {weather_data.get('precipitation', 'N/A')} mm/h
Wind Speed: {weather_data.get('wind_speed', 'N/A')} km/h
Wind Gusts: {weather_data.get('wind_gusts', 'N/A')} km/h
Snowfall: {weather_data.get('snowfall', 'N/A')} cm"""
        
        chain = validation_prompt | services.get_llm()
        logger.debug(f"🔄 Making LLM call for outage validation (shared instance)")
        response = chain.invoke({
            "outage_report": outage_report_formatted,
            "weather_data": weather_data_formatted
        })
        
        # Try to parse JSON response
        try:
            result = json.loads(response.content)
            result['outage_id'] = outage_report.get('id', 'unknown')
            result['weather_data'] = weather_data
            return result
        except json.JSONDecodeError:
            # Fallback to text parsing
            classification = "REAL OUTAGE" if "REAL OUTAGE" in response.content else "FALSE POSITIVE"
            return {
                "classification": classification,
                "confidence": 0.8,
                "reasoning": response.content,
                "weather_factors": [],
                "severity_score": 5,
                "outage_id": outage_report.get('id', 'unknown'),
                "weather_data": weather_data
            }
        
    except Exception as e:
        logger.error(f"❌ Validation error: {str(e)}")
        return {
            "classification": "VALIDATION ERROR",
            "confidence": 0.0,
            "reasoning": f"Error during validation: {str(e)}",
            "weather_factors": [],
            "severity_score": 0,
            "outage_id": outage_report.get('id', 'unknown'),
            "error": str(e)
        }

@tool
def chat_about_results(question: str, validation_results: dict) -> str:
    """Chat about validation results and analysis"""
    try:
        # Use chatbot_assistant prompt from prompts.json
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPTS["chatbot_assistant"]["system"]),
            ("human", PROMPTS["chatbot_assistant"]["human"])
        ])
        
        chain = chat_prompt | services.get_llm()
        logger.debug(f"🔄 Making LLM call for chat (shared instance)")
        response = chain.invoke({
            "user_question": question,
            "analysis_context": json.dumps(validation_results, indent=2, default=str)
        })
        
        return response.content
        
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        return f"Chat error: {str(e)}"

@tool
def generate_comprehensive_report(validation_results: dict, raw_summary: dict = None) -> str:
    """Generate a comprehensive analysis report using LLM"""
    try:
        # Use comprehensive_report_generation prompt from prompts.json
        report_prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPTS["comprehensive_report_generation"]["system"]),
            ("human", PROMPTS["comprehensive_report_generation"]["human"])
        ])
        
        # Calculate time period from raw_summary if available
        time_period = "Unknown"
        if raw_summary and raw_summary.get('date_range'):
            start = raw_summary['date_range'].get('start', 'Unknown')
            end = raw_summary['date_range'].get('end', 'Unknown')
            time_period = f"{start} to {end}"
        
        # Create map data summary for the prompt
        map_data = {
            "total_locations": len(validation_results.get('real_outages', [])) + len(validation_results.get('false_positives', [])),
            "geographic_spread": "Analysis covers multiple coordinate locations"
        }
        
        chain = report_prompt | services.get_llm()
        logger.debug(f"🔄 Making LLM call for comprehensive report generation")
        response = chain.invoke({
            "raw_summary": json.dumps(raw_summary or {}, indent=2, default=str),
            "validation_results": json.dumps(validation_results, indent=2, default=str),
            "time_period": time_period,
            "map_data": json.dumps(map_data, indent=2)
        })
        
        return response.content
        
    except Exception as e:
        logger.error(f"❌ Report generation error: {str(e)}")
        return f"Report generation failed: {str(e)}"

@tool
def generate_interactive_map(validation_results: dict, raw_summary: dict = None) -> dict:
    """Generate interactive map with validation results"""
    try:
        real_outages = validation_results.get('real_outages', [])
        false_positives = validation_results.get('false_positives', [])
        
        if not real_outages and not false_positives:
            return {"status": "no_data", "message": "No validation results to map"}
        
        # Calculate map center
        all_coords = []
        for outage in real_outages + false_positives:
            if 'latitude' in outage and 'longitude' in outage:
                all_coords.append([outage['latitude'], outage['longitude']])
        
        if not all_coords:
            return {"status": "no_coordinates", "message": "No coordinate data available"}
        
        center_lat = float(np.mean([coord[0] for coord in all_coords]))
        center_lon = float(np.mean([coord[1] for coord in all_coords]))
        
        # Create Folium map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add real outages (red markers)
        for outage in real_outages:
            if 'LATITUDE' in outage and 'LONGITUDE' in outage:
                folium.CircleMarker(
                    location=[outage['LATITUDE'], outage['LONGITUDE']],
                    radius=8,
                    popup=f"Real Outage - {outage.get('CUSTOMERS', 'Unknown')} customers",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7
                ).add_to(m)
        
        # Add false positives (blue markers)
        for outage in false_positives:
            if 'LATITUDE' in outage and 'LONGITUDE' in outage:
                folium.CircleMarker(
                    location=[outage['LATITUDE'], outage['LONGITUDE']],
                    radius=6,
                    popup=f"False Positive - {outage.get('CUSTOMERS', 'Unknown')} customers",
                    color='blue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.5
                ).add_to(m)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; top: 10px; right: 10px; z-index: 1000; 
                    background-color: white; border: 2px solid grey; border-radius: 5px;
                    padding: 10px;">
            <h4>Legend</h4>
            <p><span style="color: red;">●</span> Real Outages ({real_count})</p>
            <p><span style="color: blue;">●</span> False Positives ({false_count})</p>
        </div>
        """.format(real_count=len(real_outages), false_count=len(false_positives))
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map to HTML string
        map_html = m._repr_html_()
        
        return {
            "status": "success",
            "map_html": map_html,
            "center": [center_lat, center_lon],
            "bounds": {
                "north": float(max([coord[0] for coord in all_coords])),
                "south": float(min([coord[0] for coord in all_coords])),
                "east": float(max([coord[1] for coord in all_coords])),
                "west": float(min([coord[1] for coord in all_coords]))
            },
            "marker_count": {
                "real_outages": len(real_outages),
                "false_positives": len(false_positives)
            }
        }
        
    except Exception as e:
        logger.error(f"Map generation error: {str(e)}")
        return {"status": "error", "error": str(e)}

@tool
def generate_statistical_analysis(validation_results: dict) -> dict:
    """Generate comprehensive statistical analysis"""
    try:
        real_outages = validation_results.get('real_outages', [])
        false_positives = validation_results.get('false_positives', [])
        
        total_reports = len(real_outages) + len(false_positives)
        
        if total_reports == 0:
            return {"status": "no_data", "message": "No validation results to analyze"}
        
        # Basic statistics
        accuracy_rate = len(real_outages) / total_reports if total_reports > 0 else 0
        
        # Confidence analysis
        real_confidences = [r.get('confidence', 0.8) for r in real_outages if 'confidence' in r]
        false_confidences = [f.get('confidence', 0.8) for f in false_positives if 'confidence' in f]
        
        # Severity analysis
        real_severities = [r.get('severity_score', 5) for r in real_outages if 'severity_score' in r]
        false_severities = [f.get('severity_score', 2) for f in false_positives if 'severity_score' in f]
        
        # Customer impact analysis
        real_customers = [r.get('customers', 0) for r in real_outages if 'customers' in r]
        false_customers = [f.get('customers', 0) for f in false_positives if 'customers' in f]
        
        analysis = {
            "basic_stats": {
                "total_reports": total_reports,
                "real_outages": len(real_outages),
                "false_positives": len(false_positives),
                "accuracy_rate": accuracy_rate,
                "false_positive_rate": len(false_positives) / total_reports if total_reports > 0 else 0
            },
            "confidence_analysis": {
                "real_avg_confidence": float(np.mean(real_confidences)) if real_confidences else 0.0,
                "false_avg_confidence": float(np.mean(false_confidences)) if false_confidences else 0.0,
                "high_confidence_real": len([c for c in real_confidences if c > 0.8]),
                "high_confidence_false": len([c for c in false_confidences if c > 0.8])
            },
            "severity_analysis": {
                "real_avg_severity": float(np.mean(real_severities)) if real_severities else 0.0,
                "false_avg_severity": float(np.mean(false_severities)) if false_severities else 0.0,
                "high_severity_real": len([s for s in real_severities if s > 7]),
                "high_severity_false": len([s for s in false_severities if s > 7])
            },
            "customer_impact": {
                "real_total_customers": sum(real_customers),
                "false_total_customers": sum(false_customers),
                "real_avg_customers": float(np.mean(real_customers)) if real_customers else 0.0,
                "false_avg_customers": float(np.mean(false_customers)) if false_customers else 0.0
            }
        }
        
        return {"status": "success", "analysis": analysis}
        
    except Exception as e:
        logger.error(f"Statistical analysis error: {str(e)}")
        return {"status": "error", "error": str(e)}

# ==================== WORKFLOW NODES ====================
def load_data_node(state: OutageAnalysisState) -> OutageAnalysisState:
    """Enhanced data loading with full CSV processing"""
    logger.info("🔄 Loading and processing dataset...")
    start_time = time.time()
    
    # Load CSV dataset using enhanced tool
    dataset_path = state.get("dataset_path", "data/raw_data.csv")
    dataset_result = load_csv_dataset.invoke({"file_path": dataset_path})
    
    # Update state with results
    state["dataset_loaded"] = True
    state["raw_dataset_summary"] = dataset_result
    state["total_records"] = dataset_result.get("total_records", 0)
    state["processed_count"] = 0
    
    # Track processing time
    processing_time = time.time() - start_time
    state["processing_time"] = state.get("processing_time", {})
    state["processing_time"]["data_loading"] = processing_time
    
    # Update cost tracking
    state["api_usage_stats"] = services.get_cost_stats()
    
    success_msg = f"✅ Dataset loaded: {dataset_result['total_records']} records from {dataset_result['date_range']['start']} to {dataset_result['date_range']['end']}"
    state["messages"].append({
        "role": "assistant",
        "content": success_msg
    })
    
    state["current_step"] = "data_loaded"
    logger.info(success_msg)
    
    return state

def process_outages_node(state: OutageAnalysisState) -> OutageAnalysisState:
    """Bulk validation processing with real data format"""
    logger.info("🔄 Starting bulk validation with weather data...")
    start_time = time.time()
    
    if not state.get("dataset_loaded", False):
        raise ValueError("Cannot validate reports: dataset not loaded")
    
    # Get dataset from vector DB - directly query the collection
    try:
        # Query all records directly from ChromaDB collection
        all_results = services.vector_db_service.collection.get()
        
        # Extract metadata which contains our outage records
        if all_results and 'metadatas' in all_results and all_results['metadatas']:
            dataset_records = all_results['metadatas']
            logger.info(f"✅ Retrieved {len(dataset_records)} records from vector database")
        else:
            raise ValueError(f"No data found in vector database. Collection result: {all_results}")
    except Exception as e:
        raise ValueError(f"Unable to analyze data - vector database access failed: {str(e)}")
    total_records = len(dataset_records)
    
    if total_records == 0:
        raise ValueError("No data records found in vector database")
    
    # Check for record processing limit (useful for testing)
    max_records = state.get("max_records_to_process", None)
    if max_records and max_records < total_records:
        dataset_records = dataset_records[:max_records]
        total_records = max_records
        logger.info(f"⚠️ Processing limited to {max_records} records for testing (dataset has {len(dataset_records)} total)")
    
    real_outages = []
    false_positives = []
    detailed_logs = []
    processed_count = 0
    
    logger.info(f"Processing {total_records} outage reports for validation...")
    
    # Process each outage report
    for record in dataset_records:
        try:
            # Get weather data for the location and time (now using uppercase field names)
            weather_data = get_weather_data.invoke({
                "latitude": record["LATITUDE"],  # Use uppercase field names
                "longitude": record["LONGITUDE"], 
                "datetime_str": record["DATETIME"]
            })
            
            # Validate the outage report
            validation_result = validate_outage_report.invoke({
                "outage_report": {
                    "datetime": record["DATETIME"],
                    "latitude": record["LATITUDE"],
                    "longitude": record["LONGITUDE"],
                    "customers": record["CUSTOMERS"]
                },
                "weather_data": weather_data
            })
            
            # Add original record data to result
            validation_result.update(record)
            
            if validation_result.get("classification") == "REAL OUTAGE":
                real_outages.append(validation_result)
            else:
                false_positives.append(validation_result)
            
            # Log detailed validation decision
            detailed_logs.append({
                "record_index": processed_count,
                "classification": validation_result.get("classification"),
                "confidence": validation_result.get("confidence", 0.0),
                "reasoning": validation_result.get("reasoning", ""),
                "weather_data": weather_data
            })
            
            processed_count += 1
            state["processed_count"] = processed_count
            
            # Rate limiting
            time.sleep(0.1)
            
            # Use progress logger for better controlled output
            progress_logger.log_progress(processed_count, total_records, "reports validated")
            
            # Log individual record progress only in debug mode
            logger.debug(f"✅ Record {processed_count}/{total_records}: {validation_result.get('classification', 'Unknown')} (confidence: {validation_result.get('confidence', 0):.2f})")
                
        except Exception as e:
            logger.error(f"Validation failed for record {processed_count}: {str(e)}")
            raise ValueError(f"Unable to analyze data - validation failed for record {processed_count}: {str(e)}")
    
    # Compile final results
    validation_results = {
        "real_outages": real_outages,
        "false_positives": false_positives,
        "total_processed": processed_count,
        "validation_complete": True,
        "processing_stats": {
            "success_rate": 1.0,  # We succeeded for all records or would have failed
            "real_outage_rate": len(real_outages) / processed_count if processed_count > 0 else 0,
            "false_positive_rate": len(false_positives) / processed_count if processed_count > 0 else 0
        }
    }
    
    # Update state
    state["validation_complete"] = True
    state["validation_results"] = validation_results
    state["detailed_validation_logs"] = detailed_logs
    
    # Track processing time
    processing_time = time.time() - start_time
    state["processing_time"] = state.get("processing_time", {})
    state["processing_time"]["validation"] = processing_time
    
    # Update cost tracking
    state["api_usage_stats"] = services.get_cost_stats()
    
    success_msg = f"✅ Validation complete: {len(real_outages)} real outages, {len(false_positives)} false positives from {processed_count} reports"
    state["messages"].append({
        "role": "assistant",
        "content": success_msg
    })
    
    state["current_step"] = "validation_complete"
    logger.info(success_msg)
    
    return state

def output_node(state: OutageAnalysisState) -> OutageAnalysisState:
    """Simple output node: Save .pkl file and provide basic summary"""
    logger.info("🔄 Saving results to .pkl file...")
    start_time = time.time()
    
    validation_results = state.get("validation_results", {})
    
    if not validation_results or not validation_results.get("validation_complete"):
        raise ValueError("Cannot save output: validation not complete")
    
    # Create basic summary
    real_outages = validation_results.get("real_outages", [])
    false_positives = validation_results.get("false_positives", [])
    
    filtered_summary = {
        "total_real": len(real_outages),
        "total_false": len(false_positives),
        "total_processed": validation_results.get("total_processed", 0),
        "accuracy_rate": 0.0,
        "processing_complete": True
    }
    
    # Calculate accuracy
    total = filtered_summary["total_real"] + filtered_summary["total_false"]
    if total > 0:
        filtered_summary["accuracy_rate"] = filtered_summary["total_real"] / total
    
    state["filtered_summary"] = filtered_summary
    
    # Save complete state to .pkl file (CRITICAL for persistence)
    try:
        import joblib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        pkl_filename = f"cache/analysis_results_{timestamp}.pkl"
        os.makedirs("cache", exist_ok=True)
        
        # Save complete state for other agents to use
        joblib.dump(state, pkl_filename)
        state["pkl_file"] = pkl_filename
        logger.info(f"✅ Results saved to {pkl_filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save .pkl file: {str(e)}")
        state["errors"].append(f"PKL save failed: {str(e)}")
    
    # Track processing time
    processing_time = time.time() - start_time
    state["processing_time"] = state.get("processing_time", {})
    state["processing_time"]["output"] = processing_time
    
    success_msg = f"✅ Analysis complete: {filtered_summary['total_real']} real outages ({filtered_summary['accuracy_rate']*100:.1f}% accuracy), {filtered_summary['total_false']} false positives"
    state["messages"].append({
        "role": "assistant",
        "content": success_msg
    })
    
    state["current_step"] = "output_complete"
    logger.info(success_msg)
    
    return state

# Chat and report functionality removed from main workflow
# These will be implemented as separate agents

# ==================== SIMPLIFIED WORKFLOW ====================
# No conditional logic needed - simple linear flow

# ==================== GRAPH CONSTRUCTION ====================
def create_graph():
    """Create simplified 3-node LangGraph as per PROJECT_VISION_AND_PLAN.md"""
    
    # Initialize the graph
    workflow = StateGraph(OutageAnalysisState)
    
    # Add 3 nodes as specified in vision
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("process_outages", process_outages_node)  # Renamed from validate_reports
    workflow.add_node("output", output_node)  # Combines results processing + report generation
    
    # Simple linear flow as per vision
    workflow.add_edge(START, "load_data")
    workflow.add_edge("load_data", "process_outages")
    workflow.add_edge("process_outages", "output")
    workflow.add_edge("output", END)
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Create the graph instance for LangGraph Studio
graph = create_graph()

# ==================== HELPER FUNCTIONS ====================
def run_analysis(initial_state: dict = None) -> dict:
    """Run the complete enhanced outage analysis workflow"""
    
    # Initialize enhanced state
    if initial_state is None:
        initial_state = {}
    
    # Ensure all required fields exist with defaults
    default_state = {
        # Core data state
        "dataset_loaded": False,
        "validation_complete": False,
        "raw_dataset_summary": {},
        "validation_results": {},
        "filtered_summary": {},
        "current_window_analysis": None,
        
        # Chat and interaction state
        "chat_context": {},
        "messages": [],
        "current_step": "starting",
        "user_input": None,
        
        # Enhanced state from original application
        "uploaded_file_data": None,
        "dataset_path": "data/raw_data.csv",  # Default path
        "total_records": 0,
        "processed_count": 0,
        "max_records_to_process": initial_state.get("max_records_to_process", None),  # Default: process all records
        
        # Caching and performance
        "cache_status": {},
        "api_usage_stats": {},
        "processing_time": {},
        
        # Geographic and weather data
        "geocoding_cache": {},
        "weather_cache": {},
        "unique_locations": [],
        
        # UI and visualization
        "map_data": None,
        "selected_time_window": None,
        "ui_filters": {},
        
        # Advanced analysis
        "detailed_validation_logs": [],
        "statistical_summary": {},
        "geographic_clustering": {},
        
        # Report generation
        "report_data": {},
        "report_formats": [],
        
        # Error handling and logging
        "errors": [],
        "warnings": [],
        "debug_info": {},
        
        # LLM and cost tracking
        "llm_provider": "claude",  # Default provider
        "cost_breakdown": {},
        "usage_history": []
    }
    
    # Merge user provided state with defaults
    for key, value in default_state.items():
        if key not in initial_state:
            initial_state[key] = value
    
    # Ensure messages list exists and add session start message
    if "messages" not in initial_state or not isinstance(initial_state["messages"], list):
        initial_state["messages"] = []
    
    initial_state["messages"].append({
        "role": "assistant",
        "content": "🚀 Starting enhanced power outage analysis workflow..."
    })
    
    logger.info("🚀 Starting enhanced outage analysis workflow")
    
    try:
        # Run the workflow
        config = {"configurable": {"thread_id": "outage_analysis_session"}}
        final_state = graph.invoke(initial_state, config)
        
        # Log completion
        total_time = sum(final_state.get("processing_time", {}).values())
        logger.info(f"✅ Analysis workflow completed in {total_time:.2f} seconds")
        
        return final_state
        
    except Exception as e:
        logger.error(f"❌ Analysis workflow failed: {str(e)}")
        # Return error state
        initial_state["errors"].append(f"Workflow execution failed: {str(e)}")
        initial_state["current_step"] = "error"
        initial_state["messages"].append({
            "role": "assistant",
            "content": f"❌ Analysis failed: {str(e)}"
        })
        return initial_state

def chat_with_agent(message: str, thread_id: str = "outage_analysis_session") -> str:
    """Chat with the agent about analysis results"""
    
    # Get current state
    config = {"configurable": {"thread_id": thread_id}}
    current_state = graph.get_state(config)
    
    if current_state and current_state.values:
        # Update state with user input
        updated_state = current_state.values.copy()
        updated_state["user_input"] = message
        
        # Continue the workflow
        final_state = graph.invoke(updated_state, config)
        
        # Return the last assistant message
        messages = final_state.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "No response")
    
    return "Please run the analysis first before chatting."

if __name__ == "__main__":
    # Test the agent
    print("Testing Power Outage Analysis Agent...")
    result = run_analysis()
    print(f"Analysis complete. Current step: {result.get('current_step')}")
    print(f"Errors: {result.get('errors', [])}")