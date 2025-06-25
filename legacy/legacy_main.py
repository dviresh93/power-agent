"""
Complete LLM-Powered Outage Analysis Agent
- Properly displays validation results (real vs false positive)
- Follows 2025 LangGraph best practices with MCP integration
- Clean logging with transparent error handling
- Fixed missing function and correct data filtering display
"""

import os
import json
import logging
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta, date, time
import numpy as np
from typing import Dict, List, Optional, Tuple, TypedDict, Annotated
from dotenv import load_dotenv
import joblib
from functools import lru_cache
import matplotlib.pyplot as plt
from reportlab.platypus import Image as RLImage
import io
import time as time_module
import asyncio

from services.llm_service import LLMManager

from cost_analyzer import CostAnalyzer, enhanced_usage_decorator

cost_analyzer = CostAnalyzer()


# Set up clean logging first thing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure watchdog file watcher to be less aggressive
import os
os.environ["STREAMLIT_WATCHER_WARNING_THRESHOLD"] = "300"  # Higher threshold to reduce file watching
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  # Disable file watching to avoid inotify limits
logger = logging.getLogger(__name__)

# LangGraph imports - 2025 best practices
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# Pydantic imports with proper fallback
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

# LLM imports - Claude as default with OpenAI fallback
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Vector DB imports
import chromadb

# Reverse geocoding for city/county names
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    logger.warning("geopy not available - install with: pip install geopy")

# MCP imports for 2025 best practices
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
    logger.info("MCP adapters available for enhanced tool integration")
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP adapters not available - install with: pip install langchain-mcp-adapters")

# Load environment variables
load_dotenv()

# ==================== ENHANCED STATE MANAGEMENT ====================
class OutageAnalysisState(TypedDict):
    """2025 LangGraph State Management"""
    dataset_loaded: bool
    validation_complete: bool
    raw_dataset_summary: Dict
    validation_results: Dict
    filtered_summary: Dict  # Summary after false positive filtering
    current_window_analysis: Optional[Dict]
    chat_context: Dict
    errors: List[str]

# ==================== REMOVED DUPLICATE LLM MANAGER ====================
# The LLMManager is now imported from services.llm_service to ensure
# the single, correct version is used throughout the application.
# class LLMManager: ... (Removed)

# ==================== WEATHER SERVICE (API-based) ====================
class WeatherService:
    """Weather validation using Open-Meteo Historical API with caching"""
    
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Set up caching for weather data requests
        try:
            import requests_cache
            self.session = requests_cache.CachedSession(
                'weather_cache', 
                expire_after=86400,  # Cache for 24 hours
                allowable_methods=('GET', 'POST')
            )
            logger.info("✅ Weather API caching enabled")
        except ImportError:
            import requests
            self.session = requests
            logger.warning("⚠️ requests_cache not installed. Weather API calls will not be cached.")
        
    def get_historical_weather(self, lat: float, lon: float, date: datetime) -> Dict:
        """Fetch weather data for validation"""
        try:
            date_str = date.strftime("%Y-%m-%d")
            
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": date_str,
                "end_date": date_str,
                "hourly": "temperature_2m,precipitation,windspeed_10m,windgusts_10m,snowfall",
                "timezone": "America/Chicago"
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            hourly_data = data.get('hourly', {})
            
            # Get data for the specific hour
            target_hour = min(date.hour, len(hourly_data.get('temperature_2m', [])) - 1) if hourly_data.get('temperature_2m') else 0
            
            return {
                'timestamp': date.isoformat(),
                'coordinates': {'lat': lat, 'lon': lon},
                'temperature': self._safe_get(hourly_data.get('temperature_2m', []), target_hour),
                'precipitation': self._safe_get(hourly_data.get('precipitation', []), target_hour),
                'wind_speed': self._safe_get(hourly_data.get('windspeed_10m', []), target_hour),
                'wind_gusts': self._safe_get(hourly_data.get('windgusts_10m', []), target_hour),
                'snowfall': self._safe_get(hourly_data.get('snowfall', []), target_hour),
                'api_status': 'success'
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"⚠️  Weather API request failed: {str(e)}")
            return {'error': f'Weather API error: {str(e)}', 'timestamp': date.isoformat(), 'api_status': 'failed'}
        except Exception as e:
            logger.error(f"⚠️  Weather service error: {str(e)}")
            return {'error': f'Weather service error: {str(e)}', 'timestamp': date.isoformat(), 'api_status': 'failed'}
    
    def _safe_get(self, data_list: List, index: int):
        """Safely get data from list"""
        if not data_list or index >= len(data_list) or index < 0:
            return None
        return data_list[index]

# ==================== GEOCODING SERVICE ====================
class GeocodingService:
    """Service for reverse geocoding lat/lon to city/county names"""
    
    def __init__(self):
        if GEOPY_AVAILABLE:
            self.geolocator = Nominatim(user_agent="power-outage-agent")
            self._cache = {}  # In-memory cache
            self.cache_file = "./cache/geocoding_cache.joblib"
            self._load_persistent_cache()
            logger.info("✅ Geocoding service initialized with persistent cache")
        else:
            self.geolocator = None
            logger.warning("⚠️ Geocoding service unavailable - install geopy")
    
    def _load_persistent_cache(self):
        """Load geocoding cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                self._cache = joblib.load(self.cache_file)
                logger.info(f"📍 Loaded {len(self._cache)} cached locations from disk")
            else:
                self._cache = {}
        except Exception as e:
            logger.warning(f"⚠️ Failed to load geocoding cache: {str(e)}")
            self._cache = {}
    
    def _save_persistent_cache(self):
        """Save geocoding cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            joblib.dump(self._cache, self.cache_file)
            logger.info(f"💾 Saved {len(self._cache)} locations to geocoding cache")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save geocoding cache: {str(e)}")
    
    @lru_cache(maxsize=1000)
    def get_location_name(self, lat: float, lon: float) -> Dict[str, str]:
        """Get city, county, state for given coordinates"""
        if not self.geolocator:
            return {
                'city': 'Unknown',
                'county': 'Unknown', 
                'state': 'Unknown',
                'display_name': f"Lat: {lat:.4f}, Lon: {lon:.4f}"
            }
        
        # Round coordinates to reduce cache misses for nearby points
        lat_rounded = round(lat, 3)
        lon_rounded = round(lon, 3)
        cache_key = f"{lat_rounded},{lon_rounded}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            location = self.geolocator.reverse(f"{lat}, {lon}", timeout=15, language='en')
            if location and location.raw.get('address'):
                address = location.raw['address']
                
                # Extract location components with better fallbacks
                city = (address.get('city') or 
                       address.get('town') or 
                       address.get('village') or 
                       address.get('hamlet') or
                       address.get('suburb') or
                       address.get('neighbourhood') or
                       address.get('locality') or
                       None)
                
                county = (address.get('county') or 
                         address.get('state_district') or
                         address.get('administrative_area_level_2') or
                         None)
                
                state = (address.get('state') or 
                        address.get('province') or
                        address.get('administrative_area_level_1') or
                        None)
                
                # Build display name more intelligently
                parts = []
                if city:
                    parts.append(city)
                if county and county != city:
                    parts.append(county)
                if state and state != county:
                    parts.append(state)
                
                if parts:
                    display_name = ", ".join(parts)
                else:
                    # Try to get any meaningful location info
                    road = address.get('road') or address.get('street')
                    postcode = address.get('postcode')
                    if road or postcode:
                        display_name = f"{road or ''} {postcode or ''}".strip()
                    else:
                        display_name = f"Near {lat:.3f}, {lon:.3f}"
                
                result = {
                    'city': city or 'Unknown',
                    'county': county or 'Unknown',
                    'state': state or 'Unknown',
                    'display_name': display_name
                }
                
                # Cache the result both in memory and disk
                self._cache[cache_key] = result
                self._save_persistent_cache()
                return result
            else:
                # No address found
                result = {
                    'city': 'Unknown',
                    'county': 'Unknown', 
                    'state': 'Unknown',
                    'display_name': f"Near {lat:.3f}, {lon:.3f}"
                }
                self._cache[cache_key] = result
                self._save_persistent_cache()
                return result
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.warning(f"⚠️ Geocoding failed for {lat}, {lon}: {str(e)}")
            result = {
                'city': 'Lookup Failed',
                'county': 'Lookup Failed', 
                'state': 'Lookup Failed',
                'display_name': f"Lat: {lat:.4f}, Lon: {lon:.4f}"
            }
            return result
        except Exception as e:
            logger.error(f"❌ Geocoding error for {lat}, {lon}: {str(e)}")
            result = {
                'city': 'Error',
                'county': 'Error', 
                'state': 'Error',
                'display_name': f"Lat: {lat:.4f}, Lon: {lon:.4f}"
            }
            return result

# ==================== VECTOR DATABASE ====================
class OutageVectorDB:
    """Vector database for outage data management"""
    
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
            
            try:
                self.collection = self.client.get_collection("outages")
                logger.info("✅ Connected to existing outages collection")
            except:
                self.collection = self.client.create_collection("outages")
                logger.info("✅ Created new outages collection")
        except Exception as e:
            logger.error(f"❌ Vector DB initialization failed: {str(e)}")
            raise
    
    def load_outage_data(self, df: pd.DataFrame, force_reload: bool = False) -> Dict:
        """Load data and return summary with caching support"""
        try:
            # Check for cached data first
            cache_dir = "./cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "outage_data_summary.joblib")
            vector_cache_file = os.path.join(cache_dir, "vector_db_status.json")
            
            # Load from cache if available and not forcing reload
            if not force_reload and os.path.exists(cache_file) and os.path.exists(vector_cache_file):
                try:
                    logger.info("🔄 Loading dataset summary from cache")
                    cached_summary = joblib.load(cache_file)
                    return cached_summary
                except Exception as e:
                    logger.warning(f"⚠️ Cache loading failed: {str(e)}. Proceeding with full load.")
            
            logger.info(f"📊 Loading {len(df)} outage records into vector database")
            
            # Clear existing data
            try:
                self.client.delete_collection("outages")
                self.collection = self.client.create_collection("outages")
            except:
                pass
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in df.iterrows():
                doc_text = f"""Power outage on {row['DATETIME']} at coordinates {row['LATITUDE']}, {row['LONGITUDE']} affecting {row['CUSTOMERS']} customers."""
                
                metadata = {
                    'datetime': row['DATETIME'],
                    'latitude': float(row['LATITUDE']),
                    'longitude': float(row['LONGITUDE']),
                    'customers': int(row['CUSTOMERS']),
                    'date': row['DATETIME'][:10],
                    'hour': int(row['DATETIME'][11:13]) if len(row['DATETIME']) > 11 else 0
                }
                
                documents.append(doc_text.strip())
                metadatas.append(metadata)
                ids.append(f"outage_{idx}")
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Generate raw dataset summary
            summary = self._generate_raw_summary(df)
            
            # Cache the results
            try:
                joblib.dump(summary, cache_file)
                with open(vector_cache_file, 'w') as f:
                    json.dump({"timestamp": datetime.now().isoformat(), "record_count": len(df)}, f)
                logger.info("✅ Data cached successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cache data: {str(e)}")
                
            logger.info("✅ Data loaded successfully")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def _generate_raw_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary of raw dataset before validation"""
        try:
            df['datetime_parsed'] = pd.to_datetime(df['DATETIME'])
            
            return {
                "total_reports": int(len(df)),
                "date_range": {
                    "start": df['datetime_parsed'].min().strftime('%Y-%m-%d'),
                    "end": df['datetime_parsed'].max().strftime('%Y-%m-%d')
                },
                "raw_customer_claims": {
                    "total_claimed": int(df['CUSTOMERS'].sum()),
                    "avg_per_report": float(df['CUSTOMERS'].mean()),
                    "max_single_report": int(df['CUSTOMERS'].max())
                },
                "geographic_coverage": {
                    "lat_range": f"{float(df['LATITUDE'].min()):.3f} to {float(df['LATITUDE'].max()):.3f}",
                    "lon_range": f"{float(df['LONGITUDE'].min()):.3f} to {float(df['LONGITUDE'].max()):.3f}",
                    "center": [float(df['LATITUDE'].mean()), float(df['LONGITUDE'].mean())]
                }
            }
        except Exception as e:
            logger.error(f"❌ Error generating summary: {str(e)}")
            return {"error": str(e)}

# ==================== LLM TOOLS FOLLOWING 2025 PATTERNS ====================

@tool
def validate_outage_report(outage_report: dict, weather_data: dict) -> str:
    """Validate outage report against weather conditions using LLM analysis"""
    try:
        # Use the configured LLMManager
        llm_manager = LLMManager(model_config=st.session_state.get('llm_config'))
        
        # ... existing validation code ...
        
        chain = validation_prompt | llm_manager.get_llm()
        
        # Track this as a validation operation
        start_time = datetime.now()
        response = chain.invoke({
            "datetime": outage_report.get('datetime', 'Unknown'),
            "latitude": outage_report.get('latitude', 'Unknown'),
            "longitude": outage_report.get('longitude', 'Unknown'),
            "customers": outage_report.get('customers', 'Unknown'),
            "weather_summary": weather_summary
        })
        
        # Log enhanced usage
        cost_analyzer.log_enhanced_usage({
            'operation_type': 'validation',
            'model_name': llm_manager.get_provider_info().get('model', 'unknown'),
            'timestamp': start_time.isoformat(),
            'duration_seconds': (datetime.now() - start_time).total_seconds(),
            'success': True,
            'context': 'outage_validation'
        })
        
        return response.content
        
    except Exception as e:
        # Log failed operations
        cost_analyzer.log_enhanced_usage({
            'operation_type': 'validation',
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': str(e)
        })
        logger.error(f"❌ Validation error: {str(e)}")
        return f"VALIDATION ERROR: {str(e)}"

@tool  
def generate_comprehensive_report(validation_results: dict, raw_summary: dict) -> str:
    """Generate a comprehensive outage analysis report with enhanced tracking"""
    try:
        llm_manager = LLMManager(model_config=st.session_state.get('llm_config'))
        
        # ... existing report generation code ...
        
        start_time = datetime.now()
        response = chain.invoke({
            "raw_summary": json.dumps(raw_summary, indent=2, default=str),
            "validation_results": json.dumps(validation_results, indent=2, default=str),
            "time_period": time_period,
            "map_data": json.dumps(map_data, indent=2, default=str)
        })
        
        # Log enhanced usage
        cost_analyzer.log_enhanced_usage({
            'operation_type': 'report',
            'report_type': 'standard',
            'model_name': llm_manager.get_provider_info().get('model', 'unknown'),
            'timestamp': start_time.isoformat(),
            'duration_seconds': (datetime.now() - start_time).total_seconds(),
            'success': True,
            'context': 'comprehensive_report'
        })
        
        return f"{response.content}\n\n{map_section}"
        
    except Exception as e:
        cost_analyzer.log_enhanced_usage({
            'operation_type': 'report',
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': str(e)
        })
        logger.error(f"❌ Report generation error: {str(e)}")
        return f"Report generation error: {str(e)}"

@tool
def chat_about_results(question: str, context: dict) -> str:
    """Chat about validation results with enhanced tracking"""
    try:
        llm_manager = LLMManager(model_config=st.session_state.get('llm_config'))
        
        # ... existing chat code ...
        
        start_time = datetime.now()
        response = chain.invoke({
            "user_question": question,
            "analysis_context": json.dumps(context, indent=2, default=str)
        })
        
        # Log enhanced usage
        cost_analyzer.log_enhanced_usage({
            'operation_type': 'chat',
            'model_name': llm_manager.get_provider_info().get('model', 'unknown'),
            'timestamp': start_time.isoformat(),
            'duration_seconds': (datetime.now() - start_time).total_seconds(),
            'success': True,
            'context': 'results_chat',
            'question_length': len(question)
        })
        
        return response.content
        
    except Exception as e:
        cost_analyzer.log_enhanced_usage({
            'operation_type': 'chat',
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': str(e)
        })
        logger.error(f"❌ Chat error: {str(e)}")
        return f"Chat error: {str(e)}"

@tool
def validate_outage_report(outage_report: dict, weather_data: dict) -> str:
    """Validate outage report against weather conditions using LLM analysis"""
    try:
        # Use the configured LLMManager
        llm_manager = LLMManager(model_config=st.session_state.get('llm_config'))
        
        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert power grid engineer specializing in outage validation.

Your task: Determine if an outage REPORT is REAL or FALSE POSITIVE based on weather conditions.

FALSE POSITIVES occur when:
- Weather conditions are mild (winds <25 mph, minimal precipitation, normal temperatures)
- Equipment sensor malfunctions
- Data processing errors
- Communication glitches

REAL OUTAGES are caused by:
- High winds: >25 mph sustained or >35 mph gusts
- Heavy precipitation: >0.5 inches/hour
- Ice/snow accumulation: >2 inches
- Temperature extremes: <10°F or >95°F
- Lightning/storms
- Equipment failures during severe weather

Respond with 'REAL OUTAGE' or 'FALSE POSITIVE' followed by detailed technical reasoning."""),
            ("human", """Analyze this outage report:

REPORT DETAILS:
Time: {datetime}
Location: {latitude}, {longitude}
Customers Claimed: {customers}

WEATHER CONDITIONS:
{weather_summary}

Classification: REAL OUTAGE or FALSE POSITIVE? Provide detailed analysis.""")
        ])
        
        # Format weather data
        if weather_data.get('api_status') == 'failed':
            weather_summary = f"Weather data unavailable: {weather_data.get('error', 'Unknown error')}"
        else:
            weather_summary = f"""
Temperature: {weather_data.get('temperature', 'N/A')}°C
Precipitation: {weather_data.get('precipitation', 'N/A')} mm/h
Wind Speed: {weather_data.get('wind_speed', 'N/A')} km/h
Wind Gusts: {weather_data.get('wind_gusts', 'N/A')} km/h
Snowfall: {weather_data.get('snowfall', 'N/A')} cm
"""
        
        chain = validation_prompt | llm_manager.get_llm()
        response = chain.invoke({
            "datetime": outage_report.get('datetime', 'Unknown'),
            "latitude": outage_report.get('latitude', 'Unknown'),
            "longitude": outage_report.get('longitude', 'Unknown'),
            "customers": outage_report.get('customers', 'Unknown'),
            "weather_summary": weather_summary
        })
        
        return response.content
        
    except Exception as e:
        logger.error(f"❌ Validation error: {str(e)}")
        return f"VALIDATION ERROR: {str(e)}"

@tool
def generate_comprehensive_report(validation_results: dict, raw_summary: dict) -> str:
    """Generate a comprehensive outage analysis report with false positive details and map data"""
    try:
        # Use the configured LLMManager
        llm_manager = LLMManager(model_config=st.session_state.get('llm_config'))
        
        # Load the report generation prompt
        with open('prompts.json', 'r') as f:
            prompts = json.load(f)
        
        report_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts["comprehensive_report_generation"]["system"]),
            ("human", prompts["comprehensive_report_generation"]["human"])
        ])
        
        # Determine time period from raw data
        date_range = raw_summary.get('date_range', {})
        time_period = f"{date_range.get('start', 'Unknown')} to {date_range.get('end', 'Unknown')}"
        
        # Generate map data summary for the report
        map_data = generate_map_data_summary(validation_results)
        
        chain = report_prompt | llm_manager.get_llm()
        
        response = chain.invoke({
            "raw_summary": json.dumps(raw_summary, indent=2, default=str),
            "validation_results": json.dumps(validation_results, indent=2, default=str),
            "time_period": time_period,
            "map_data": json.dumps(map_data, indent=2, default=str)
        })
        
        # Append map data section to the report
        report_content = response.content
        map_section = generate_map_section_for_report(validation_results)
        
        return f"{report_content}\n\n{map_section}"
        
    except Exception as e:
        logger.error(f"❌ Report generation error: {str(e)}")
        return f"Report generation error: {str(e)}"

@tool
def generate_exhaustive_report(validation_results: dict, raw_summary: dict) -> str:
    """Generate an exhaustive technical report with detailed explanations"""
    try:
        # Use the configured LLMManager
        llm_manager = LLMManager(model_config=st.session_state.get('llm_config'))
        with open('prompts.json', 'r') as f:
            prompts = json.load(f)
        
        report_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts["exhaustive_report_generation"]["system"]),
            ("human", prompts["exhaustive_report_generation"]["human"])
        ])
        
        # Determine time period from raw data
        date_range = raw_summary.get('date_range', {})
        time_period = f"{date_range.get('start', 'Unknown')} to {date_range.get('end', 'Unknown')}"
        
        # Generate map data summary for the report
        map_data = generate_map_data_summary(validation_results)
        
        # Extract detailed decision data for exhaustive analysis
        false_positives = validation_results.get('false_positives', [])
        real_outages = validation_results.get('real_outages', [])
        
        # Sort false positives first (as requested), then real outages
        all_decisions = []
        for fp in false_positives:
            fp['classification'] = 'FALSE_POSITIVE'
            all_decisions.append(fp)
        for ro in real_outages:
            ro['classification'] = 'REAL_OUTAGE'
            all_decisions.append(ro)
        
        # Limit data size to prevent LLM hanging
        max_decisions = 50  # Limit to prevent context overload
        limited_decisions = all_decisions[:max_decisions] if len(all_decisions) > max_decisions else all_decisions
        
        # Truncate large text fields to prevent context explosion
        def truncate_text_fields(obj, max_length=200):
            if isinstance(obj, dict):
                return {k: truncate_text_fields(v, max_length) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [truncate_text_fields(item, max_length) for item in obj]
            elif isinstance(obj, str) and len(obj) > max_length:
                return obj[:max_length] + "..."
            return obj
        
        # Apply truncation to prevent massive context
        truncated_raw_summary = truncate_text_fields(raw_summary)
        truncated_validation_results = truncate_text_fields(validation_results)
        truncated_decisions = truncate_text_fields(limited_decisions)
        
        logger.info(f"📊 Exhaustive report: Processing {len(limited_decisions)}/{len(all_decisions)} decisions")
        
        chain = report_prompt | llm_manager.get_llm()
        
        logger.info("🤖 Invoking LLM for exhaustive report generation...")
        try:
            # Add timeout to prevent infinite hanging
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("LLM call timed out after 300 seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5-minute timeout
            
            response = chain.invoke({
                "raw_summary": json.dumps(truncated_raw_summary, indent=2, default=str),
                "validation_results": json.dumps(truncated_validation_results, indent=2, default=str),
                "time_period": time_period,
                "map_data": json.dumps(map_data, indent=2, default=str),
                "all_decisions": json.dumps(truncated_decisions, indent=2, default=str),
                "false_positives_count": len(false_positives),
                "real_outages_count": len(real_outages),
                "decisions_shown": len(limited_decisions),
                "total_decisions": len(all_decisions)
            })
            
            signal.alarm(0)  # Cancel timeout
            logger.info("✅ LLM response received successfully")
            
        except TimeoutError as e:
            logger.error(f"❌ LLM call timed out: {str(e)}")
            return "Exhaustive report generation timed out. The dataset may be too large. Try using the Default report mode or reducing the date range."
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            logger.error(f"❌ LLM call failed: {str(e)}")
            raise
        
        # Append map data section to the report
        report_content = response.content
        map_section = generate_map_section_for_report(validation_results)
        
        return f"{report_content}\n\n{map_section}"
        
    except Exception as e:
        logger.error(f"❌ Exhaustive report generation error: {str(e)}")
        return f"Exhaustive report generation error: {str(e)}"

@tool
def chat_about_results(question: str, context: dict) -> str:
    """Chat about validation results with full context"""
    try:
        # Use the configured LLMManager
        llm_manager = LLMManager(model_config=st.session_state.get('llm_config'))
        
        # Check if we have sufficient validation context
        validation_results = context.get('validation_results', {})
        has_validation_data = bool(validation_results.get('real_outages') or validation_results.get('false_positives'))
        
        if not has_validation_data:
            return """I don't have sufficient validation results to answer your question comprehensively. 

**Current Status:** Only raw dataset information is available from cache.

**To get complete analysis:**
1. Click 'Update Cache from Data Folder' in the sidebar
2. This will load data from `data/raw_data.csv` and run full validation
3. Once complete, I'll have detailed weather analysis and real vs false positive classifications

**What I can tell you now:** Basic dataset statistics from the cached raw data. For detailed outage analysis, weather correlations, and false positive identification, please update the cache first."""
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert power grid operations assistant. You have completed validation of outage reports, determining which are real outages vs false positives based on weather analysis.

You can help users understand:
1. Why specific reports were classified as false positives
2. What weather conditions caused real outages  
3. Patterns in false positive causes
4. Operational recommendations
5. Infrastructure improvements

Be specific and provide actionable insights based on the validation results.

If the user asks about data that might not be in the current time/location range, suggest they may need to update the cache with newer data."""),
            ("human", "{user_question}\n\nValidation Context:\n{analysis_context}")
        ])
        
        chain = chat_prompt | llm_manager.get_llm()
        
        response = chain.invoke({
            "user_question": question,
            "analysis_context": json.dumps(context, indent=2, default=str)
        })
        
        return response.content
        
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        return f"Chat error: {str(e)}"

# ==================== MAIN VALIDATION LOGIC ====================
def validate_all_reports(df: pd.DataFrame, request_delay: float = 0.5, max_retries: int = 3) -> Dict:
    """Validate all outage reports and return comprehensive results"""
    
    weather_service = WeatherService()
    geocoding_service = GeocodingService()
    real_outages = []
    false_positives = []
    validation_errors = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Use passed parameters for rate limiting configuration
    REQUEST_DELAY = request_delay  # Seconds between requests
    MAX_RETRIES = max_retries
    BASE_RETRY_DELAY = 2  # Base delay for retries
    
    logger.info(f"🔍 Starting validation of {len(df)} outage reports with rate limiting")
    
    for idx, row in df.iterrows():
        progress = (idx + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Validating report {idx+1}/{len(df)}: {row['CUSTOMERS']} customers claimed")
        
        try:
            # Parse datetime
            report_datetime = datetime.strptime(row['DATETIME'], "%Y-%m-%d %H:%M:%S")
            
            # Get weather data
            weather = weather_service.get_historical_weather(
                lat=row['LATITUDE'],
                lon=row['LONGITUDE'],
                date=report_datetime
            )
            
            # Get location name using reverse geocoding
            location_info = geocoding_service.get_location_name(
                lat=row['LATITUDE'],
                lon=row['LONGITUDE']
            )
            
            # Prepare report data
            report_data = {
                'datetime': row['DATETIME'],
                'latitude': row['LATITUDE'],
                'longitude': row['LONGITUDE'],
                'customers': row['CUSTOMERS'],
                'location_name': location_info['display_name'],
                'city': location_info['city'],
                'county': location_info['county'],
                'state': location_info['state']
            }
            
            # Validate with LLM - add rate limiting to prevent API overload
            validation_result = None
            retry_delay = BASE_RETRY_DELAY
            
            for attempt in range(MAX_RETRIES):
                try:
                    validation_result = validate_outage_report.invoke({
                        "outage_report": report_data,
                        "weather_data": weather
                    })
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if "overloaded" in error_msg or "rate" in error_msg or "limit" in error_msg:
                        if attempt < MAX_RETRIES - 1:
                            logger.warning(f"API overloaded, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{MAX_RETRIES})")
                            status_text.text(f"API overloaded - retrying report {idx+1}/{len(df)} in {retry_delay}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                            time_module.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            logger.error(f"Max retries reached for report {idx+1}, marking as validation error")
                            validation_result = f"VALIDATION ERROR: API overloaded after {MAX_RETRIES} attempts"
                    else:
                        logger.error(f"Validation error for report {idx+1}: {str(e)}")
                        validation_result = f"VALIDATION ERROR: {str(e)}"
                    break
            
            # Add delay between requests to prevent overwhelming the API
            time_module.sleep(REQUEST_DELAY)
            
            # Determine classification
            is_real = "REAL OUTAGE" in validation_result.upper()
            
            result_entry = {
                **report_data,
                'weather_data': weather,
                'claude_analysis': validation_result,
                'is_real_outage': is_real
            }
            
            if is_real:
                real_outages.append(result_entry)
            else:
                false_positives.append(result_entry)
                
        except Exception as e:
            logger.error(f"❌ Error validating report {idx}: {str(e)}")
            validation_errors.append(f"Report {idx}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    # Calculate comprehensive statistics
    total_reports = len(df)
    real_count = len(real_outages)
    false_count = len(false_positives)
    
    # Calculate actual customer impact (only real outages)
    actual_customers_affected = sum(r['customers'] for r in real_outages)
    claimed_customers_total = int(df['CUSTOMERS'].sum())
    
    logger.info(f"✅ Validation complete: {real_count} real, {false_count} false positives")
    
    validation_results = {
        'total_reports': total_reports,
        'real_outages': real_outages,
        'false_positives': false_positives,
        'validation_errors': validation_errors,
        'statistics': {
            'real_count': real_count,
            'false_positive_count': false_count,
            'false_positive_rate': (false_count / total_reports * 100) if total_reports > 0 else 0,
            'total_customers_actually_affected': actual_customers_affected,
            'total_customers_claimed': claimed_customers_total,
            'customer_impact_reduction': claimed_customers_total - actual_customers_affected
        }
    }
    
    # Cache validation results
    try:
        cache_dir = "./cache"
        os.makedirs(cache_dir, exist_ok=True)
        validation_cache_file = os.path.join(cache_dir, "validation_results.joblib")
        joblib.dump(validation_results, validation_cache_file)
        logger.info("✅ Validation results cached successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to cache validation results: {str(e)}")
    
    return validation_results

def generate_map_data_summary(validation_results: dict) -> dict:
    """Generate map data summary for reports"""
    real_outages = validation_results.get('real_outages', [])
    false_positives = validation_results.get('false_positives', [])
    
    if not real_outages and not false_positives:
        return {"error": "No outage data available"}
    
    all_outages = real_outages + false_positives
    
    # Calculate geographic bounds and center
    if all_outages:
        lats = [o['latitude'] for o in all_outages]
        lons = [o['longitude'] for o in all_outages]
        
        map_summary = {
            "geographic_bounds": {
                "north": max(lats),
                "south": min(lats),
                "east": max(lons),
                "west": min(lons)
            },
            "center_point": {
                "latitude": sum(lats) / len(lats),
                "longitude": sum(lons) / len(lons)
            },
            "real_outages_count": len(real_outages),
            "false_positives_count": len(false_positives),
            "total_points": len(all_outages)
        }
        
        # Add geographic distribution statistics
        if real_outages:
            real_lats = [o['latitude'] for o in real_outages]
            real_lons = [o['longitude'] for o in real_outages]
            map_summary["real_outages_center"] = {
                "latitude": sum(real_lats) / len(real_lats),
                "longitude": sum(real_lons) / len(real_lons)
            }
        
        if false_positives:
            false_lats = [o['latitude'] for o in false_positives]
            false_lons = [o['longitude'] for o in false_positives]
            map_summary["false_positives_center"] = {
                "latitude": sum(false_lats) / len(false_lats),
                "longitude": sum(false_lons) / len(false_lons)
            }
        
        return map_summary
    
    return {"error": "No valid coordinate data"}

def generate_map_section_for_report(validation_results: dict) -> str:
    """Generate a detailed map section for the markdown report"""
    real_outages = validation_results.get('real_outages', [])
    false_positives = validation_results.get('false_positives', [])
    
    map_section = "\n\n# 🗺️ Geographic Distribution Analysis\n\n"
    
    if not real_outages and not false_positives:
        return map_section + "No outage data available for mapping.\n"
    
    # Geographic summary
    map_data = generate_map_data_summary(validation_results)
    if "error" not in map_data:
        bounds = map_data["geographic_bounds"]
        center = map_data["center_point"]
        
        map_section += f"""## Geographic Coverage

**Outage Distribution:**
- Total locations mapped: {map_data['total_points']}
- Real outages: {map_data['real_outages_count']} locations
- False positives: {map_data['false_positives_count']} locations

**Geographic Bounds:**
- North: {bounds['north']:.4f}°
- South: {bounds['south']:.4f}°
- East: {bounds['east']:.4f}°
- West: {bounds['west']:.4f}°
- Center: {center['latitude']:.4f}°, {center['longitude']:.4f}°

"""
    
    # Real outages table
    if real_outages:
        map_section += "## ✅ Real Outages - Geographic Details\n\n"
        map_section += "| Time | Customers | Latitude | Longitude | Weather Analysis |\n"
        map_section += "|------|-----------|----------|-----------|------------------|\n"
        
        for outage in real_outages[:20]:  # Limit to first 20 for readability
            analysis = outage.get('claude_analysis', 'No analysis')[:100].replace('\n', ' ').replace('|', '\|')
            map_section += f"| {outage['datetime'][:16]} | {outage['customers']} | {outage['latitude']:.4f} | {outage['longitude']:.4f} | {analysis}... |\n"
        
        if len(real_outages) > 20:
            map_section += f"\n*... and {len(real_outages) - 20} more real outages*\n"
    
    # False positives table
    if false_positives:
        map_section += "\n## ❌ False Positives - Geographic Details\n\n"
        map_section += "| Time | Customers Claimed | Latitude | Longitude | False Positive Reason |\n"
        map_section += "|------|-------------------|----------|-----------|----------------------|\n"
        
        for outage in false_positives[:20]:  # Limit to first 20 for readability
            reason = outage.get('claude_analysis', 'No analysis')[:100].replace('\n', ' ').replace('|', '\|')
            map_section += f"| {outage['datetime'][:16]} | {outage['customers']} | {outage['latitude']:.4f} | {outage['longitude']:.4f} | {reason}... |\n"
        
        if len(false_positives) > 20:
            map_section += f"\n*... and {len(false_positives) - 20} more false positives*\n"
    
    # Map visualization note
    map_section += "\n## 📍 Map Visualization Notes\n\n"
    map_section += """**Legend:**
- 🔴 Red markers: Confirmed real outages (larger markers indicate more customers affected)
- 🔵 Blue markers: False positive reports (filtered out from actual impact)
- Marker size is proportional to customer count claimed in the original report

**Geographic Patterns:**
- Real outages may cluster around areas with severe weather conditions
- False positives often appear in areas with mild weather or equipment sensor issues
- Use the coordinates above to recreate the map visualization in your preferred mapping tool

**For Interactive Map:**
To view the interactive map with clickable markers, use the web application interface.
"""
    
    return map_section

# ==================== UI FUNCTIONS ====================
def display_validation_results():
    """Display validation results with proper filtering - THE MISSING FUNCTION!"""
    st.header("🔍 Validation Results - Real vs False Positive")
    validation_results = st.session_state.analysis_state.get('validation_results', {})
    if not validation_results:
        st.warning("⚠️ No validation results available")
        return
    stats = validation_results.get('statistics', {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Reports", 
            f"{stats.get('real_count', 0) + stats.get('false_positive_count', 0):,}",
            help="Original number of outage reports"
        )
    with col2:
        st.metric(
            "Real Outages", 
            f"{stats.get('real_count', 0):,}",
            delta=f"-{stats.get('false_positive_count', 0)} false positives",
            delta_color="normal",
            help="Confirmed outages after validation"
        )
    with col3:
        st.metric(
            "Actual Customers Affected", 
            f"{stats.get('total_customers_actually_affected', 0):,}",
            delta=f"-{stats.get('customer_impact_reduction', 0):,} filtered out",
            delta_color="normal",
            help="Customers affected by real outages only"
        )
    with col4:
        st.metric(
            "False Positive Rate", 
            f"{stats.get('false_positive_rate', 0):.1f}%",
            help="Percentage of reports that were false positives"
        )
    st.subheader("📊 Validation Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **Real Outages Found:**
        - Count: {stats.get('real_count', 0):,}
        - Customers Affected: {stats.get('total_customers_actually_affected', 0):,}
        """)
    with col2:
        st.warning(f"""
        **False Positives Filtered:**
        - Count: {stats.get('false_positive_count', 0):,}
        - Customer Claims Removed: {stats.get('customer_impact_reduction', 0):,}
        """)
    st.divider()
    
    # Report mode selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        report_mode = st.selectbox(
            "📊 Report Detail Level",
            ["Default", "Exhaustive"],
            help="Default: Standard summary report\nExhaustive: Detailed explanations for every decision with full transparency"
        )
        
        if st.button("📋 Generate & Download Report", type="primary", use_container_width=True):
            generate_and_download_report(report_mode)
    # Always show download buttons if report is available
    if 'report_markdown' in st.session_state and 'report_pdf' in st.session_state and 'report_timestamp' in st.session_state:
        report_type = st.session_state.get('report_type', 'standard')
        report_label = "Exhaustive" if report_type == "exhaustive" else "Standard"
        st.info(f"⬇️ Download your latest {report_label} report below:")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label=f"📄 Download {report_label} Report (Markdown)",
                data=st.session_state['report_markdown'],
                file_name=f"outage_analysis_{report_type}_report_{st.session_state['report_timestamp']}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label=f"📋 Download {report_label} Report (PDF)",
                data=st.session_state['report_pdf'],
                file_name=f"outage_analysis_{report_type}_report_{st.session_state['report_timestamp']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    st.subheader("📊 Detailed Outage Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Real Outages")
        real_outages = validation_results.get('real_outages', [])
        if real_outages:
            real_df = pd.DataFrame(real_outages)
            
            # Add location names with persistent caching
            if 'location_name' not in real_df.columns:
                st.info("🔄 Fetching location names for real outages...")
                geocoding_service = GeocodingService()
                
                location_names = []
                progress_bar = st.progress(0)
                
                for idx, row in real_df.iterrows():
                    progress_bar.progress((idx + 1) / len(real_df))
                    try:
                        location_info = geocoding_service.get_location_name(
                            lat=row['latitude'], 
                            lon=row['longitude']
                        )
                        location_names.append(location_info['display_name'])
                    except Exception as e:
                        location_names.append(f"Near {row['latitude']:.3f}, {row['longitude']:.3f}")
                
                real_df['location_name'] = location_names
                
                # IMPORTANT: Update the cached validation results permanently
                real_outages_with_locations = []
                for i, outage in enumerate(real_outages):
                    outage_copy = outage.copy()
                    outage_copy['location_name'] = location_names[i]
                    # Get full location info from geocoding cache
                    full_location = geocoding_service.get_location_name(outage['latitude'], outage['longitude'])
                    outage_copy['city'] = full_location['city']
                    outage_copy['county'] = full_location['county']
                    outage_copy['state'] = full_location['state']
                    real_outages_with_locations.append(outage_copy)
                
                # Update the validation cache permanently
                st.session_state.analysis_state['validation_results']['real_outages'] = real_outages_with_locations
                
                # Save to disk cache so it persists across app restarts
                try:
                    cache_dir = "./cache"
                    validation_cache_file = os.path.join(cache_dir, "validation_results.joblib")
                    updated_validation_results = st.session_state.analysis_state['validation_results']
                    joblib.dump(updated_validation_results, validation_cache_file)
                    logger.info("💾 Updated validation cache with real outage location names")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update validation cache: {str(e)}")
                
                progress_bar.empty()
                st.success("✅ Location names retrieved and cached permanently!")
            
            # Show key columns with full data
            display_cols = ['datetime', 'customers', 'location_name', 'latitude', 'longitude']
            if 'claude_analysis' in real_df.columns:
                # Add truncated analysis for better readability
                real_df['analysis_summary'] = real_df['claude_analysis'].apply(
                    lambda x: x[:100] + '...' if len(str(x)) > 100 else str(x)
                )
                display_cols.append('analysis_summary')
            
            # Dynamic table height based on data size
            table_height = min(500, max(200, len(real_df) * 35 + 50))  # 35px per row + header
            st.dataframe(
                real_df[display_cols], 
                use_container_width=True,
                height=table_height,
                column_config={
                    "datetime": st.column_config.TextColumn("Date/Time"),
                    "customers": st.column_config.NumberColumn("Customers", format="%d"),
                    "location_name": st.column_config.TextColumn("Location"),
                    "latitude": st.column_config.NumberColumn("Latitude", format="%.4f"),
                    "longitude": st.column_config.NumberColumn("Longitude", format="%.4f"),
                    "analysis_summary": st.column_config.TextColumn("Weather Analysis")
                }
            )
            st.caption(f"Total: {len(real_outages)} confirmed real outages")
        else:
            st.info("No real outages found")
    
    with col2:
        st.markdown("### ❌ False Positives")
        false_positives = validation_results.get('false_positives', [])
        if false_positives:
            false_df = pd.DataFrame(false_positives)
            
            # Add location names with persistent caching
            if 'location_name' not in false_df.columns:
                st.info("🔄 Fetching location names for false positives...")
                if 'geocoding_service' not in locals():
                    geocoding_service = GeocodingService()
                
                location_names = []
                progress_bar = st.progress(0)
                
                for idx, row in false_df.iterrows():
                    progress_bar.progress((idx + 1) / len(false_df))
                    try:
                        location_info = geocoding_service.get_location_name(
                            lat=row['latitude'], 
                            lon=row['longitude']
                        )
                        location_names.append(location_info['display_name'])
                    except Exception as e:
                        location_names.append(f"Near {row['latitude']:.3f}, {row['longitude']:.3f}")
                
                false_df['location_name'] = location_names
                
                # IMPORTANT: Update the cached validation results permanently
                false_positives_with_locations = []
                for i, outage in enumerate(false_positives):
                    outage_copy = outage.copy()
                    outage_copy['location_name'] = location_names[i]
                    # Get full location info from geocoding cache  
                    full_location = geocoding_service.get_location_name(outage['latitude'], outage['longitude'])
                    outage_copy['city'] = full_location['city']
                    outage_copy['county'] = full_location['county']
                    outage_copy['state'] = full_location['state']
                    false_positives_with_locations.append(outage_copy)
                
                # Update the validation cache permanently
                st.session_state.analysis_state['validation_results']['false_positives'] = false_positives_with_locations
                
                # Save to disk cache so it persists across app restarts
                try:
                    cache_dir = "./cache"
                    validation_cache_file = os.path.join(cache_dir, "validation_results.joblib")
                    updated_validation_results = st.session_state.analysis_state['validation_results']
                    joblib.dump(updated_validation_results, validation_cache_file)
                    logger.info("💾 Updated validation cache with false positive location names")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update validation cache: {str(e)}")
                
                progress_bar.empty()
                st.success("✅ False positive location names retrieved and cached permanently!")
            
            # Show key columns with full data
            display_cols = ['datetime', 'customers', 'location_name', 'latitude', 'longitude']
            if 'claude_analysis' in false_df.columns:
                # Add truncated analysis for better readability  
                false_df['false_positive_reason'] = false_df['claude_analysis'].apply(
                    lambda x: x[:100] + '...' if len(str(x)) > 100 else str(x)
                )
                display_cols.append('false_positive_reason')
            
            # Dynamic table height based on data size
            table_height = min(500, max(200, len(false_df) * 35 + 50))  # 35px per row + header
            st.dataframe(
                false_df[display_cols], 
                use_container_width=True,
                height=table_height,
                column_config={
                    "datetime": st.column_config.TextColumn("Date/Time"),
                    "customers": st.column_config.NumberColumn("Customers", format="%d"),
                    "location_name": st.column_config.TextColumn("Location"),
                    "latitude": st.column_config.NumberColumn("Latitude", format="%.4f"),
                    "longitude": st.column_config.NumberColumn("Longitude", format="%.4f"),
                    "false_positive_reason": st.column_config.TextColumn("Why False Positive?")
                }
            )
            st.caption(f"Total: {len(false_positives)} false positive reports filtered")
        else:
            st.info("No false positives found")

def generate_static_map_image(validation_results):
    """Generate a static PNG map image of outages using matplotlib."""
    real_outages = validation_results.get('real_outages', [])
    false_positives = validation_results.get('false_positives', [])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot real outages
    if real_outages:
        lats = [o['latitude'] for o in real_outages]
        lons = [o['longitude'] for o in real_outages]
        sizes = [max(30, min(200, o['customers'])) for o in real_outages]
        ax.scatter(lons, lats, s=sizes, c='red', alpha=0.7, label='Real Outages', edgecolors='k')
    
    # Plot false positives
    if false_positives:
        lats = [o['latitude'] for o in false_positives]
        lons = [o['longitude'] for o in false_positives]
        sizes = [max(20, min(100, o['customers'])) for o in false_positives]
        ax.scatter(lons, lats, s=sizes, c='blue', alpha=0.5, label='False Positives', edgecolors='k')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Outage Locations')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_pdf_report(report_content, validation_results, raw_summary):
    """Generate PDF report using ReportLab, now with map image."""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from io import BytesIO
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Power Outage Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Executive Summary
        stats = validation_results.get('statistics', {})
        summary_data = [
            ['Metric', 'Value'],
            ['Total Reports', f"{stats.get('real_count', 0) + stats.get('false_positive_count', 0):,}"],
            ['Real Outages', f"{stats.get('real_count', 0):,}"],
            ['False Positives', f"{stats.get('false_positive_count', 0):,}"],
            ['False Positive Rate', f"{stats.get('false_positive_rate', 0):.1f}%"],
            ['Customers Actually Affected', f"{stats.get('total_customers_actually_affected', 0):,}"],
            ['Customer Claims Filtered', f"{stats.get('customer_impact_reduction', 0):,}"]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(summary_table)
        story.append(Spacer(1, 12))
        
        # Convert markdown report to PDF paragraphs (simplified)
        lines = report_content.split('\n')
        for line in lines:
            if line.strip():
                if line.startswith('#'):
                    # Header
                    header_text = line.replace('#', '').strip()
                    story.append(Paragraph(header_text, styles['Heading2']))
                elif line.startswith('-') or line.startswith('*'):
                    # Bullet point
                    bullet_text = line.replace('-', '').replace('*', '').strip()
                    story.append(Paragraph(f"• {bullet_text}", styles['Normal']))
                else:
                    # Normal text
                    story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 6))
        
        # False Positives Table
        false_positives = validation_results.get('false_positives', [])
        if false_positives:
            story.append(PageBreak())
            story.append(Paragraph("False Positive Details", styles['Heading2']))
            
            fp_data = [['Time', 'Customers', 'Location', 'Reason (First 100 chars)']]
            for fp in false_positives[:20]:  # Limit to first 20 for PDF
                reason = fp.get('claude_analysis', 'No analysis')[:100] + '...' if len(fp.get('claude_analysis', '')) > 100 else fp.get('claude_analysis', 'No analysis')
                fp_data.append([
                    fp['datetime'][:16],  # Just date and hour
                    str(fp['customers']),
                    f"{fp['latitude']:.3f}, {fp['longitude']:.3f}",
                    reason
                ])
            
            fp_table = Table(fp_data, colWidths=[1.5*inch, 0.8*inch, 1.5*inch, 3*inch])
            fp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            
            story.append(fp_table)
        
        # Add map data to PDF
        story.append(PageBreak())
        story.append(Paragraph("Geographic Distribution Analysis", styles['Heading2']))
        
        # Map summary table
        map_data = generate_map_data_summary(validation_results)
        if "error" not in map_data:
            bounds = map_data["geographic_bounds"]
            center = map_data["center_point"]
            
            map_summary_data = [
                ['Geographic Metric', 'Value'],
                ['Total Locations', str(map_data['total_points'])],
                ['Real Outages', str(map_data['real_outages_count'])],
                ['False Positives', str(map_data['false_positives_count'])],
                ['North Bound', f"{bounds['north']:.4f}°"],
                ['South Bound', f"{bounds['south']:.4f}°"],
                ['East Bound', f"{bounds['east']:.4f}°"],
                ['West Bound', f"{bounds['west']:.4f}°"],
                ['Geographic Center', f"{center['latitude']:.4f}°, {center['longitude']:.4f}°"]
            ]
            
            map_summary_table = Table(map_summary_data, colWidths=[2.5*inch, 2.5*inch])
            map_summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(map_summary_table)
            story.append(Spacer(1, 12))
        
        # Embed static map image
        try:
            map_img_buf = generate_static_map_image(validation_results)
            img = RLImage(map_img_buf, width=5*inch, height=5*inch)
            story.append(Spacer(1, 12))
            story.append(Paragraph("Outage Locations Map", styles['Heading3']))
            story.append(img)
            story.append(Spacer(1, 12))
        except Exception as e:
            story.append(Paragraph(f"[Map image could not be generated: {str(e)}]", styles['Normal']))
        
        # Real outages geographic table
        real_outages = validation_results.get('real_outages', [])
        if real_outages:
            story.append(Paragraph("Real Outages - Geographic Coordinates", styles['Heading3']))
            
            real_geo_data = [['Time', 'Customers', 'Latitude', 'Longitude']]
            for outage in real_outages[:15]:  # Limit for PDF space
                real_geo_data.append([
                    outage['datetime'][:16],
                    str(outage['customers']),
                    f"{outage['latitude']:.4f}",
                    f"{outage['longitude']:.4f}"
                ])
            
            real_geo_table = Table(real_geo_data, colWidths=[1.8*inch, 0.8*inch, 1.2*inch, 1.2*inch])
            real_geo_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightpink),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(real_geo_table)
            story.append(Spacer(1, 12))
            
            if len(real_outages) > 15:
                story.append(Paragraph(f"... and {len(real_outages) - 15} more real outages", styles['Normal']))
                story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        st.error("❌ PDF generation requires reportlab. Please install: pip install reportlab")
        return None
    except Exception as e:
        logger.error(f"❌ PDF generation error: {str(e)}")
        st.error(f"❌ PDF generation failed: {str(e)}")
        return None

def generate_and_download_report(report_mode="Default"):
    """Generate report and store in session state for persistent download buttons."""
    try:
        validation_results = st.session_state.analysis_state.get('validation_results', {})
        raw_summary = st.session_state.analysis_state.get('raw_dataset_summary', {})
        
        if not validation_results or not raw_summary:
            st.error("❌ Cannot generate report: Missing validation data")
            return
        
        spinner_text = "🤖 Generating comprehensive report..." if report_mode == "Default" else "🔍 Generating exhaustive report with detailed explanations..."
        with st.spinner(spinner_text):
            # Generate the report using the LLM tool
            if report_mode == "Exhaustive":
                report_content = generate_exhaustive_report.invoke({
                    "validation_results": validation_results,
                    "raw_summary": raw_summary
                })
            else:
                report_content = generate_comprehensive_report.invoke({
                    "validation_results": validation_results,
                    "raw_summary": raw_summary
                })
            
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            report_type = "exhaustive" if report_mode == "Exhaustive" else "standard"
            
            # Generate PDF
            pdf_data = generate_pdf_report(report_content, validation_results, raw_summary)
            
            # Store in session state for persistent download
            st.session_state['report_markdown'] = report_content
            st.session_state['report_pdf'] = pdf_data
            st.session_state['report_timestamp'] = timestamp
            st.session_state['report_type'] = report_type
            
            if report_mode == "Exhaustive":
                st.success("✅ Exhaustive report generated successfully! This report includes detailed explanations for every decision made by the agent. Download options below.")
            else:
                st.success("✅ Standard report generated successfully! Download options below.")
    except Exception as e:
        logger.error(f"❌ Report generation failed: {str(e)}")
        st.error(f"❌ Report generation failed: {str(e)}")

def display_map_for_report(validation_results):
    """Display a simplified map for report purposes with zoom reset"""
    real_outages = validation_results.get('real_outages', [])
    false_positives = validation_results.get('false_positives', [])
    
    if not real_outages and not false_positives:
        st.warning("No outage data to display on map")
        return
    
    # Add zoom reset button for report map
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**📍 Map Legend:** 🔴 Real Outages | 🔵 False Positives")
    with col2:
        report_zoom_reset = st.button("🔍 Reset Zoom", help="Reset map view", key="zoom_reset_report")
    
    # Calculate map center and smart zoom
    all_outages = real_outages + false_positives
    if all_outages:
        lats = [o['latitude'] for o in all_outages]
        lons = [o['longitude'] for o in all_outages]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Smart zoom calculation
        if report_zoom_reset or 'report_map_zoom' not in st.session_state:
            lat_range = max(lats) - min(lats)
            lon_range = max(lons) - min(lons)
            max_range = max(lat_range, lon_range)
            
            if max_range < 0.01:
                zoom_level = 13
            elif max_range < 0.1:
                zoom_level = 10
            elif max_range < 1:
                zoom_level = 7
            else:
                zoom_level = 5
                
            st.session_state.report_map_zoom = zoom_level
        else:
            zoom_level = st.session_state.report_map_zoom
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level)
        
        # Add real outages in red (larger markers)
        for outage in real_outages:
            folium.CircleMarker(
                location=[outage['latitude'], outage['longitude']],
                radius=max(8, min(20, outage['customers'] * 0.015)),
                popup=f"REAL OUTAGE<br>Time: {outage['datetime']}<br>Customers: {outage['customers']}",
                tooltip=f"Real Outage: {outage['customers']} customers",
                color='darkred',
                fillColor='red',
                fillOpacity=0.8,
                weight=2
            ).add_to(m)
        
        # Add false positives in blue (smaller markers)
        for outage in false_positives:
            folium.CircleMarker(
                location=[outage['latitude'], outage['longitude']],
                radius=max(6, min(15, outage['customers'] * 0.01)),  # Slightly larger for visibility
                popup=f"FALSE POSITIVE<br>Time: {outage['datetime']}<br>Customers Claimed: {outage['customers']}<br>Reason: {outage.get('claude_analysis', 'Analysis unavailable')[:100]}...",
                tooltip=f"False Positive: {outage['customers']} customers claimed",
                color='blue',
                fillColor='lightblue',
                fillOpacity=0.7,  # More opaque for better visibility
                weight=2
            ).add_to(m)
        
        # Map is ready for display
        
        # Dynamic sizing for report map
        total_points = len(real_outages) + len(false_positives)
        report_height = max(250, min(400, total_points * 8))  # Scale with data
        st_folium(m, use_container_width=True, height=report_height)

def display_map_visualization():
    """Display map with real outages vs false positives - Interactive Overview"""
    validation_results = st.session_state.analysis_state.get('validation_results', {})
    if not validation_results:
        return
    real_outages = validation_results.get('real_outages', [])
    false_positives = validation_results.get('false_positives', [])
    if not real_outages and not false_positives:
        st.warning("No outage data to display on map")
        return
    # Enhanced controls above the map
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        show_real = st.checkbox("Show Real Outages", value=True, key="show_real_map")
    with col2:
        show_false = st.checkbox("Show False Positives", value=True, key="show_false_map")
    with col3:
        zoom_reset = st.button("🔍 Reset Zoom", help="Reset map to show all outages", key="zoom_reset_main")
    
    # Map header with legend
    st.subheader("🗺️ Interactive Outage Map")
    
    # Simple legend above map
    legend_col1, legend_col2 = st.columns(2)
    with legend_col1:
        st.markdown("🔴 **Real Outages** (Weather-confirmed)")
    with legend_col2:
        st.markdown("🔵 **False Positives** (Filtered out)")
    # Calculate map center and bounds
    all_outages = real_outages + false_positives
    if all_outages:
        lats = [o['latitude'] for o in all_outages]
        lons = [o['longitude'] for o in all_outages]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Reset zoom if button was clicked or use smart zoom level
        if zoom_reset or 'main_map_zoom' not in st.session_state:
            # Calculate appropriate zoom level based on data spread
            lat_range = max(lats) - min(lats)
            lon_range = max(lons) - min(lons)
            max_range = max(lat_range, lon_range)
            
            if max_range < 0.01:
                zoom_level = 14
            elif max_range < 0.1:
                zoom_level = 11
            elif max_range < 1:
                zoom_level = 8
            else:
                zoom_level = 6
                
            st.session_state.main_map_zoom = zoom_level
        else:
            zoom_level = st.session_state.main_map_zoom
        
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=zoom_level,
            tiles='OpenStreetMap'
        )
        # Add real outages in red (only if checkbox is checked)
        if show_real:
            for outage in real_outages:
                popup_content = f"""
                <b>REAL OUTAGE</b><br>
                📅 Time: {outage['datetime']}<br>
                👥 Customers: {outage['customers']}<br>
                📍 Location: {outage['latitude']:.4f}, {outage['longitude']:.4f}<br>
                <details>
                <summary>🌤️ Weather Analysis</summary>
                <small>{outage.get('claude_analysis', 'Analysis unavailable')[:200]}...</small>
                </details>
                """
                folium.CircleMarker(
                    location=[outage['latitude'], outage['longitude']],
                    radius=max(6, min(18, outage['customers'] * 0.012)),
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"Real Outage: {outage['customers']} customers",
                    color='darkred',
                    fillColor='red',
                    fillOpacity=0.8,
                    weight=2
                ).add_to(m)
        # Add false positives in blue (only if checkbox is checked)
        if show_false:
            for outage in false_positives:
                popup_content = f"""
                <b>FALSE POSITIVE</b><br>
                📅 Time: {outage['datetime']}<br>
                👥 Customers Claimed: {outage['customers']}<br>
                📍 Location: {outage['latitude']:.4f}, {outage['longitude']:.4f}<br>
                <details>
                <summary>❌ Why False Positive?</summary>
                <small>{outage.get('claude_analysis', 'Analysis unavailable')[:200]}...</small>
                </details>
                """
                folium.CircleMarker(
                    location=[outage['latitude'], outage['longitude']],
                    radius=max(6, min(15, outage['customers'] * 0.01)),
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"False Positive: {outage['customers']} customers claimed",
                    color='blue',
                    fillColor='lightblue',
                    fillOpacity=0.8,
                    weight=2
                ).add_to(m)
        # Dynamic and responsive map sizing
        num_outages = len(all_outages)
        
        # Calculate responsive height based on viewport and data
        # Use percentage of viewport height for better responsive design
        if num_outages < 10:
            # Small datasets - compact view
            map_height = "40vh"  # 40% of viewport height
            map_height_px = 350  # Fallback pixels
        elif num_outages < 50:
            # Medium datasets - balanced view
            map_height = "50vh"  # 50% of viewport height
            map_height_px = 450  # Fallback pixels
        else:
            # Large datasets - full view
            map_height = "60vh"  # 60% of viewport height
            map_height_px = 550  # Fallback pixels
        
        # Create a custom container with no extra padding
        with st.container():
            # Use CSS to eliminate gaps and make fully responsive
            st.markdown(
                f"""
                <style>
                .stContainer > div:last-child {{
                    margin-bottom: 0 !important;
                    padding-bottom: 0 !important;
                }}
                div[data-testid="stDecoration"] {{
                    display: none;
                }}
                </style>
                """, 
                unsafe_allow_html=True
            )
            
            # Render map with dynamic sizing
            try:
                map_data = st_folium(
                    m, 
                    use_container_width=True, 
                    height=map_height_px,  # Fallback to pixels for compatibility
                    returned_objects=["last_object_clicked"]
                )
            except:
                # Fallback rendering without advanced features
                st_folium(m, use_container_width=True, height=map_height_px)

def display_chat_interface():
    """Display streamlined chat interface for validation results"""
    
    st.header("💬 Ask Claude About Results")
    
    # Check if we have validation results
    validation_results = st.session_state.analysis_state.get('validation_results', {})
    if not validation_results.get('real_outages') and not validation_results.get('false_positives'):
        st.info("💡 Complete validation analysis to enable chat functionality")
        return
    
    # Display recent chat (max 4 exchanges to keep clean)
    if st.session_state.chat_history:
        st.subheader("Recent Conversation")
        recent_chats = st.session_state.chat_history[-8:]  # Last 4 exchanges (user + assistant pairs)
        
        for chat in recent_chats:
            if chat['role'] == 'user':
                st.markdown(f"**🙋 You:** {chat['content']}")
            else:
                st.markdown(f"**🤖 Claude:** {chat['content']}")
        
        if len(st.session_state.chat_history) > 8:
            st.caption(f"... {len(st.session_state.chat_history) - 8} earlier messages")
    
    # Streamlined input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask about validation results:",
            placeholder="e.g., 'Why were reports false positives?' or 'What caused real outages?'",
            key="chat_input"
        )
    
    with col2:
        ask_button = st.button("💭 Ask", type="primary", use_container_width=True)
    
    # Quick questions as buttons in a row
    st.markdown("**Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_questions = {
        "❌ False Positives": "Why were these reports classified as false positives? What were the main reasons?",
        "⚡ Real Causes": "What weather conditions caused the confirmed real outages?", 
        "📊 Summary": "Provide a summary of the validation results and key recommendations.",
        "💰 Savings": "What are the cost savings from filtering out false positives?"
    }
    
    for i, (button_text, question) in enumerate(quick_questions.items()):
        with [col1, col2, col3, col4][i]:
            if st.button(button_text, use_container_width=True):
                user_question = question
                ask_button = True
    
    # Process question
    if ask_button and user_question:
        with st.spinner("🤖 Analyzing..."):
            try:
                context = st.session_state.analysis_state.get('chat_context', {})
                
                response = chat_about_results.invoke({
                    "question": user_question,
                    "context": context
                })
                
                # Add to history and keep it manageable
                st.session_state.chat_history.extend([
                    {'role': 'user', 'content': user_question},
                    {'role': 'assistant', 'content': response}
                ])
                
                # Keep only last 20 messages to prevent memory issues
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Chat error: {str(e)}")
    
    # Clear chat - moved to sidebar to keep main area clean
    if st.session_state.chat_history:
        with st.sidebar:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

def display_llm_usage_monitoring():
    """Display LLM usage monitoring statistics in the sidebar."""
    st.subheader("📈 LLM Usage Monitoring")

    usage_log_file = "llm_usage.log"

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Usage Stats", use_container_width=True):
            st.rerun()

    if not os.path.exists(usage_log_file):
        st.info("No usage data recorded yet. Run some analysis to see stats.")
        return

    try:
        with open(usage_log_file, "r") as f:
            lines = f.readlines()

        if not lines:
            st.info("No usage data recorded yet.")
            return

        usage_data = []
        for line in lines:
            try:
                if line.strip():
                    usage_data.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping corrupted line in usage log: {line.strip()}")
                continue
        
        if not usage_data:
            st.info("No valid usage data recorded yet.")
            return
            
        df = pd.DataFrame(usage_data)

        # Download button
        with col2:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name="llm_usage_report.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Summary Stats
        total_cost = df["cost"].sum()
        total_tokens = df["total_tokens"].sum()
        avg_latency = df["duration_seconds"].mean()

        st.metric("💰 Total Estimated Cost", f"${total_cost:.4f}")
        st.metric("🪙 Total Tokens Used", f"{total_tokens:,}")
        st.metric("⏱️ Avg. Latency (sec)", f"{avg_latency:.2f}s")

        with st.expander("Detailed Usage Log"):
            st.dataframe(df[[
                "timestamp", "model_name", "total_tokens", "cost", "duration_seconds"
            ]], use_container_width=True)

    except Exception as e:
        st.error(f"Error reading usage log: {e}")

# Add this function to your main.py

def display_cost_projections():
    """Display cost projections and model comparisons in sidebar"""
    from cost_analyzer import CostAnalyzer
    
    st.subheader("💰 Cost Projections")
    
    try:
        cost_analyzer = CostAnalyzer()
        
        # Quick scenario buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Usage Scenarios", use_container_width=True):
                with st.spinner("Calculating projections..."):
                    scenarios = cost_analyzer.calculate_usage_scenarios()
                    
                    if "error" not in scenarios:
                        st.write("**Your Scenario (10 reports/day, 2hr chat):**")
                        your_scenario = scenarios.get('your_scenario', {})
                        if your_scenario:
                            st.metric("Monthly Cost", f"${your_scenario['monthly_cost']:.2f}")
                            st.metric("Daily Cost", f"${your_scenario['daily_cost']:.4f}")
                        
                        with st.expander("All Scenarios"):
                            for name, data in scenarios.items():
                                st.write(f"**{name.replace('_', ' ').title()}:** ${data['monthly_cost']:.2f}/month")
                    else:
                        st.info("Run some operations first to get projections")
        
        with col2:
            if st.button("🤖 Model Comparison", use_container_width=True):
                with st.spinner("Comparing models..."):
                    comparisons = cost_analyzer.compare_models_for_scenario()
                    
                    st.write("**Monthly costs for your usage:**")
                    for model, data in comparisons.items():
                        st.write(f"• {model}: ${data['monthly_cost']:.2f}")
        
        # Full report download
        if st.button("📄 Download Full Report", use_container_width=True):
            report = cost_analyzer.generate_cost_report()
            st.download_button(
                label="💾 Download Cost Report",
                data=report,
                file_name=f"cost_analysis_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    except Exception as e:
        st.error(f"Cost analysis error: {str(e)}")

# Add this call in your main() function sidebar, after display_llm_usage_monitoring():
# st.divider()
# display_cost_projections()

# ==================== MAIN APPLICATION ====================
def main():
    """Main Streamlit application"""
    
    # Additional Streamlit config to prevent file watcher issues
    try:
        st.set_page_config(
            page_title="Outage Analysis Agent - 2025",
            page_icon="🤖⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        logger.warning(f"Page config warning (can be ignored): {str(e)}")
        # Continue anyway as this is non-critical
    
    # Initialize session state
    if 'analysis_state' not in st.session_state:
        st.session_state.analysis_state = {
            'dataset_loaded': False,
            'validation_complete': False,
            'raw_dataset_summary': {},
            'validation_results': {},
            'filtered_summary': {},
            'current_window_analysis': None,
            'chat_context': {},
            'errors': []
        }
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Always try to load from cache first (cache-first approach)
    cache_dir = "./cache"
    cache_file = os.path.join(cache_dir, "outage_data_summary.joblib")
    vector_cache_file = os.path.join(cache_dir, "vector_db_status.json")
    validation_cache_file = os.path.join(cache_dir, "validation_results.joblib")
    
    # Initialize cache info for UI display
    st.session_state.cache_info = {
        'exists': False,
        'last_updated': None,
        'record_count': 0
    }
    
    if os.path.exists(cache_file) and os.path.exists(vector_cache_file):
        try:
            logger.info("🔄 Loading from cache (cache-first approach)")
            cached_summary = joblib.load(cache_file)
            
            # Load cache metadata
            with open(vector_cache_file, 'r') as f:
                cache_metadata = json.load(f)
            
            st.session_state.cache_info = {
                'exists': True,
                'last_updated': datetime.fromisoformat(cache_metadata.get('timestamp', '')),
                'record_count': cache_metadata.get('record_count', 0)
            }
            
            # Load validation results if available
            cached_validation_results = {}
            validation_was_completed = False
            
            if os.path.exists(validation_cache_file):
                try:
                    cached_validation_results = joblib.load(validation_cache_file)
                    validation_was_completed = True
                    logger.info("✅ Complete cached results loaded (data + validation)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load cached validation results: {str(e)}")
            
            if not validation_was_completed:
                # Create empty validation structure for display
                cached_validation_results = {
                    'total_reports': cached_summary.get('total_reports', 0),
                    'real_outages': [],
                    'false_positives': [],
                    'validation_errors': [],
                    'statistics': {
                        'real_count': 0,
                        'false_positive_count': 0,
                        'false_positive_rate': 0,
                        'total_customers_actually_affected': 0,
                        'total_customers_claimed': cached_summary.get('raw_customer_claims', {}).get('total_claimed', 0),
                        'customer_impact_reduction': 0
                    }
                }
                logger.info("📊 Raw data loaded from cache (validation pending)")
            
            # Update session state with cached data
            st.session_state.analysis_state.update({
                'dataset_loaded': True,
                'validation_complete': validation_was_completed,
                'raw_dataset_summary': cached_summary,
                'validation_results': cached_validation_results,
                'chat_context': {
                    'raw_summary': cached_summary,
                    'validation_results': cached_validation_results
                }
            })
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cached data: {str(e)}")
            st.session_state.cache_info['exists'] = False
    
    st.title("🤖⚡ Advanced Outage Analysis Agent")
    st.markdown("**Claude-powered outage validation with false positive detection using 2025 LangGraph patterns**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Initialize session state for LLM config on first run
        if 'llm_config' not in st.session_state:
            st.session_state.llm_config = {
                "via": os.getenv("LLM_PROVIDER"),
                "claude_model": os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229"),
                "openai_model": os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
                "ollama_model": os.getenv("OLLAMA_MODEL", "llama3"),
            }

        # LLM Status Check and Model Selection
        st.subheader("🤖 LLM Selection")
        
        try:
            with open("pricing.json", "r") as f:
                pricing_data = json.load(f)
            claude_models = sorted([m for m in pricing_data if 'claude' in m])
            openai_models = sorted([m for m in pricing_data if 'gpt' in m])
            ollama_models = sorted([m for m in pricing_data if 'llama' in m])
        except (FileNotFoundError, json.JSONDecodeError):
            claude_models = ["claude-3-sonnet-20240229"]
            openai_models = ["gpt-4-turbo-preview"]
            ollama_models = ["llama3"]
        
        # Get current selections from session state to set dropdown indices
        current_provider = st.session_state.llm_config.get("via") or "claude"
        provider_options = ["claude", "openai", "ollama"]
        provider_index = provider_options.index(current_provider) if current_provider in provider_options else 0

        selected_provider = st.selectbox(
            "Select LLM Provider",
            provider_options,
            index=provider_index
        )

        model_changed = False
        if selected_provider == 'claude':
            current_model = st.session_state.llm_config.get("claude_model")
            model_index = claude_models.index(current_model) if current_model in claude_models else 0
            selected_model = st.selectbox("Select Claude Model", claude_models, index=model_index)
            if current_model != selected_model or current_provider != 'claude':
                st.session_state.llm_config['via'] = 'claude'
                st.session_state.llm_config['claude_model'] = selected_model
                model_changed = True

        elif selected_provider == 'openai':
            current_model = st.session_state.llm_config.get("openai_model")
            model_index = openai_models.index(current_model) if current_model in openai_models else 0
            selected_model = st.selectbox("Select OpenAI Model", openai_models, index=model_index)
            if current_model != selected_model or current_provider != 'openai':
                st.session_state.llm_config['via'] = 'openai'
                st.session_state.llm_config['openai_model'] = selected_model
                model_changed = True
        
        else:  # ollama
            current_model = st.session_state.llm_config.get("ollama_model")
            model_index = ollama_models.index(current_model) if current_model in ollama_models else 0
            selected_model = st.selectbox("Select Ollama Model", ollama_models, index=model_index)
            if current_model != selected_model or current_provider != 'ollama':
                st.session_state.llm_config['via'] = 'ollama'
                st.session_state.llm_config['ollama_model'] = selected_model
                model_changed = True
        
        if model_changed:
            st.info("Model selection changed. New analyses will use the new model.")
            # We don't need to rerun, the next tool call will pick up the new config
        
        # LLM Status Check
        try:
            # Use the configured LLMManager
            llm_manager = LLMManager(model_config=st.session_state.get('llm_config'))
            provider_info = llm_manager.get_provider_info()
            
            provider_name = provider_info.get('provider', 'Unknown')
            model_name = provider_info.get('model', 'Unknown')

            if 'Ollama' in str(llm_manager.llm):
                provider_name = 'Ollama'

            st.success(f"✅ {provider_name.capitalize()} Connected")
            st.caption(f"Model: {model_name}")

        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.info("💡 Add your API key to .env file or run Ollama.")
            return
        except Exception as e:
            st.error(f"❌ LLM Error: {str(e)}")
            return
        
        if MCP_AVAILABLE:
            st.success("✅ MCP Integration Available")
        else:
            st.warning("⚠️ MCP Not Available")
            
        # Cache Status Display
        st.subheader("💾 Cache Status")
        cache_info = st.session_state.get('cache_info', {'exists': False})
        
        if cache_info['exists']:
            last_update = cache_info['last_updated']
            record_count = cache_info['record_count']
            st.success(f"✅ Cache Active")
            st.write(f"**Last Updated:** {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Records:** {record_count:,}")
        else:
            st.warning("⚠️ No Cache Found")
            st.write("Click 'Update Cache' to load data from data/ folder")
        
        st.divider()
        
        # Update Cache Section
        st.header("🔄 Data Management")
        
        # Check for any CSV files in data folder
        data_folder = "/home/viresh/Documents/repo/power-agent/data/"
        csv_files = []
        if os.path.exists(data_folder):
            csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        
        if csv_files:
            if len(csv_files) == 1:
                data_file = os.path.join(data_folder, csv_files[0])
                st.info(f"📁 Data file found: `data/{csv_files[0]}`")
            else:
                st.info(f"📁 Found {len(csv_files)} CSV files in data folder")
                selected_file = st.selectbox(
                    "Select CSV file to process:",
                    csv_files,
                    help="Choose which CSV file to load and validate"
                )
                data_file = os.path.join(data_folder, selected_file)
            
            # Rate limiting controls
            with st.expander("⚙️ Processing Options", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    processing_speed = st.selectbox(
                        "Processing Speed",
                        ["Conservative (Slower, More Stable)", "Standard", "Fast (Higher Risk of Rate Limits)"],
                        index=1,
                        help="Conservative: 1s delay between requests\nStandard: 0.5s delay\nFast: 0.2s delay"
                    )
                with col2:
                    max_retries = st.number_input(
                        "Max Retries per Request",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="Number of times to retry if API is overloaded"
                    )
            
            if st.button("🔄 Update Cache from Data Folder", type="primary"):
                with st.spinner("🤖 Loading data and running validation..."):
                    try:
                        # Load CSV from data folder
                        df = pd.read_csv(data_file)
                        
                        # Validate required columns
                        required_columns = ['DATETIME', 'LATITUDE', 'LONGITUDE', 'CUSTOMERS']
                        if not all(col in df.columns for col in required_columns):
                            st.error(f"❌ Missing required columns: {required_columns}")
                            return
                        
                        st.write(f"📊 Loading {len(df)} records from data folder...")
                        
                        # Load into vector DB (force reload)
                        vector_db = OutageVectorDB()
                        raw_summary = vector_db.load_outage_data(df, force_reload=True)
                        
                        # Convert processing speed to delay
                        if "Conservative" in processing_speed:
                            delay = 1.0
                        elif "Standard" in processing_speed:
                            delay = 0.5
                        else:  # Fast
                            delay = 0.2
                        
                        # Validate all reports with user-selected rate limiting
                        validation_results = validate_all_reports(df, request_delay=delay, max_retries=max_retries)
                        
                        # Update session state
                        st.session_state.analysis_state.update({
                            'dataset_loaded': True,
                            'validation_complete': True,
                            'raw_dataset_summary': raw_summary,
                            'validation_results': validation_results,
                            'chat_context': {
                                'raw_summary': raw_summary,
                                'validation_results': validation_results
                            }
                        })
                        
                        # Update cache info
                        st.session_state.cache_info = {
                            'exists': True,
                            'last_updated': datetime.now(),
                            'record_count': len(df)
                        }
                        
                        st.success("✅ Cache updated successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Cache update failed: {str(e)}")
                        logger.error(f"Cache update error: {str(e)}")
        else:
            st.error(f"❌ Data file not found: `{data_file}`")
            st.write("Please ensure `raw_data.csv` exists in the `data/` folder")
        
        # Clear cache option
        if cache_info['exists']:
            st.divider()
            if st.button("🗑️ Clear Cache"):
                import shutil
                try:
                    cache_dir = "./cache"
                    if os.path.exists(cache_dir):
                        shutil.rmtree(cache_dir)
                    if os.path.exists("weather_cache.sqlite"):
                        os.remove("weather_cache.sqlite")
                    if os.path.exists("llm_usage.log"):
                        os.remove("llm_usage.log")
                    
                    # Reset session state
                    st.session_state.analysis_state = {
                        'dataset_loaded': False,
                        'validation_complete': False,
                        'raw_dataset_summary': {},
                        'validation_results': {},
                        'filtered_summary': {},
                        'current_window_analysis': None,
                        'chat_context': {},
                        'errors': []
                    }
                    st.session_state.cache_info = {'exists': False}
                    
                    st.success("✅ Cache cleared successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to clear cache: {str(e)}")
        
        # Show status
        if st.session_state.analysis_state['validation_complete']:
            st.success("✅ Validation Complete")
            validation_stats = st.session_state.analysis_state['validation_results'].get('statistics', {})
            st.write(f"**Real Outages:** {validation_stats.get('real_count', 0)}")
            st.write(f"**False Positives:** {validation_stats.get('false_positive_count', 0)}")
            st.write(f"**Accuracy:** {100 - validation_stats.get('false_positive_rate', 0):.1f}%")

        st.divider()
        display_llm_usage_monitoring()
        st.divider()
        display_cost_projections()
    # Main content area - Always show cache info first
    cache_info = st.session_state.get('cache_info', {'exists': False})
    if cache_info['exists']:
        last_update = cache_info['last_updated']
        st.info(f"💾 **Displaying results from cache** | Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')} | Records: {cache_info['record_count']:,}")
    
    # Always display available results
    if st.session_state.analysis_state['dataset_loaded']:
        # Show what we have - either full validation results or raw data summary
        if st.session_state.analysis_state['validation_complete']:
            # Full validation results available - show everything
            pass  # Will continue to display_validation_results() below
        else:
            # Show meaningful raw data visualization
            st.warning("📊 **Raw dataset from cache** | Click 'Update Cache' for weather validation analysis")
            
            raw_summary = st.session_state.analysis_state.get('raw_dataset_summary', {})
            if raw_summary:
                # Main metrics
                st.subheader("📈 Outage Reports Overview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Reports", f"{raw_summary.get('total_reports', 0):,}")
                with col2:
                    total_claimed = raw_summary.get('raw_customer_claims', {}).get('total_claimed', 0)
                    st.metric("Customers Claimed", f"{total_claimed:,}")
                with col3:
                    avg_claimed = raw_summary.get('raw_customer_claims', {}).get('avg_per_report', 0)
                    st.metric("Avg per Report", f"{avg_claimed:.1f}")
                with col4:
                    date_range = raw_summary.get('date_range', {})
                    if date_range.get('start') and date_range.get('end'):
                        st.metric("Date Range", f"{date_range['start']} to {date_range['end']}")
                
                # Geographic info
                st.subheader("🗺️ Geographic Coverage")
                geo_info = raw_summary.get('geographic_coverage', {})
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Latitude Range:** {geo_info.get('lat_range', 'N/A')}")
                    st.write(f"**Longitude Range:** {geo_info.get('lon_range', 'N/A')}")
                with col2:
                    center = geo_info.get('center', [])
                    if center:
                        st.write(f"**Geographic Center:** {center[0]:.3f}, {center[1]:.3f}")
                
                # Call-to-action
                st.info("💡 **For detailed analysis:** Click 'Update Cache from Data Folder' to run weather validation and see real vs false positive classifications.")
        
        # Continue to show other sections even with raw data
    else:
        st.warning("⚠️ No cached data found. Click 'Update Cache from Data Folder' to load and analyze data.")
        
        # Show example data format
        with st.expander("📋 Expected CSV Format"):
            st.code("""
DATETIME,LATITUDE,LONGITUDE,CUSTOMERS
2022-01-01 00:15:00,40.7128,-74.0060,150
2022-01-01 01:30:00,40.7589,-73.9851,200
2022-01-01 02:45:00,40.6892,-74.0445,75
            """)
        return
    
    # Display validation results
    display_validation_results()
    
    # Map visualization - positioned immediately after validation results
    display_map_visualization()
    
    # Advanced features when validation is complete - eliminate spacing gaps
    if st.session_state.analysis_state['validation_complete']:
        # Minimal separator with dynamic spacing
        st.markdown(
            """
            <style>
            .minimal-separator {
                margin: 0.5rem 0 !important;
                padding: 0 !important;
                border: none;
                border-top: 1px solid rgba(128, 128, 128, 0.3);
                height: 1px;
            }
            .stExpander {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
            </style>
            <hr class="minimal-separator">
            """, 
            unsafe_allow_html=True
        )
        with st.expander("🎯 Time Window Analysis", expanded=False):
            raw_summary = st.session_state.analysis_state.get('raw_dataset_summary', {})
            date_range = raw_summary.get('date_range', {})
            
            if date_range.get('start') and date_range.get('end'):
                min_date = datetime.strptime(date_range['start'], '%Y-%m-%d').date()
                max_date = datetime.strptime(date_range['end'], '%Y-%m-%d').date()
                
                # More compact layout for time selection
                st.write("**Select Time Window for Focused Analysis:**")
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    start_date = st.date_input("Start Date:", min_value=min_date, max_value=max_date, value=min_date, key="window_start_date")
                    start_time = st.time_input("Start Time:", value=time(0, 0), key="window_start_time")
                
                with col2:
                    end_date = st.date_input("End Date:", min_value=min_date, max_value=max_date, value=min_date, key="window_end_date")
                    end_time = st.time_input("End Time:", value=time(23, 59), key="window_end_time")
                
                with col3:
                    st.markdown("<br>", unsafe_allow_html=True)  # Better spacing
                    if st.button("🔍 Analyze Window", use_container_width=True):
                        start_datetime = datetime.combine(start_date, start_time)
                        end_datetime = datetime.combine(end_date, end_time)
                        analyze_time_window(start_datetime, end_datetime)
        # Chat interface - main feature when validation complete  
        st.markdown('<hr class="minimal-separator">', unsafe_allow_html=True)
        display_chat_interface()

def analyze_time_window(start_datetime: datetime, end_datetime: datetime):
    """Analyze specific time window with validation results"""
    
    with st.spinner(f"🤖 Analyzing {start_datetime} to {end_datetime}..."):
        try:
            validation_results = st.session_state.analysis_state.get('validation_results', {})
            
            if not validation_results:
                st.error("❌ No validation results available")
                return
            
            # Filter real outages and false positives by time window
            real_outages = validation_results.get('real_outages', [])
            false_positives = validation_results.get('false_positives', [])
            
            def in_time_window(outage_datetime_str):
                try:
                    outage_dt = datetime.strptime(outage_datetime_str, '%Y-%m-%d %H:%M:%S')
                    return start_datetime <= outage_dt <= end_datetime
                except:
                    return False
            
            # Filter outages in time window
            window_real = [o for o in real_outages if in_time_window(o['datetime'])]
            window_false = [o for o in false_positives if in_time_window(o['datetime'])]
            
            total_in_window = len(window_real) + len(window_false)
            
            if total_in_window == 0:
                st.warning(f"⚠️ No outages found in the selected time window")
                return
            
            st.success(f"✅ Found {total_in_window} reports in time window ({len(window_real)} real, {len(window_false)} false)")
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Reports", total_in_window)
            with col2:
                st.metric("Real Outages", len(window_real))
            with col3:
                st.metric("False Positives", len(window_false))
            with col4:
                real_customers = sum(o['customers'] for o in window_real)
                st.metric("Actual Customers Affected", real_customers)
            
            # Show breakdown
            if window_real or window_false:
                col1, col2 = st.columns(2)
                
                with col1:
                    if window_real:
                        st.subheader("✅ Real Outages in Window")
                        real_df = pd.DataFrame(window_real)
                        st.dataframe(real_df[['datetime', 'customers', 'latitude', 'longitude']])
                
                with col2:
                    if window_false:
                        st.subheader("❌ False Positives in Window")
                        false_df = pd.DataFrame(window_false)
                        st.dataframe(false_df[['datetime', 'customers', 'latitude', 'longitude']])
            
            # Update chat context
            st.session_state.analysis_state['current_window_analysis'] = {
                'time_window': f"{start_datetime} to {end_datetime}",
                'total_reports': total_in_window,
                'real_outages': len(window_real),
                'false_positives': len(window_false),
                'real_customers_affected': sum(o['customers'] for o in window_real)
            }
            
        except Exception as e:
            logger.error(f"Time window analysis error: {str(e)}")
            st.error(f"❌ Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()