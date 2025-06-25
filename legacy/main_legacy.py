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
import matplotlib.pyplot as plt
from reportlab.platypus import Image as RLImage
import io
import time as time_module
import asyncio

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

# Vector DB imports moved to services/vector_db_service.py

# Geocoding import moved to services module

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

# ==================== SERVICE IMPORTS ====================
from services.llm_service import LLMManager
from services.weather_service import WeatherService
from services.geocoding_service import GeocodingService
from services.vector_db_service import OutageVectorDB


# ==================== GEOCODING SERVICE ====================
# Moved to services/geocoding_service.py

# ==================== VECTOR DATABASE ====================
# Moved to services/vector_db_service.py

# ==================== LLM TOOLS FOLLOWING 2025 PATTERNS ====================

@tool
def validate_outage_report(outage_report: dict, weather_data: dict) -> str:
    """Validate outage report against weather conditions using LLM analysis"""
    try:
        llm_manager = LLMManager()
        
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
        llm_manager = LLMManager()
        
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
    """Generate an exhaustive outage analysis report with detailed explanations for every decision"""
    try:
        llm_manager = LLMManager()
        
        # Load the exhaustive report generation prompt
        with open('prompts.json', 'r') as f:
            prompts = json.load(f)
        
        exhaustive_prompt = ChatPromptTemplate.from_messages([
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
        
        chain = exhaustive_prompt | llm_manager.get_llm()
        
        response = chain.invoke({
            "raw_summary": json.dumps(raw_summary, indent=2, default=str),
            "validation_results": json.dumps(validation_results, indent=2, default=str),
            "time_period": time_period,
            "map_data": json.dumps(map_data, indent=2, default=str),
            "all_decisions": json.dumps(all_decisions, indent=2, default=str),
            "false_positives_count": len(false_positives),
            "real_outages_count": len(real_outages)
        })
        
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
        llm_manager = LLMManager()
        
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
        
        # LLM Status Check
        try:
            llm_manager = LLMManager()
            st.success("✅ Claude Connected")
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.info("💡 Add your API key to .env file")
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