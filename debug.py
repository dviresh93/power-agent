"""
Improved LLM-Powered Outage Analysis Agent
- Initial full dataset analysis on upload
- Date/time range picker
- LLM suggests interesting time windows to explore
- Enhanced chat interface with dataset context
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

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
# Fix the pydantic import issue properly
try:
    from pydantic import BaseModel, Field
except ImportError:
    try:
        from pydantic.v1 import BaseModel, Field
    except ImportError:
        # Fallback - create dummy classes if pydantic fails completely
        class BaseModel:
            pass
        class Field:
            def __init__(self, *args, **kwargs):
                pass

# LLM imports - Claude as default
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Vector DB imports
import chromadb

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION MANAGEMENT ====================
class PromptManager:
    """Manages prompts loaded from JSON configuration"""
    
    def __init__(self, prompts_file: str = "prompts.json"):
        self.prompts_file = prompts_file
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict:
        """Load prompts from JSON file"""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
            logger.info(f"Loaded prompts from {self.prompts_file}")
            return prompts
        except FileNotFoundError:
            logger.error(f"Prompts file {self.prompts_file} not found")
            return self._get_default_prompts()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing prompts JSON: {str(e)}")
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict:
        """Fallback default prompts"""
        return {
            "weather_analysis": {
                "system": "You are an expert power grid engineer analyzing weather conditions for outage causation.",
                "human": "Analyze weather data: {weather_data}"
            },
            "severity_assessment": {
                "system": "You are a power system operations manager assessing outage severity.",
                "human": "Assess outage: {outage_data} with weather: {weather_analysis}"
            },
            "dataset_overview": {
                "system": "You are an expert power grid analyst. Analyze the complete outage dataset and provide comprehensive insights including patterns, trends, peak times, geographic clusters, and interesting time windows to explore further.",
                "human": "Analyze this complete outage dataset:\n\nDataset Summary:\n{dataset_summary}\n\nSample Outages:\n{sample_outages}\n\nProvide insights on patterns, trends, and suggest 3-5 interesting time windows for detailed analysis."
            },
            "suggest_time_windows": {
                "system": "You are a data analysis expert. Based on outage patterns, suggest interesting time windows for focused analysis. Consider peak activity periods, unusual patterns, severe weather correlations, and seasonal variations.",
                "human": "Based on this outage data analysis:\n{analysis_summary}\n\nSuggest 3-5 specific date/time windows that would be most interesting to analyze in detail. Provide reasoning for each suggestion."
            },
            "chatbot_assistant": {
                "system": "You are an expert power grid operations assistant with full knowledge of the outage dataset. You can answer questions about patterns, trends, specific outages, recommendations, and help users explore the data more effectively.",
                "human": "{user_question}\n\nContext:\n{analysis_context}"
            }
        }
    
    def get_prompt(self, prompt_name: str) -> ChatPromptTemplate:
        """Get a specific prompt template"""
        if prompt_name not in self.prompts:
            logger.warning(f"Prompt '{prompt_name}' not found, using default")
            return ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant."),
                ("human", "{input}")
            ])
        
        prompt_config = self.prompts[prompt_name]
        return ChatPromptTemplate.from_messages([
            ("system", prompt_config["system"]),
            ("human", prompt_config["human"])
        ])

# ==================== STATE MANAGEMENT ====================
class AnalysisState(TypedDict):
    """Enhanced state for comprehensive analysis"""
    dataset_loaded: bool
    full_analysis_complete: bool
    dataset_summary: Dict
    interesting_windows: List[Dict]
    current_analysis: Optional[Dict]
    chat_context: Dict
    error_messages: List[str]

# ==================== LLM MANAGER ====================
class LLMManager:
    """Enhanced LLM manager with Claude as default"""
    
    def __init__(self):
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM with Claude as default preference"""
        if os.getenv("ANTHROPIC_API_KEY"):
            logger.info("Using Anthropic Claude (default)")
            return ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=0.1,
                streaming=True
            )
        elif os.getenv("OPENAI_API_KEY"):
            logger.info("Using OpenAI GPT-4 (fallback)")
            return ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.1,
                streaming=True
            )
        else:
            raise ValueError("No LLM API keys configured. Please set ANTHROPIC_API_KEY (recommended) or OPENAI_API_KEY")
    
    def get_llm(self):
        return self.llm

# ==================== SERVICES ====================
class WeatherService:
    """Weather service for fetching historical data"""
    
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
    def get_historical_weather(self, lat: float, lon: float, date: datetime) -> Dict:
        """Fetch comprehensive weather data"""
        try:
            date_str = date.strftime("%Y-%m-%d")
            
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": date_str,
                "end_date": date_str,
                "hourly": "temperature_2m,precipitation,windspeed_10m,windgusts_10m,snowfall",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
                "timezone": "America/Chicago"
            }
            
            import requests
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            hourly_data = data.get('hourly', {})
            target_hour = min(date.hour, len(hourly_data.get('temperature_2m', [])) - 1) if hourly_data.get('temperature_2m') else 0
            
            return {
                'timestamp': date.isoformat(),
                'coordinates': {'lat': lat, 'lon': lon},
                'temperature': self._safe_get(hourly_data.get('temperature_2m', []), target_hour),
                'precipitation': self._safe_get(hourly_data.get('precipitation', []), target_hour),
                'wind_speed': self._safe_get(hourly_data.get('windspeed_10m', []), target_hour),
                'wind_gusts': self._safe_get(hourly_data.get('windgusts_10m', []), target_hour),
                'snowfall': self._safe_get(hourly_data.get('snowfall', []), target_hour)
            }
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return {'error': str(e), 'timestamp': date.isoformat()}
    
    def _safe_get(self, data_list: List, index: int):
        if not data_list or index >= len(data_list) or index < 0:
            return None
        return data_list[index]

class OutageVectorDB:
    """Enhanced vector database with comprehensive analysis"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        try:
            self.collection = self.client.get_collection("outages")
            logger.info("Connected to existing outages collection")
        except Exception:
            self.collection = self.client.create_collection("outages")
            logger.info("Created new outages collection")
    
    def load_outage_data(self, df: pd.DataFrame) -> Dict:
        """Load data and return comprehensive summary"""
        try:
            logger.info(f"Loading {len(df)} outage records")
            
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
                doc_text = f"""
                Power outage on {row['DATETIME']} at {row['LATITUDE']}, {row['LONGITUDE']} 
                affecting {row['CUSTOMERS']} customers.
                """
                
                metadata = {
                    'datetime': row['DATETIME'],
                    'latitude': float(row['LATITUDE']),
                    'longitude': float(row['LONGITUDE']),
                    'customers': int(row['CUSTOMERS']),
                    'date': row['DATETIME'][:10],
                    'hour': int(row['DATETIME'][11:13]) if len(row['DATETIME']) > 11 else 0,
                    'day_of_week': datetime.strptime(row['DATETIME'][:10], '%Y-%m-%d').weekday(),
                    'month': int(row['DATETIME'][5:7]),
                    'year': int(row['DATETIME'][:4])
                }
                
                documents.append(doc_text.strip())
                metadatas.append(metadata)
                ids.append(f"outage_{idx}")
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Generate comprehensive summary
            summary = self._generate_dataset_summary(df)
            logger.info("Data loaded and analyzed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error loading outage data: {str(e)}")
            raise
    
    def _generate_dataset_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive dataset summary"""
        try:
            # Convert datetime column for analysis
            df['datetime_parsed'] = pd.to_datetime(df['DATETIME'])
            df['date'] = df['datetime_parsed'].dt.date
            df['hour'] = df['datetime_parsed'].dt.hour
            df['day_of_week'] = df['datetime_parsed'].dt.day_name()
            
            # Calculate summary statistics with explicit type conversion
            summary = {
                "total_records": int(len(df)),
                "date_range": {
                    "start": df['datetime_parsed'].min().strftime('%Y-%m-%d'),
                    "end": df['datetime_parsed'].max().strftime('%Y-%m-%d')
                },
                "customer_impact": {
                    "total_affected": int(df['CUSTOMERS'].sum()),
                    "avg_per_outage": float(df['CUSTOMERS'].mean()),
                    "max_single_outage": int(df['CUSTOMERS'].max()),
                    "min_single_outage": int(df['CUSTOMERS'].min())
                },
                "geographic_coverage": {
                    "lat_range": f"{float(df['LATITUDE'].min()):.3f} to {float(df['LATITUDE'].max()):.3f}",
                    "lon_range": f"{float(df['LONGITUDE'].min()):.3f} to {float(df['LONGITUDE'].max()):.3f}",
                    "center": [float(df['LATITUDE'].mean()), float(df['LONGITUDE'].mean())]
                },
                "temporal_patterns": {
                    "peak_hour": int(df.groupby('hour')['CUSTOMERS'].sum().idxmax()),
                    "peak_day": str(df.groupby('day_of_week')['CUSTOMERS'].sum().idxmax()),
                    "busiest_date": df.groupby('date')['CUSTOMERS'].sum().idxmax().strftime('%Y-%m-%d'),
                    "hourly_distribution": {str(k): int(v) for k, v in df.groupby('hour').size().to_dict().items()},
                    "daily_distribution": {str(k): int(v) for k, v in df.groupby('date').size().to_dict().items()}
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {"error": str(e)}
    
    def query_outages_by_range(self, start_datetime: datetime, end_datetime: datetime) -> List[Dict]:
        """Query outages within a specific time range"""
        try:
            results = self.collection.query(
                query_texts=[f"outages between {start_datetime.strftime('%Y-%m-%d')} and {end_datetime.strftime('%Y-%m-%d')}"],
                n_results=100,
                where={
                    "datetime": {
                        "$gte": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                        "$lte": end_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
            )
            
            outages = []
            if results['metadatas']:
                for metadata in results['metadatas'][0]:
                    outages.append(metadata)
            
            return outages
            
        except Exception as e:
            logger.error(f"Error querying outages by range: {str(e)}")
            return []

# ==================== LLM TOOLS ====================

@tool
def validate_outage_report(outage_report: dict, weather_data: dict) -> str:
    """Use Claude to validate if outage report is real or false positive"""
    try:
        llm_manager = LLMManager()
        prompt_manager = PromptManager()
        
        # Create specialized validation prompt
        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert power grid engineer specializing in outage validation. 
            
Your job is to determine if an outage REPORT is REAL or a FALSE POSITIVE based on weather conditions.

FALSE POSITIVES occur when:
- Weather conditions are too mild to cause outages (calm winds <25 mph, little/no precipitation, normal temperatures)
- Equipment sensor malfunctions reporting phantom outages
- Data processing glitches creating erroneous reports
- Network communication issues
- Planned maintenance misreported as outages

REAL OUTAGES are caused by:
- High winds: sustained >25 mph (40 km/h) or gusts >35 mph (56 km/h)
- Heavy precipitation: >0.5 inches/hour (12.7 mm/hour)  
- Ice/snow accumulation: >2 inches (5 cm)
- Temperature extremes: <10°F (-12°C) or >95°F (35°C)
- Lightning during storms
- Equipment failures during severe weather

Respond with: 'REAL OUTAGE' or 'FALSE POSITIVE' followed by detailed reasoning."""),
            ("human", """Analyze this outage report:

OUTAGE REPORT:
Time: {datetime}
Location: {latitude}, {longitude}
Customers Claimed: {customers}

WEATHER CONDITIONS:
{weather_summary}

Is this a REAL OUTAGE or FALSE POSITIVE? Provide detailed reasoning.""")
        ])
        
        # Format weather data for analysis
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
        logger.error(f"Error validating outage report: {str(e)}")
        return f"Error validating report: {str(e)}"
@tool
def analyze_outage_reports_dataset(dataset_summary: dict, sample_reports: list) -> str:
    """Analyze dataset of outage reports for false positive detection"""
    try:
        llm_manager = LLMManager()
        prompt_manager = PromptManager()
        
        # Create specialized dataset analysis prompt for reports
        dataset_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a dataset of OUTAGE REPORTS (not confirmed outages) that need validation.
            
Each row represents a potential outage report claiming a certain number of customers affected. Your job is to:
1. Identify patterns in the reports
2. Assess the time period characteristics (Jan 1, 2022 midnight-5AM)
3. Provide insights for systematic false positive detection
4. Suggest what to look for during validation

Remember: These are REPORTS that may contain false positives, not confirmed outages."""),
            ("human", """Analyze these outage reports for validation insights:

Dataset Summary: {dataset_summary}
Sample Reports: {sample_reports}

Provide insights about the reporting patterns and what to expect during validation process.""")
        ])
        
        chain = dataset_prompt | llm_manager.get_llm()
        
        # Convert numpy types safely
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        clean_summary = convert_numpy_types(dataset_summary)
        clean_samples = convert_numpy_types(sample_reports)
        
        response = chain.invoke({
            "dataset_summary": json.dumps(clean_summary, indent=2, default=str),
            "sample_reports": json.dumps(clean_samples, indent=2, default=str)
        })
        
        return response.content
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}")
        return f"Error analyzing dataset: {str(e)}"

@tool
def suggest_interesting_windows(analysis_summary: str) -> str:
    """Suggest interesting time windows for analysis"""
    try:
        llm_manager = LLMManager()
        prompt_manager = PromptManager()
        
        suggest_prompt = prompt_manager.get_prompt("suggest_time_windows")
        chain = suggest_prompt | llm_manager.get_llm()
        
        response = chain.invoke({"analysis_summary": analysis_summary})
        return response.content
        
    except Exception as e:
        return f"Error suggesting time windows: {str(e)}"

@tool
def chat_about_validation_results(question: str, context: dict) -> str:
    """Chat about validation results with full context"""
    try:
        llm_manager = LLMManager()
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert power grid operations assistant. You have just completed validation of outage reports, 
            determining which are real outages vs false positives based on weather analysis.
            
            You can help users understand:
            1. Why specific reports were classified as false positives  
            2. What weather conditions caused the real outages
            3. Patterns in false positive causes
            4. Operational recommendations to reduce false alarms
            5. Infrastructure improvements to prevent real outages
            
            Be specific about the validation results and provide actionable insights."""),
            ("human", "{user_question}\n\nValidation Context:\n{analysis_context}")
        ])
        
        chain = chat_prompt | llm_manager.get_llm()
        
        response = chain.invoke({
            "user_question": question,
            "analysis_context": json.dumps(context, indent=2, default=str)
        })
        
        return response.content
        
    except Exception as e:
        return f"Error in chat: {str(e)}"

# ==================== MAIN APPLICATION ====================
def main():
    """Enhanced Streamlit application"""
    
    st.set_page_config(
        page_title="Improved Outage Analysis Agent",
        page_icon="🤖⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'analysis_state' not in st.session_state:
        st.session_state.analysis_state = {
            'dataset_loaded': False,
            'full_analysis_complete': False,
            'dataset_summary': {},
            'interesting_windows': [],
            'current_analysis': None,
            'chat_context': {},
            'error_messages': []
        }
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Remove debug messages and return to clean UI
    st.title("🤖⚡ Improved Outage Analysis Agent")
    st.markdown("**Claude-powered comprehensive outage analysis with intelligent exploration**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # LLM Status
        try:
            llm_manager = LLMManager()
            st.success("✅ Claude Ready")
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            return
        
        st.divider()
        
        # Data Upload
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload Outage CSV", type=['csv'])
        
        if uploaded_file is not None and not st.session_state.analysis_state['dataset_loaded']:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"📊 Loaded {len(df)} records")
                
                if st.button("🔍 Validate All Reports"):
                    with st.spinner("🤖 Claude validating outage reports..."):
                        # Load into vector DB and get summary
                        vector_db = OutageVectorDB()
                        summary = vector_db.load_outage_data(df)
                        
                        # Get sample reports for LLM analysis
                        sample_reports = df.head(10).to_dict('records')
                        
                        # Analyze reports dataset with Claude
                        analysis = analyze_outage_reports_dataset.invoke({
                            "dataset_summary": summary,
                            "sample_reports": sample_reports
                        })
                        
                        # Validate each report against weather data
                        validation_results = validate_all_reports(df)
                        
                        # Update state
                        st.session_state.analysis_state.update({
                            'dataset_loaded': True,
                            'full_analysis_complete': True,
                            'dataset_summary': summary,
                            'validation_results': validation_results,
                            'chat_context': {
                                'dataset_summary': summary,
                                'dataset_analysis': analysis,
                                'validation_results': validation_results,
                                'raw_data_sample': sample_reports
                            }
                        })
                        
                        st.success("✅ Report validation finished!")
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        # Show analysis status
        if st.session_state.analysis_state['dataset_loaded']:
            st.success("✅ Dataset Analyzed")
            summary = st.session_state.analysis_state['dataset_summary']
            if summary:
                st.write(f"**Records:** {summary.get('total_records', 'N/A')}")
                st.write(f"**Date Range:** {summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}")
                st.write(f"**Total Customers Affected:** {summary.get('customer_impact', {}).get('total_affected', 'N/A'):,}")
    
    # Main content area
    if not st.session_state.analysis_state['full_analysis_complete']:
        st.info("👆 Please upload your CSV file and click 'Analyze Complete Dataset' to get started")
        return
    
    # Display validation results instead of generic dataset overview
    # Define display_validation_results function that's missing
    def display_validation_results():
        """Display validation results with proper filtering"""
        
        st.header("🔍 Validation Results")
        
        validation_results = st.session_state.analysis_state.get('validation_results', {})
        
        if not validation_results:
            st.warning("⚠️ No validation results available")
            return
        
        stats = validation_results.get('statistics', {})
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Reports", 
                f"{stats.get('total_reports', 0):,}",
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
                help="Customers affected by real outages only"
            )
        
        with col4:
            st.metric(
                "False Positive Rate", 
                f"{stats.get('false_positive_rate', 0):.1f}%",
                help="Percentage of reports that were false positives"
            )
    
    display_validation_results()
    
    st.divider()
    
    # Time window selection for focused analysis
    st.header("🎯 Focused Time Window Analysis")
    
    summary = st.session_state.analysis_state['dataset_summary']
    date_range = summary.get('date_range', {})
    
    if date_range.get('start') and date_range.get('end'):
        min_date = datetime.strptime(date_range['start'], '%Y-%m-%d').date()
        max_date = datetime.strptime(date_range['end'], '%Y-%m-%d').date()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Start Date & Time")
            start_date = st.date_input("Start Date:", min_value=min_date, max_value=max_date, value=min_date)
            start_time = st.time_input("Start Time:", value=time(0, 0))
        
        with col2:
            st.subheader("📅 End Date & Time")
            end_date = st.date_input("End Date:", min_value=min_date, max_value=max_date, value=min_date)
            end_time = st.time_input("End Time:", value=time(23, 59))
        
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
        
        if st.button("🔍 Analyze Time Window"):
            analyze_time_window(start_datetime, end_datetime)
    
    # Chat interface (always available after dataset analysis)
    st.divider()
    display_chat_interface()

def validate_all_reports(df: pd.DataFrame) -> Dict:
    """Validate all outage reports against weather data"""
    
    weather_service = WeatherService()
    real_outages = []
    false_positives = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
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
            
            # Prepare report data
            report_data = {
                'datetime': row['DATETIME'],
                'latitude': row['LATITUDE'],
                'longitude': row['LONGITUDE'],
                'customers': row['CUSTOMERS']
            }
            
            # Validate with Claude
            validation_result = validate_outage_report.invoke({
                "outage_report": report_data,
                "weather_data": weather
            })
            
            # Determine if real or false positive
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
            logger.error(f"Error validating report {idx}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    # Calculate statistics
    total_reports = len(df)
    real_count = len(real_outages)
    false_count = len(false_positives)
    
    return {
        'total_reports': total_reports,
        'real_outages': real_outages,
        'false_positives': false_positives,
        'statistics': {
            'real_count': real_count,
            'false_positive_count': false_count,
            'false_positive_rate': (false_count / total_reports * 100) if total_reports > 0 else 0,
            'total_customers_actually_affected': sum(r['customers'] for r in real_outages)
        }
    }
    """Display comprehensive dataset overview"""
    
    st.header("📊 Complete Dataset Analysis")
    
    summary = st.session_state.analysis_state['dataset_summary']
    context = st.session_state.analysis_state['chat_context']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Outages", f"{summary.get('total_records', 0):,}")
    with col2:
        st.metric("Customers Affected", f"{summary.get('customer_impact', {}).get('total_affected', 0):,}")
    with col3:
        st.metric("Avg per Outage", f"{summary.get('customer_impact', {}).get('avg_per_outage', 0):.1f}")
    with col4:
        st.metric("Max Single Outage", f"{summary.get('customer_impact', {}).get('max_single_outage', 0):,}")
    
    # Claude's analysis
    if context.get('full_analysis'):
        st.subheader("🧠 Claude's Dataset Insights")
        st.markdown(context['full_analysis'])
    
    # Suggested time windows
    if context.get('suggestions'):
        st.subheader("💡 Suggested Time Windows to Explore")
        st.markdown(context['suggestions'])

def analyze_time_window(start_datetime: datetime, end_datetime: datetime):
    """Analyze specific time window"""
    
    with st.spinner(f"🤖 Analyzing {start_datetime} to {end_datetime}..."):
        try:
            vector_db = OutageVectorDB()
            outages = vector_db.query_outages_by_range(start_datetime, end_datetime)
            
            if not outages:
                st.warning(f"⚠️ No outages found in the selected time window")
                return
            
            # Simple analysis for focused window
            st.success(f"✅ Found {len(outages)} outages in time window")
            
            # Display results
            df_window = pd.DataFrame(outages)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Outages Found", len(outages))
            with col2:
                st.metric("Total Customers", df_window['customers'].sum())
            with col3:
                st.metric("Peak Hour", df_window.groupby('hour')['customers'].sum().idxmax())
            
            # Map visualization
            if len(outages) > 0:
                st.subheader("🗺️ Outage Locations")
                
                center_lat = df_window['latitude'].mean()
                center_lon = df_window['longitude'].mean()
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                
                for outage in outages:
                    folium.CircleMarker(
                        location=[outage['latitude'], outage['longitude']],
                        radius=max(5, min(15, outage['customers'] * 0.3)),
                        popup=f"Time: {outage['datetime']}<br>Customers: {outage['customers']}",
                        tooltip=f"{outage['customers']} customers",
                        fillOpacity=0.7
                    ).add_to(m)
                
                st_folium(m, width=700, height=400)
            
            # Update chat context with window analysis
            st.session_state.analysis_state['current_analysis'] = {
                'time_window': f"{start_datetime} to {end_datetime}",
                'outages_found': len(outages),
                'outages_data': outages
            }
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

def display_chat_interface():
    """Display enhanced chat interface"""
    
    st.header("💬 Chat with Claude about Your Data")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f"**🙋 You:** {chat['content']}")
        else:
            st.markdown(f"**🤖 Claude:** {chat['content']}")
    
    # Chat input
    user_question = st.text_input(
        "Ask Claude about your outage data:",
        placeholder="e.g., 'What patterns do you see?' or 'When do most outages occur?'",
        key="chat_input"
    )
    
    # Quick question buttons focused on validation
    st.write("**Quick Questions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("❌ Why false positives?"):
            user_question = "Why were these reports classified as false positives? What were the main reasons?"
    
    with col2:
        if st.button("⚡ What caused real outages?"):
            user_question = "What weather conditions caused the confirmed real outages?"
    
    with col3:
        if st.button("📊 Validation summary?"):
            user_question = "Provide a summary of the validation results and false positive rate."
    
    if st.button("💭 Ask Claude") and user_question:
        with st.spinner("🤖 Claude is analyzing..."):
            try:
                # Prepare context
                context = st.session_state.analysis_state['chat_context'].copy()
                if st.session_state.analysis_state.get('current_analysis'):
                    context['current_window_analysis'] = st.session_state.analysis_state['current_analysis']
                
                # Get response focused on validation results
                response = chat_about_validation_results.invoke({
                    "question": user_question,
                    "context": context
                })
                
                # Add to history
                st.session_state.chat_history.extend([
                    {'role': 'user', 'content': user_question},
                    {'role': 'assistant', 'content': response}
                ])
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Chat error: {str(e)}")
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()