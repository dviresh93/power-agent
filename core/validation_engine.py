"""
Validation Engine Module

Core validation logic and orchestration extracted from main.py.
Handles LangGraph tools, agent setup, validation workflow management,
and integration with all services (LLM, weather, geocoding, vector DB).

Features:
- Framework-independent validation engine
- Dependency injection for services
- LangGraph 2025 best practices with tools and agents
- Comprehensive validation workflow management
- Result processing and formatting
- Rate limiting and error handling
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

# LangGraph imports - 2025 best practices
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# Core models and services
from core.models import (
    OutageAnalysisState, 
    ValidationResult, 
    OutageRecord, 
    WeatherData,
    LocationInfo
)
from services.llm_service import LLMManager
from services.weather_service import WeatherService
from services.geocoding_service import GeocodingService
from services.vector_db_service import OutageVectorDB
from config.settings import Settings

# Set up logging
logger = logging.getLogger(__name__)


# ==================== LANGGRAPH TOOLS ====================

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
        settings = Settings()
        
        # Load the report generation prompt from settings
        prompts = settings.prompts
        
        report_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.get("comprehensive_report_generation", {}).get("system", 
                                   "Generate a comprehensive power outage analysis report.")),
            ("human", prompts.get("comprehensive_report_generation", {}).get("human", 
                                  "Create report based on validation results and raw summary."))
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
        settings = Settings()
        
        # Load the exhaustive report generation prompt
        prompts = settings.prompts
        
        report_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.get("exhaustive_report_generation", {}).get("system", 
                                   "Generate an exhaustive technical report with detailed explanations.")),
            ("human", prompts.get("exhaustive_report_generation", {}).get("human", 
                                  "Create exhaustive report with detailed analysis."))
        ])
        
        # Determine time period from raw data
        date_range = raw_summary.get('date_range', {})
        time_period = f"{date_range.get('start', 'Unknown')} to {date_range.get('end', 'Unknown')}"
        
        # Generate detailed analysis context
        detailed_context = {
            "validation_methodology": "LLM-powered weather correlation analysis",
            "false_positive_detection": "Automated classification based on weather thresholds",
            "data_sources": ["Open-Meteo Historical Weather API", "Nominatim Geocoding"],
            "analysis_scope": time_period,
            "total_reports_analyzed": validation_results.get('total_reports', 0)
        }
        
        chain = report_prompt | llm_manager.get_llm()
        
        response = chain.invoke({
            "raw_summary": json.dumps(raw_summary, indent=2, default=str),
            "validation_results": json.dumps(validation_results, indent=2, default=str),
            "time_period": time_period,
            "detailed_context": json.dumps(detailed_context, indent=2, default=str)
        })
        
        return response.content
        
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
1. Run the validation process to load data and perform weather analysis
2. This will provide detailed weather analysis and real vs false positive classifications
3. Once complete, I'll have detailed outage analysis, weather correlations, and false positive identification

**What I can tell you now:** Basic dataset statistics from the cached raw data. For detailed outage analysis, weather correlations, and false positive identification, please run the validation process first."""
        
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


# ==================== VALIDATION ENGINE CLASS ====================

class ValidationEngine:
    """
    Core validation engine with dependency injection and LangGraph integration.
    
    Orchestrates the validation workflow:
    1. Load and validate data
    2. Perform weather correlation analysis  
    3. Use LLM to classify real outages vs false positives
    4. Generate comprehensive reports
    5. Provide chat interface for results exploration
    """
    
    def __init__(self, 
                 settings: Optional[Settings] = None,
                 llm_manager: Optional[LLMManager] = None,
                 weather_service: Optional[WeatherService] = None,
                 geocoding_service: Optional[GeocodingService] = None,
                 vector_db_service: Optional[OutageVectorDB] = None):
        """
        Initialize validation engine with dependency injection.
        
        Args:
            settings: Configuration settings (defaults to global settings)
            llm_manager: LLM service instance (optional)
            weather_service: Weather service instance (optional) 
            geocoding_service: Geocoding service instance (optional)
            vector_db_service: Vector DB service instance (optional)
        """
        self.settings = settings or Settings()
        self.llm_manager = llm_manager or LLMManager()
        self.weather_service = weather_service or WeatherService()
        self.geocoding_service = geocoding_service or GeocodingService()
        self.vector_db_service = vector_db_service or OutageVectorDB()
        
        # Initialize LangGraph components
        self.tools = [
            validate_outage_report,
            generate_comprehensive_report,
            generate_exhaustive_report,
            chat_about_results
        ]
        self.tool_node = ToolNode(self.tools)
        
        # Build validation workflow graph
        self.workflow = self._build_validation_workflow()
        
        logger.info("✅ ValidationEngine initialized with dependency injection")
    
    def _build_validation_workflow(self) -> StateGraph:
        """
        Build LangGraph workflow for validation process.
        
        Returns:
            StateGraph: Configured validation workflow
        """
        workflow = StateGraph(OutageAnalysisState)
        
        # Define workflow nodes
        workflow.add_node("load_data", self._load_data_node)
        workflow.add_node("validate_reports", self._validate_reports_node)
        workflow.add_node("process_results", self._process_results_node)
        workflow.add_node("generate_report", self._generate_report_node)
        workflow.add_node("tools", self.tool_node)
        
        # Define workflow edges
        workflow.add_edge(START, "load_data")
        workflow.add_edge("load_data", "validate_reports")
        workflow.add_edge("validate_reports", "process_results")
        workflow.add_edge("process_results", "generate_report")
        workflow.add_edge("generate_report", END)
        
        # Add conditional edges for tool usage
        workflow.add_conditional_edges("tools", self._should_continue_with_tools)
        
        return workflow.compile()
    
    def _load_data_node(self, state: OutageAnalysisState) -> OutageAnalysisState:
        """Load and validate outage data."""
        try:
            # Load data from configured source
            if os.path.exists(self.settings.default_data_file):
                df = pd.read_csv(self.settings.default_data_file)
                
                # Validate required columns
                missing_cols = [col for col in self.settings.required_columns if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
                
                state['dataset_loaded'] = True
                state['raw_dataset_summary'] = self._generate_raw_summary(df)
                logger.info(f"✅ Loaded {len(df)} outage reports for validation")
            else:
                raise FileNotFoundError(f"Data file not found: {self.settings.default_data_file}")
                
        except Exception as e:
            logger.error(f"❌ Data loading failed: {str(e)}")
            state['dataset_loaded'] = False
            state['validation_errors'] = [f"Data loading error: {str(e)}"]
        
        return state
    
    def _validate_reports_node(self, state: OutageAnalysisState) -> OutageAnalysisState:
        """Validate all outage reports using weather analysis."""
        if not state.get('dataset_loaded', False):
            logger.error("❌ Cannot validate reports - data not loaded")
            return state
        
        try:
            # Load the dataframe again (in real implementation, pass through state)
            df = pd.read_csv(self.settings.default_data_file)
            
            validation_results = self._validate_all_reports(df)
            
            state['validation_complete'] = True
            state['validation_results'] = validation_results
            
            logger.info(f"✅ Validation complete: {validation_results.get('statistics', {}).get('real_count', 0)} real outages, "
                       f"{validation_results.get('statistics', {}).get('false_positive_count', 0)} false positives")
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            state['validation_complete'] = False
            state['validation_errors'] = state.get('validation_errors', []) + [f"Validation error: {str(e)}"]
        
        return state
    
    def _process_results_node(self, state: OutageAnalysisState) -> OutageAnalysisState:
        """Process and format validation results."""
        if not state.get('validation_complete', False):
            logger.error("❌ Cannot process results - validation not complete")
            return state
        
        try:
            validation_results = state.get('validation_results', {})
            
            # Generate filtered summary
            filtered_summary = self._generate_filtered_summary(validation_results)
            state['filtered_summary'] = filtered_summary
            
            # Update vector database with validated results
            if self.vector_db_service and validation_results.get('real_outages'):
                self._update_vector_database(validation_results['real_outages'])
            
            logger.info("✅ Results processed and formatted successfully")
            
        except Exception as e:
            logger.error(f"❌ Results processing failed: {str(e)}")
            state['validation_errors'] = state.get('validation_errors', []) + [f"Processing error: {str(e)}"]
        
        return state
    
    def _generate_report_node(self, state: OutageAnalysisState) -> OutageAnalysisState:
        """Generate final analysis report."""
        try:
            validation_results = state.get('validation_results', {})
            raw_summary = state.get('raw_dataset_summary', {})
            
            # Generate comprehensive report using LangGraph tool
            report_content = generate_comprehensive_report.invoke({
                "validation_results": validation_results,
                "raw_summary": raw_summary
            })
            
            state['final_report'] = report_content
            logger.info("✅ Comprehensive report generated successfully")
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {str(e)}")
            state['validation_errors'] = state.get('validation_errors', []) + [f"Report generation error: {str(e)}"]
        
        return state
    
    def _should_continue_with_tools(self, state: OutageAnalysisState) -> str:
        """Determine if workflow should continue using tools."""
        # Simple logic - in practice this would be more sophisticated
        if state.get('validation_complete', False):
            return END
        else:
            return "tools"
    
    def _validate_all_reports(self, df: pd.DataFrame, 
                             request_delay: float = 0.5, 
                             max_retries: int = 3) -> Dict:
        """
        Validate all outage reports and return comprehensive results.
        
        Args:
            df: DataFrame containing outage reports
            request_delay: Delay between API requests (rate limiting)
            max_retries: Maximum retry attempts for failed requests
            
        Returns:
            Dict: Comprehensive validation results
        """
        real_outages = []
        false_positives = []
        validation_errors = []
        
        logger.info(f"🔍 Starting validation of {len(df)} outage reports with rate limiting")
        
        for idx, row in df.iterrows():
            try:
                # Parse datetime
                report_datetime = datetime.strptime(row['DATETIME'], "%Y-%m-%d %H:%M:%S")
                
                # Get weather data
                weather = self.weather_service.get_historical_weather(
                    lat=row['LATITUDE'],
                    lon=row['LONGITUDE'],
                    date=report_datetime
                )
                
                # Get location name using reverse geocoding
                location_info = self.geocoding_service.get_location_name(
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
                retry_delay = 2  # Base retry delay
                
                for attempt in range(max_retries):
                    try:
                        validation_result = validate_outage_report.invoke({
                            "outage_report": report_data,
                            "weather_data": weather
                        })
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "overloaded" in error_msg or "rate" in error_msg or "limit" in error_msg:
                            if attempt < max_retries - 1:
                                logger.warning(f"API overloaded, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            else:
                                logger.error(f"Max retries reached for report {idx+1}, marking as validation error")
                                validation_result = f"VALIDATION ERROR: API overloaded after {max_retries} attempts"
                        else:
                            logger.error(f"Validation error for report {idx+1}: {str(e)}")
                            validation_result = f"VALIDATION ERROR: {str(e)}"
                        break
                
                # Add delay between requests to prevent overwhelming the API
                time.sleep(request_delay)
                
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
        self._cache_validation_results(validation_results)
        
        return validation_results
    
    def _generate_raw_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary of raw dataset."""
        try:
            # Basic statistics
            total_reports = len(df)
            total_customers = int(df['CUSTOMERS'].sum())
            avg_customers = df['CUSTOMERS'].mean()
            
            # Date range
            df['DATETIME'] = pd.to_datetime(df['DATETIME'])
            date_range = {
                'start': df['DATETIME'].min().strftime('%Y-%m-%d %H:%M'),
                'end': df['DATETIME'].max().strftime('%Y-%m-%d %H:%M')
            }
            
            # Geographic coverage
            lat_range = f"{df['LATITUDE'].min():.3f} to {df['LATITUDE'].max():.3f}"
            lon_range = f"{df['LONGITUDE'].min():.3f} to {df['LONGITUDE'].max():.3f}"
            center = [df['LATITUDE'].mean(), df['LONGITUDE'].mean()]
            
            return {
                'total_reports': total_reports,
                'date_range': date_range,
                'raw_customer_claims': {
                    'total_claimed': total_customers,
                    'avg_per_report': avg_customers,
                    'max_single_report': int(df['CUSTOMERS'].max()),
                    'min_single_report': int(df['CUSTOMERS'].min())
                },
                'geographic_coverage': {
                    'lat_range': lat_range,
                    'lon_range': lon_range,
                    'center': center
                },
                'data_quality': {
                    'missing_datetime': df['DATETIME'].isna().sum(),
                    'missing_coordinates': df[['LATITUDE', 'LONGITUDE']].isna().sum().sum(),
                    'missing_customers': df['CUSTOMERS'].isna().sum()
                }
            }
        except Exception as e:
            logger.error(f"❌ Raw summary generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_filtered_summary(self, validation_results: Dict) -> Dict:
        """Generate summary of filtered/validated results."""
        try:
            real_outages = validation_results.get('real_outages', [])
            false_positives = validation_results.get('false_positives', [])
            stats = validation_results.get('statistics', {})
            
            # Real outages analysis
            if real_outages:
                real_customers = [r['customers'] for r in real_outages]
                real_summary = {
                    'count': len(real_outages),
                    'total_customers': sum(real_customers),
                    'avg_customers': sum(real_customers) / len(real_customers),
                    'max_customers': max(real_customers),
                    'min_customers': min(real_customers)
                }
            else:
                real_summary = {'count': 0, 'total_customers': 0}
            
            # False positives analysis
            if false_positives:
                false_customers = [fp['customers'] for fp in false_positives]
                false_summary = {
                    'count': len(false_positives),
                    'customers_incorrectly_claimed': sum(false_customers),
                    'avg_false_claim': sum(false_customers) / len(false_customers)
                }
            else:
                false_summary = {'count': 0, 'customers_incorrectly_claimed': 0}
            
            return {
                'real_outages': real_summary,
                'false_positives': false_summary,
                'impact_analysis': {
                    'false_positive_rate': stats.get('false_positive_rate', 0),
                    'customer_impact_reduction': stats.get('customer_impact_reduction', 0),
                    'accuracy_improvement': f"{100 - stats.get('false_positive_rate', 0):.1f}%"
                }
            }
        except Exception as e:
            logger.error(f"❌ Filtered summary generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _update_vector_database(self, real_outages: List[Dict]) -> None:
        """Update vector database with validated real outages."""
        try:
            if not self.vector_db_service:
                logger.warning("⚠️ Vector DB service not available")
                return
            
            # Convert real outages to DataFrame for the existing interface
            import pandas as pd
            
            # Create DataFrame from real outages
            real_outages_df = pd.DataFrame(real_outages)
            
            # The OutageVectorDB expects a DataFrame with specific columns
            # Map our validated data to the expected format
            if len(real_outages_df) > 0:
                # Ensure required columns exist
                if 'DATETIME' not in real_outages_df.columns:
                    real_outages_df['DATETIME'] = real_outages_df['datetime']
                if 'LATITUDE' not in real_outages_df.columns:
                    real_outages_df['LATITUDE'] = real_outages_df['latitude']
                if 'LONGITUDE' not in real_outages_df.columns:
                    real_outages_df['LONGITUDE'] = real_outages_df['longitude']
                if 'CUSTOMERS' not in real_outages_df.columns:
                    real_outages_df['CUSTOMERS'] = real_outages_df['customers']
                
                # Load the validated real outages into vector DB
                self.vector_db_service.load_outage_data(real_outages_df, force_reload=False)
                logger.info(f"✅ Added {len(real_outages)} validated outages to vector database")
            else:
                logger.info("ℹ️ No real outages to add to vector database")
            
        except Exception as e:
            logger.error(f"❌ Vector database update failed: {str(e)}")
    
    def _cache_validation_results(self, validation_results: Dict) -> None:
        """Cache validation results to disk."""
        try:
            import joblib
            cache_file = self.settings.validation_results_cache_file
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            joblib.dump(validation_results, cache_file)
            logger.info(f"💾 Validation results cached to {cache_file}")
        except Exception as e:
            logger.error(f"❌ Failed to cache validation results: {str(e)}")
    
    # ==================== PUBLIC INTERFACE ====================
    
    def run_validation_workflow(self, data_file: Optional[str] = None) -> OutageAnalysisState:
        """
        Run the complete validation workflow.
        
        Args:
            data_file: Optional path to data file (uses default if None)
            
        Returns:
            OutageAnalysisState: Final workflow state with results
        """
        try:
            # Initialize workflow state
            initial_state = OutageAnalysisState(
                dataset_loaded=False,
                validation_complete=False,
                raw_dataset_summary={},
                validation_results={},
                filtered_summary={},
                current_window_analysis=None,
                chat_context={}
            )
            
            # Override data file if provided
            if data_file:
                self.settings.default_data_file = data_file
            
            # Execute workflow
            logger.info("🚀 Starting validation workflow...")
            final_state = self.workflow.invoke(initial_state)
            
            logger.info("✅ Validation workflow completed successfully")
            return final_state
            
        except Exception as e:
            logger.error(f"❌ Validation workflow failed: {str(e)}")
            raise
    
    def validate_single_report(self, report_data: Dict) -> ValidationResult:
        """
        Validate a single outage report.
        
        Args:
            report_data: Single outage report data
            
        Returns:
            ValidationResult: Validation result for the report
        """
        try:
            # Get weather data for the report
            report_datetime = datetime.strptime(report_data['datetime'], "%Y-%m-%d %H:%M:%S")
            weather = self.weather_service.get_historical_weather(
                lat=report_data['latitude'],
                lon=report_data['longitude'], 
                date=report_datetime
            )
            
            # Validate using LLM
            validation_response = validate_outage_report.invoke({
                "outage_report": report_data,
                "weather_data": weather
            })
            
            # Determine classification
            is_real = "REAL OUTAGE" in validation_response.upper()
            classification = "real_outage" if is_real else "false_positive"
            
            return ValidationResult(
                record_id=f"{report_data['datetime']}_{report_data['latitude']}_{report_data['longitude']}",
                is_valid=is_real,
                confidence_score=0.8,  # Could implement confidence scoring
                weather_correlation=weather,
                llm_analysis=validation_response,
                classification=classification
            )
            
        except Exception as e:
            logger.error(f"❌ Single report validation failed: {str(e)}")
            raise
    
    def chat_with_results(self, question: str, context: Dict) -> str:
        """
        Chat interface for exploring validation results.
        
        Args:
            question: User question about the results
            context: Analysis context and results
            
        Returns:
            str: LLM response to the question
        """
        try:
            return chat_about_results.invoke({
                "question": question,
                "context": context
            })
        except Exception as e:
            logger.error(f"❌ Chat failed: {str(e)}")
            return f"Chat error: {str(e)}"
    
    def generate_report(self, validation_results: Dict, raw_summary: Dict, 
                       report_type: str = "comprehensive") -> str:
        """
        Generate analysis report.
        
        Args:
            validation_results: Validation results data
            raw_summary: Raw dataset summary
            report_type: Type of report ("comprehensive" or "exhaustive")
            
        Returns:
            str: Generated report content
        """
        try:
            if report_type == "exhaustive":
                return generate_exhaustive_report.invoke({
                    "validation_results": validation_results,
                    "raw_summary": raw_summary
                })
            else:
                return generate_comprehensive_report.invoke({
                    "validation_results": validation_results,
                    "raw_summary": raw_summary
                })
        except Exception as e:
            logger.error(f"❌ Report generation failed: {str(e)}")
            return f"Report generation error: {str(e)}"


# ==================== UTILITY FUNCTIONS ====================

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