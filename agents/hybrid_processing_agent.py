"""
Hybrid Processing Agent - Implements Dual-Write Pattern
- Creates lightweight .pkl files for UI rendering (summaries/metadata)
- Stores detailed analysis results with reasoning in vector database
- Maintains backward compatibility with existing UI code
"""

import os
import json
import logging
import time
import joblib
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Import services
from services.llm_service import LLMManager
from services.weather_service import WeatherService
from services.geocoding_service import GeocodingService
from services.hybrid_vector_service import HybridOutageVectorDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridProcessingAgent:
    """
    Processing agent that implements hybrid storage approach:
    1. Lightweight .pkl files for UI rendering
    2. Detailed analysis results in vector database
    """
    
    def __init__(self, vector_db_path: str = "./chroma_db"):
        """Initialize hybrid processing agent"""
        self.vector_db = HybridOutageVectorDB(vector_db_path)
        
        # Initialize services
        self.llm_manager = LLMManager()
        self.weather_service = WeatherService()
        self.geocoding_service = GeocodingService()
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        logger.info("✅ Hybrid Processing Agent initialized")
    
    def _load_prompts(self) -> Dict:
        """Load prompts from prompts.json"""
        try:
            with open('prompts.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("prompts.json file not found!")
            return {}
    
    def process_analysis(self, dataset_path: str, max_records: int = None) -> Dict:
        """
        Run complete analysis with hybrid storage
        
        Args:
            dataset_path: Path to the outage data CSV
            max_records: Limit processing for testing (None for all records)
            
        Returns:
            Dict: Analysis results with storage information
        """
        try:
            logger.info("🔄 Starting hybrid analysis processing...")
            start_time = time.time()
            
            # Step 1: Load dataset
            dataset_summary = self._load_dataset(dataset_path)
            
            # Step 2: Run validation analysis
            validation_results = self._run_validation_analysis(max_records)
            
            # Step 3: Create lightweight summary for UI
            ui_summary = self._create_ui_summary(validation_results, dataset_summary)
            
            # Step 4: Save lightweight .pkl file for UI
            pkl_file = self._save_lightweight_pkl(ui_summary)
            
            # Step 5: Store detailed analysis in vector database
            vector_storage_success = self._store_detailed_analysis(validation_results)
            
            # Step 6: Generate final results
            processing_time = time.time() - start_time
            
            final_results = {
                "processing_complete": True,
                "processing_time": {
                    "total_seconds": processing_time,
                    "completion_timestamp": datetime.now().isoformat()
                },
                "storage": {
                    "pkl_file": pkl_file,
                    "vector_db_updated": vector_storage_success,
                    "ui_summary": ui_summary
                },
                "analysis_overview": {
                    "total_records": validation_results.get("total_processed", 0),
                    "real_outages": len(validation_results.get("real_outages", [])),
                    "false_positives": len(validation_results.get("false_positives", [])),
                    "accuracy_rate": ui_summary.get("accuracy_rate", 0.0)
                }
            }
            
            logger.info(f"✅ Hybrid analysis completed in {processing_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Hybrid analysis failed: {str(e)}")
            return {"error": str(e), "processing_complete": False}
    
    def _load_dataset(self, dataset_path: str) -> Dict:
        """Load dataset into vector database"""
        try:
            if not os.path.exists(dataset_path):
                raise ValueError(f"Dataset file not found: {dataset_path}")
            
            # Read CSV data
            df = pd.read_csv(dataset_path)
            logger.info(f"📊 Loaded {len(df)} records from {dataset_path}")
            
            # Validate required columns
            required_columns = ['DATETIME', 'LATITUDE', 'LONGITUDE', 'CUSTOMERS']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Load into vector database (this also generates summary)
            summary = self.vector_db.load_outage_data(df, force_reload=True)
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Dataset loading failed: {str(e)}")
            raise
    
    def _run_validation_analysis(self, max_records: int = None) -> Dict:
        """Run validation analysis on loaded data"""
        try:
            logger.info("🔄 Running validation analysis...")
            
            # Get all raw data from vector database
            raw_collection = self.vector_db.raw_collection
            all_results = raw_collection.get()
            
            if not all_results or not all_results.get('metadatas'):
                raise ValueError("No raw data found in vector database")
            
            dataset_records = all_results['metadatas']
            
            # Apply record limit if specified
            if max_records and max_records < len(dataset_records):
                dataset_records = dataset_records[:max_records]
                logger.info(f"⚠️ Processing limited to {max_records} records")
            
            real_outages = []
            false_positives = []
            
            # Process each record
            for idx, record in enumerate(dataset_records):
                try:
                    # Get weather data for this outage
                    weather_data = self._get_weather_for_outage(record)
                    
                    # Classify the outage
                    classification_result = self._classify_outage(record, weather_data)
                    
                    # Store result with detailed information
                    enhanced_record = {
                        **record,
                        "weather_conditions": weather_data,
                        "confidence": classification_result.get("confidence", 0.8),
                        "reasoning": classification_result.get("reasoning", ""),
                        "severity": classification_result.get("severity", 5),
                        "analysis_timestamp": datetime.now().isoformat()
                    }
                    
                    if classification_result.get("classification") == "real_outage":
                        real_outages.append(enhanced_record)
                    else:
                        false_positives.append(enhanced_record)
                    
                    # Progress logging
                    if (idx + 1) % 10 == 0:
                        logger.info(f"📊 Processed {idx + 1}/{len(dataset_records)} records")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process record {idx}: {str(e)}")
                    continue
            
            validation_results = {
                "validation_complete": True,
                "total_processed": len(dataset_records),
                "real_outages": real_outages,
                "false_positives": false_positives,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Validation completed: {len(real_outages)} real, {len(false_positives)} false positives")
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Validation analysis failed: {str(e)}")
            raise
    
    def _get_weather_for_outage(self, outage_record: Dict) -> Dict:
        """Get weather data for an outage record"""
        try:
            latitude = outage_record.get('latitude', 0)
            longitude = outage_record.get('longitude', 0)
            datetime_str = outage_record.get('datetime', '')
            
            if latitude and longitude and datetime_str:
                return self.weather_service.get_historical_weather(latitude, longitude, datetime_str)
            else:
                return {"error": "Missing location or datetime data"}
                
        except Exception as e:
            logger.warning(f"Weather data retrieval failed: {str(e)}")
            return {"error": str(e)}
    
    def _classify_outage(self, outage_record: Dict, weather_data: Dict) -> Dict:
        """Classify an outage as real or false positive using LLM"""
        try:
            # Use weather validation prompt
            if 'weather_validation' in self.prompts:
                system_prompt = self.prompts["weather_validation"]["system"]
                human_prompt = self.prompts["weather_validation"]["human"]
            else:
                # Fallback prompt
                system_prompt = """You are an expert power grid analyst. Classify power outage reports as real outages or false positives based on weather conditions and other factors.

Consider:
1. Weather severity (wind speed, temperature, precipitation)
2. Customer count reported
3. Geographic location and time
4. Historical patterns

Respond with JSON: {"classification": "real_outage" or "false_positive", "confidence": 0.0-1.0, "reasoning": "detailed explanation", "severity": 1-10}"""
                
                human_prompt = """Outage Report:
Location: {latitude}, {longitude}
Date/Time: {datetime}
Customers: {customers}

Weather Data:
{weather_conditions}

Classify this outage:"""
            
            # Format inputs
            weather_str = json.dumps(weather_data, indent=2, default=str)
            
            # Get LLM classification
            from langchain_core.prompts import ChatPromptTemplate
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            chain = chat_prompt | self.llm_manager.get_llm()
            response = chain.invoke({
                "latitude": outage_record.get('latitude', 0),
                "longitude": outage_record.get('longitude', 0),
                "datetime": outage_record.get('datetime', ''),
                "customers": outage_record.get('customers', 0),
                "weather_conditions": weather_str
            })
            
            # Parse LLM response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            try:
                # Try to parse as JSON
                classification_result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback parsing
                if "real_outage" in response_text.lower():
                    classification_result = {
                        "classification": "real_outage",
                        "confidence": 0.7,
                        "reasoning": response_text,
                        "severity": 6
                    }
                else:
                    classification_result = {
                        "classification": "false_positive",
                        "confidence": 0.7,
                        "reasoning": response_text,
                        "severity": 3
                    }
            
            return classification_result
            
        except Exception as e:
            logger.warning(f"Classification failed: {str(e)}")
            return {
                "classification": "false_positive",
                "confidence": 0.5,
                "reasoning": f"Classification failed: {str(e)}",
                "severity": 1
            }
    
    def _create_ui_summary(self, validation_results: Dict, dataset_summary: Dict) -> Dict:
        """Create lightweight summary for UI rendering"""
        try:
            real_outages = validation_results.get("real_outages", [])
            false_positives = validation_results.get("false_positives", [])
            total = len(real_outages) + len(false_positives)
            
            ui_summary = {
                "validation_complete": True,
                "processing_timestamp": datetime.now().isoformat(),
                "totals": {
                    "real_outages": len(real_outages),
                    "false_positives": len(false_positives),
                    "total_processed": total,
                    "accuracy_rate": len(real_outages) / total if total > 0 else 0.0
                },
                "confidence_stats": {
                    "avg_real_confidence": sum([r.get('confidence', 0.8) for r in real_outages]) / len(real_outages) if real_outages else 0.0,
                    "avg_false_confidence": sum([f.get('confidence', 0.8) for f in false_positives]) / len(false_positives) if false_positives else 0.0
                },
                "geographic_summary": {
                    "real_outage_locations": [(r.get('latitude', 0), r.get('longitude', 0)) for r in real_outages[:50]],  # Limit for UI
                    "false_positive_locations": [(f.get('latitude', 0), f.get('longitude', 0)) for f in false_positives[:50]]
                },
                "dataset_info": dataset_summary,
                "storage_type": "hybrid"  # Indicator for UI
            }
            
            return ui_summary
            
        except Exception as e:
            logger.error(f"❌ UI summary creation failed: {str(e)}")
            return {"error": str(e)}
    
    def _save_lightweight_pkl(self, ui_summary: Dict) -> str:
        """Save lightweight .pkl file for UI"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pkl_filename = f"cache/analysis_results_{timestamp}.pkl"
            os.makedirs("cache", exist_ok=True)
            
            # Save only the UI summary, not detailed results
            joblib.dump(ui_summary, pkl_filename)
            
            logger.info(f"✅ Lightweight .pkl saved: {pkl_filename}")
            return pkl_filename
            
        except Exception as e:
            logger.error(f"❌ Failed to save .pkl file: {str(e)}")
            return ""
    
    def _store_detailed_analysis(self, validation_results: Dict) -> bool:
        """Store detailed analysis results in vector database"""
        try:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            success = self.vector_db.store_analysis_results(validation_results, batch_id)
            
            if success:
                logger.info("✅ Detailed analysis stored in vector database")
            else:
                logger.error("❌ Failed to store analysis in vector database")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Vector database storage failed: {str(e)}")
            return False

# Helper function for backward compatibility
def run_hybrid_analysis(dataset_path: str, max_records: int = None) -> Dict:
    """Run analysis with hybrid storage approach"""
    agent = HybridProcessingAgent()
    return agent.process_analysis(dataset_path, max_records)

if __name__ == "__main__":
    # Test the hybrid processing agent
    print("Testing Hybrid Processing Agent...")
    
    # Test with sample data
    test_results = run_hybrid_analysis("data/raw_data.csv", max_records=5)
    
    print(f"Results: {test_results}")
    print("✅ Hybrid Processing Agent test completed") 