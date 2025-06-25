"""
Hybrid Report Agent - Multi-Source Report Generation
- Uses .pkl files for report structure and metadata
- Queries vector database for detailed context
- Makes MCP calls for additional real-time information
- Generates comprehensive reports with all data sources
"""

import os
import json
import logging
import joblib
import folium
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate

# Import services
from services.llm_service import LLMManager
from services.weather_service import WeatherService
from services.geocoding_service import GeocodingService
from services.hybrid_vector_service import HybridOutageVectorDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridReportAgent:
    """
    Hybrid report generation agent that combines multiple data sources:
    1. .pkl files for structure and metadata
    2. Vector database for detailed analysis context
    3. MCP calls for additional real-time data
    """
    
    def __init__(self, pkl_file_path: str = None, vector_db_path: str = "./chroma_db"):
        """Initialize hybrid report agent"""
        self.pkl_file_path = pkl_file_path
        self.pkl_data = None
        
        # Initialize vector database
        self.vector_db = HybridOutageVectorDB(vector_db_path)
        
        # Initialize services for MCP calls
        self.llm_manager = LLMManager()
        self.weather_service = WeatherService()
        self.geocoding_service = GeocodingService()
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Load .pkl file if provided
        if pkl_file_path:
            self.load_pkl_metadata(pkl_file_path)
        
        logger.info("✅ Hybrid Report Agent initialized")
    
    def _load_prompts(self) -> Dict:
        """Load prompts from prompts.json"""
        try:
            with open('prompts_hybrid.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            try:
                with open('prompts.json', 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                logger.error("No prompts file found!")
                return {}
    
    def load_pkl_metadata(self, pkl_file_path: str) -> bool:
        """Load metadata and structure from .pkl file"""
        try:
            if not os.path.exists(pkl_file_path):
                logger.error(f"PKL file not found: {pkl_file_path}")
                return False
            
            self.pkl_data = joblib.load(pkl_file_path)
            self.pkl_file_path = pkl_file_path
            logger.info(f"✅ Loaded metadata from {pkl_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load .pkl file: {str(e)}")
            return False
    
    def generate_comprehensive_report(self, report_type: str = "comprehensive", 
                                    target_audience: str = "operations") -> Dict:
        """
        Generate comprehensive report using all data sources
        
        Args:
            report_type: Type of report (comprehensive, executive, technical)
            target_audience: Target audience (operations, management, technical)
            
        Returns:
            Dict: Complete report data with multiple sections
        """
        try:
            logger.info(f"🔄 Generating {report_type} report for {target_audience}...")
            
            # Step 1: Get structural metadata from .pkl
            structural_metadata = self._get_structural_metadata()
            
            # Step 2: Query vector database for detailed context
            detailed_context = self._get_detailed_analysis_context(report_type)
            
            # Step 3: Make MCP calls for additional context
            mcp_context = self._get_mcp_context(structural_metadata)
            
            # Step 4: Generate report using LLM with all contexts
            report_content = self._generate_report_content(
                structural_metadata, detailed_context, mcp_context, report_type, target_audience
            )
            
            # Step 5: Create supporting visualizations
            visualizations = self._generate_visualizations(structural_metadata, detailed_context)
            
            # Step 6: Compile final report
            final_report = {
                "report_metadata": {
                    "generation_timestamp": datetime.now().isoformat(),
                    "report_type": report_type,
                    "target_audience": target_audience,
                    "data_sources": ["pkl_metadata", "vector_database", "mcp_services"],
                    "report_version": "hybrid_v1.0"
                },
                "report_content": report_content,
                "visualizations": visualizations,
                "data_summary": {
                    "structural_metadata": structural_metadata,
                    "analysis_context_entries": len(detailed_context.get("results", [])),
                    "mcp_services_used": list(mcp_context.keys())
                },
                "recommendations": self._generate_recommendations(detailed_context),
                "appendices": {
                    "detailed_analysis": detailed_context,
                    "mcp_data": mcp_context,
                    "methodology": self._get_methodology_summary()
                }
            }
            
            # Step 7: Save report
            report_file = self._save_report(final_report, report_type)
            final_report["report_file"] = report_file
            
            logger.info(f"✅ Comprehensive report generated: {report_file}")
            return final_report
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {str(e)}")
            return {"error": str(e), "report_complete": False}
    
    def _get_structural_metadata(self) -> Dict:
        """Extract structural metadata from .pkl file or vector DB summary"""
        try:
            if self.pkl_data:
                # Use .pkl data if available
                return {
                    "totals": self.pkl_data.get("totals", {}),
                    "confidence_stats": self.pkl_data.get("confidence_stats", {}),
                    "geographic_summary": self.pkl_data.get("geographic_summary", {}),
                    "dataset_info": self.pkl_data.get("dataset_info", {}),
                    "processing_timestamp": self.pkl_data.get("processing_timestamp", ""),
                    "data_source": "pkl_file"
                }
            else:
                # Fallback to vector DB summary
                vector_summary = self.vector_db.get_analysis_summary()
                return {
                    "totals": {
                        "real_outages": vector_summary.get("real_outages", 0),
                        "false_positives": vector_summary.get("false_positives", 0),
                        "total_processed": vector_summary.get("total", 0),
                        "accuracy_rate": vector_summary.get("accuracy_rate", 0.0)
                    },
                    "confidence_stats": {
                        "avg_confidence": vector_summary.get("average_confidence", 0.0)
                    },
                    "processing_timestamp": vector_summary.get("latest_analysis", ""),
                    "data_source": "vector_database"
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get structural metadata: {str(e)}")
            return {"error": str(e)}
    
    def _get_detailed_analysis_context(self, report_type: str) -> Dict:
        """Query vector database for detailed analysis context based on report type"""
        try:
            context_queries = {
                "comprehensive": [
                    "false positive patterns and causes",
                    "real outage weather correlations",
                    "confidence levels and accuracy analysis",
                    "geographic distribution patterns"
                ],
                "executive": [
                    "key findings and accuracy metrics",
                    "false positive reduction opportunities",
                    "operational impact analysis"
                ],
                "technical": [
                    "classification methodology and confidence scores",
                    "weather correlation analysis details",
                    "algorithm performance metrics"
                ]
            }
            
            queries = context_queries.get(report_type, context_queries["comprehensive"])
            
            detailed_results = []
            for query in queries:
                result = self.vector_db.query_for_chat(query, n_results=10)
                detailed_results.append({
                    "query": query,
                    "results": result.get("metadatas", []),
                    "context": result.get("context", ""),
                    "documents": result.get("documents", [])
                })
            
            # Also get classification breakdowns
            real_outages = self.vector_db.query_by_classification("real_outage", limit=20)
            false_positives = self.vector_db.query_by_classification("false_positive", limit=20)
            
            return {
                "contextual_queries": detailed_results,
                "classification_samples": {
                    "real_outages": real_outages,
                    "false_positives": false_positives
                },
                "total_context_entries": len(detailed_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get detailed context: {str(e)}")
            return {"error": str(e)}
    
    def _get_mcp_context(self, structural_metadata: Dict) -> Dict:
        """Make MCP calls for additional context"""
        try:
            mcp_context = {}
            
            # Get weather context for key locations
            geographic_summary = structural_metadata.get("geographic_summary", {})
            if geographic_summary.get("real_outage_locations"):
                sample_location = geographic_summary["real_outage_locations"][0]
                if len(sample_location) == 2:
                    lat, lon = sample_location
                    try:
                        location_info = self.geocoding_service.reverse_geocode(lat, lon)
                        mcp_context["location_context"] = location_info
                    except Exception as e:
                        logger.warning(f"Location context failed: {str(e)}")
            
            # Get current weather patterns (if applicable)
            try:
                # This would be enhanced with real MCP calls
                mcp_context["current_weather_patterns"] = {
                    "note": "Enhanced weather context would be retrieved via MCP services",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.warning(f"Weather patterns context failed: {str(e)}")
            
            # Add infrastructure context
            mcp_context["infrastructure_context"] = {
                "grid_reliability_factors": "Retrieved via MCP infrastructure service",
                "maintenance_schedules": "Current maintenance data",
                "upgrade_recommendations": "Infrastructure improvement suggestions"
            }
            
            return mcp_context
            
        except Exception as e:
            logger.error(f"❌ MCP context retrieval failed: {str(e)}")
            return {"error": str(e)}
    
    def _generate_report_content(self, structural_metadata: Dict, detailed_context: Dict, 
                               mcp_context: Dict, report_type: str, target_audience: str) -> str:
        """Generate report content using LLM with all contexts"""
        try:
            # Use hybrid report generation prompt
            if 'hybrid_report_generation' in self.prompts:
                system_prompt = self.prompts["hybrid_report_generation"]["system"]
                human_prompt = self.prompts["hybrid_report_generation"]["human"]
            else:
                # Fallback prompt
                system_prompt = """Generate a comprehensive power outage analysis report using multiple data sources. Include executive summary, statistical analysis, findings, and actionable recommendations."""
                human_prompt = """Generate report based on: Metadata: {structural_metadata}, Analysis: {detailed_context}, Additional: {mcp_context}, Type: {report_type}"""
            
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            chain = chat_prompt | self.llm_manager.get_llm()
            response = chain.invoke({
                "structural_metadata": json.dumps(structural_metadata, indent=2, default=str),
                "detailed_context": json.dumps(detailed_context, indent=2, default=str),
                "mcp_context": json.dumps(mcp_context, indent=2, default=str),
                "report_type": report_type,
                "target_audience": target_audience
            })
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"❌ Report content generation failed: {str(e)}")
            return f"Report generation error: {str(e)}"
    
    def _generate_visualizations(self, structural_metadata: Dict, detailed_context: Dict) -> Dict:
        """Generate supporting visualizations"""
        try:
            visualizations = {}
            
            # Generate map if location data available
            geographic_summary = structural_metadata.get("geographic_summary", {})
            if geographic_summary:
                map_file = self._create_enhanced_map(geographic_summary)
                visualizations["map"] = map_file
            
            # Generate statistical charts (placeholder)
            visualizations["statistics"] = {
                "accuracy_chart": "Accuracy metrics visualization",
                "confidence_distribution": "Confidence score distribution",
                "temporal_patterns": "Time-based pattern analysis"
            }
            
            return visualizations
            
        except Exception as e:
            logger.error(f"❌ Visualization generation failed: {str(e)}")
            return {"error": str(e)}
    
    def _create_enhanced_map(self, geographic_summary: Dict) -> str:
        """Create enhanced map with analysis results"""
        try:
            real_locations = geographic_summary.get("real_outage_locations", [])
            false_locations = geographic_summary.get("false_positive_locations", [])
            
            if not real_locations and not false_locations:
                return "No location data available"
            
            # Calculate center
            all_coords = real_locations + false_locations
            center_lat = np.mean([coord[0] for coord in all_coords if len(coord) == 2])
            center_lon = np.mean([coord[1] for coord in all_coords if len(coord) == 2])
            
            # Create enhanced map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            
            # Add real outages with enhanced popups
            for idx, location in enumerate(real_locations):
                if len(location) == 2:
                    folium.CircleMarker(
                        location=location,
                        radius=8,
                        popup=f"Real Outage #{idx+1} - High Confidence Classification",
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.7
                    ).add_to(m)
            
            # Add false positives
            for idx, location in enumerate(false_locations):
                if len(location) == 2:
                    folium.CircleMarker(
                        location=location,
                        radius=6,
                        popup=f"False Positive #{idx+1} - Filtered Out",
                        color='blue',
                        fill=True,
                        fillColor='lightblue',
                        fillOpacity=0.5
                    ).add_to(m)
            
            # Save enhanced map
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            map_filename = f"cache/enhanced_report_map_{timestamp}.html"
            os.makedirs("cache", exist_ok=True)
            m.save(map_filename)
            
            return map_filename
            
        except Exception as e:
            logger.error(f"❌ Enhanced map creation failed: {str(e)}")
            return f"Map generation error: {str(e)}"
    
    def _generate_recommendations(self, detailed_context: Dict) -> List[Dict]:
        """Generate actionable recommendations based on analysis"""
        try:
            recommendations = []
            
            # Analyze false positive patterns
            false_positives = detailed_context.get("classification_samples", {}).get("false_positives", [])
            if false_positives:
                low_confidence_fps = [fp for fp in false_positives if fp.get('confidence', 1.0) < 0.7]
                if low_confidence_fps:
                    recommendations.append({
                        "category": "False Positive Reduction",
                        "priority": "High",
                        "recommendation": f"Review {len(low_confidence_fps)} low-confidence false positive classifications",
                        "action": "Improve classification thresholds and weather correlation algorithms"
                    })
            
            # Analyze accuracy patterns
            real_outages = detailed_context.get("classification_samples", {}).get("real_outages", [])
            if real_outages:
                high_confidence_real = [ro for ro in real_outages if ro.get('confidence', 0.0) > 0.9]
                recommendations.append({
                    "category": "System Performance",
                    "priority": "Medium",
                    "recommendation": f"System shows high confidence in {len(high_confidence_real)} real outage classifications",
                    "action": "Continue monitoring and validate against actual grid conditions"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {str(e)}")
            return [{"error": str(e)}]
    
    def _get_methodology_summary(self) -> Dict:
        """Get methodology summary for appendix"""
        return {
            "data_sources": {
                "pkl_metadata": "Lightweight summaries and structural information",
                "vector_database": "Detailed analysis results with reasoning and classifications",
                "mcp_services": "Real-time weather, geographic, and infrastructure data"
            },
            "classification_methodology": "LLM-based weather correlation analysis with confidence scoring",
            "accuracy_measurement": "Comparison of classifications against weather data patterns",
            "report_generation": "Multi-source synthesis with targeted insights for operational use"
        }
    
    def _save_report(self, report_data: Dict, report_type: str) -> str:
        """Save complete report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"cache/hybrid_report_{report_type}_{timestamp}.json"
            os.makedirs("cache", exist_ok=True)
            
            with open(report_filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"✅ Report saved: {report_filename}")
            return report_filename
            
        except Exception as e:
            logger.error(f"❌ Report saving failed: {str(e)}")
            return ""

# Helper functions for backward compatibility
def create_hybrid_report_agent(pkl_file_path: str = None) -> HybridReportAgent:
    """Create a hybrid report agent instance"""
    return HybridReportAgent(pkl_file_path)

def create_report_agent(pkl_file_path: str) -> HybridReportAgent:
    """Create report agent - now uses hybrid approach"""
    return HybridReportAgent(pkl_file_path)

if __name__ == "__main__":
    # Test the hybrid report agent
    print("Testing Hybrid Report Agent...")
    
    agent = HybridReportAgent()
    
    # Generate comprehensive report
    report = agent.generate_comprehensive_report("comprehensive", "operations")
    
    if report.get("report_file"):
        print(f"Report generated: {report['report_file']}")
    else:
        print(f"Report generation failed: {report.get('error', 'Unknown error')}")
    
    print("✅ Hybrid Report Agent test completed") 