"""
Standalone Report Generation Agent - Reads .pkl files and generates comprehensive reports
Uses MCP calls for additional context and detailed analysis
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

# Import services for MCP calls
from services.llm_service import LLMManager
from services.weather_service import WeatherService
from services.geocoding_service import GeocodingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportAgent:
    """Standalone report generation agent that reads .pkl files and creates comprehensive reports"""
    
    def __init__(self, pkl_file_path: str = None):
        """Initialize report agent with .pkl file"""
        self.pkl_file_path = pkl_file_path
        self.results = None
        
        # Initialize services for MCP calls
        self.llm_manager = LLMManager()
        self.weather_service = WeatherService()
        self.geocoding_service = GeocodingService()
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Load .pkl file if provided
        if pkl_file_path:
            self.load_results(pkl_file_path)
    
    def _load_prompts(self) -> Dict:
        """Load prompts from prompts.json"""
        try:
            with open('prompts.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("prompts.json file not found!")
            return {}
    
    def load_results(self, pkl_file_path: str) -> bool:
        """Load analysis results from .pkl file"""
        try:
            if not os.path.exists(pkl_file_path):
                logger.error(f"PKL file not found: {pkl_file_path}")
                return False
            
            self.results = joblib.load(pkl_file_path)
            self.pkl_file_path = pkl_file_path
            logger.info(f"✅ Loaded results from {pkl_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load .pkl file: {str(e)}")
            return False
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary using LLM"""
        if not self.results:
            return "No analysis results loaded"
        
        try:
            # Use comprehensive_report_generation prompt if available
            if 'comprehensive_report_generation' in self.prompts:
                report_prompt = ChatPromptTemplate.from_messages([
                    ("system", self.prompts["comprehensive_report_generation"]["system"]),
                    ("human", self.prompts["comprehensive_report_generation"]["human"])
                ])
            else:
                # Fallback prompt
                report_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Generate a comprehensive executive summary of power outage analysis results."),
                    ("human", "Analysis results: {validation_results}\nRaw data summary: {raw_summary}")
                ])
            
            validation_results = self.results.get('validation_results', {})
            raw_summary = self.results.get('raw_dataset_summary', {})
            
            # Create time period info
            time_period = "Unknown"
            if raw_summary.get('date_range'):
                start = raw_summary['date_range'].get('start', 'Unknown')
                end = raw_summary['date_range'].get('end', 'Unknown')
                time_period = f"{start} to {end}"
            
            chain = report_prompt | self.llm_manager.get_llm()
            response = chain.invoke({
                "validation_results": json.dumps(validation_results, indent=2, default=str),
                "raw_summary": json.dumps(raw_summary, indent=2, default=str),
                "time_period": time_period,
                "map_data": json.dumps({"total_locations": len(validation_results.get('real_outages', []))}, indent=2)
            })
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Executive summary generation error: {str(e)}")
            return f"Error generating executive summary: {str(e)}"
    
    def generate_statistical_report(self) -> Dict:
        """Generate detailed statistical analysis"""
        if not self.results:
            return {"error": "No analysis results loaded"}
        
        validation_results = self.results.get('validation_results', {})
        real_outages = validation_results.get('real_outages', [])
        false_positives = validation_results.get('false_positives', [])
        
        total_reports = len(real_outages) + len(false_positives)
        
        if total_reports == 0:
            return {"error": "No validation results to analyze"}
        
        # Basic statistics
        accuracy_rate = len(real_outages) / total_reports if total_reports > 0 else 0
        
        # Confidence analysis
        real_confidences = [r.get('confidence', 0.8) for r in real_outages if 'confidence' in r]
        false_confidences = [f.get('confidence', 0.8) for f in false_positives if 'confidence' in f]
        
        # Customer impact analysis
        real_customers = [r.get('CUSTOMERS', 0) for r in real_outages if 'CUSTOMERS' in r]
        false_customers = [f.get('CUSTOMERS', 0) for f in false_positives if 'CUSTOMERS' in f]
        
        return {
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
            "customer_impact": {
                "real_total_customers": sum(real_customers),
                "false_total_customers": sum(false_customers),
                "real_avg_customers": float(np.mean(real_customers)) if real_customers else 0.0,
                "false_avg_customers": float(np.mean(false_customers)) if false_customers else 0.0
            }
        }
    
    def generate_map_report(self) -> str:
        """Generate map visualization and save to file"""
        if not self.results:
            return "No analysis results loaded"
        
        validation_results = self.results.get('validation_results', {})
        real_outages = validation_results.get('real_outages', [])
        false_positives = validation_results.get('false_positives', [])
        
        if not real_outages and not false_positives:
            return "No data available for mapping"
        
        try:
            # Calculate map center
            all_coords = []
            for outage in real_outages + false_positives:
                if outage.get('LATITUDE') and outage.get('LONGITUDE'):
                    all_coords.append([float(outage['LATITUDE']), float(outage['LONGITUDE'])])
            
            if not all_coords:
                return "No coordinate data available"
            
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
                if outage.get('LATITUDE') and outage.get('LONGITUDE'):
                    folium.CircleMarker(
                        location=[float(outage['LATITUDE']), float(outage['LONGITUDE'])],
                        radius=8,
                        popup=f"Real Outage - {outage.get('CUSTOMERS', 'Unknown')} customers",
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.7
                    ).add_to(m)
            
            # Add false positives (blue markers)
            for outage in false_positives:
                if outage.get('LATITUDE') and outage.get('LONGITUDE'):
                    folium.CircleMarker(
                        location=[float(outage['LATITUDE']), float(outage['LONGITUDE'])],
                        radius=6,
                        popup=f"False Positive - {outage.get('CUSTOMERS', 'Unknown')} customers",
                        color='blue',
                        fill=True,
                        fillColor='lightblue',
                        fillOpacity=0.5
                    ).add_to(m)
            
            # Save map
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            map_filename = f"cache/report_map_{timestamp}.html"
            os.makedirs("cache", exist_ok=True)
            m.save(map_filename)
            
            return map_filename
            
        except Exception as e:
            logger.error(f"Map generation error: {str(e)}")
            return f"Error generating map: {str(e)}"
    
    def generate_comprehensive_report(self, report_type: str = "standard") -> Dict:
        """Generate comprehensive report with multiple sections"""
        if not self.results:
            return {"error": "No analysis results loaded"}
        
        logger.info(f"Generating {report_type} report...")
        
        # Generate all report sections
        executive_summary = self.generate_executive_summary()
        statistical_report = self.generate_statistical_report()
        map_filename = self.generate_map_report()
        
        # Create comprehensive report
        report = {
            "report_metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "report_type": report_type,
                "source_pkl_file": self.pkl_file_path,
                "report_version": "1.0"
            },
            "executive_summary": executive_summary,
            "statistical_analysis": statistical_report,
            "map_visualization": map_filename,
            "raw_data_summary": self.results.get('raw_dataset_summary', {}),
            "validation_details": self.results.get('validation_results', {}),
            "processing_metadata": {
                "processing_time": self.results.get('processing_time', {}),
                "total_records": self.results.get('total_records', 0),
                "processed_count": self.results.get('processed_count', 0)
            }
        }
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"cache/comprehensive_report_{timestamp}.json"
        
        try:
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            report["report_file"] = report_filename
            logger.info(f"✅ Report saved to {report_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
            report["save_error"] = str(e)
        
        return report
    
    def export_report_formats(self, report_data: Dict, formats: List[str] = ["json", "txt"]) -> Dict:
        """Export report in multiple formats"""
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            os.makedirs("cache", exist_ok=True)
            
            if "json" in formats:
                json_file = f"cache/report_{timestamp}.json"
                with open(json_file, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                exported_files["json"] = json_file
            
            if "txt" in formats:
                txt_file = f"cache/report_{timestamp}.txt"
                with open(txt_file, 'w') as f:
                    f.write(f"Power Outage Analysis Report\n")
                    f.write(f"Generated: {report_data.get('report_metadata', {}).get('generation_timestamp', 'Unknown')}\n")
                    f.write(f"=" * 50 + "\n\n")
                    f.write(f"EXECUTIVE SUMMARY:\n{report_data.get('executive_summary', 'Not available')}\n\n")
                    f.write(f"STATISTICAL ANALYSIS:\n{json.dumps(report_data.get('statistical_analysis', {}), indent=2, default=str)}\n")
                exported_files["txt"] = txt_file
            
            return {"status": "success", "files": exported_files}
            
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            return {"status": "error", "error": str(e)}

# Helper function for UI integration
def create_report_agent(pkl_file_path: str) -> ReportAgent:
    """Create a report agent instance with loaded .pkl file"""
    return ReportAgent(pkl_file_path)

if __name__ == "__main__":
    # Test the report agent
    import glob
    
    # Find the latest .pkl file
    pkl_files = glob.glob("cache/analysis_results_*.pkl")
    if pkl_files:
        latest_pkl = max(pkl_files, key=os.path.getctime)
        print(f"Testing report agent with: {latest_pkl}")
        
        agent = ReportAgent(latest_pkl)
        
        # Generate comprehensive report
        report = agent.generate_comprehensive_report("standard")
        print("Report generated:", report.get("report_file", "No file"))
        
        # Export in multiple formats
        export_result = agent.export_report_formats(report, ["json", "txt"])
        print("Exported files:", export_result.get("files", {}))
    else:
        print("No .pkl files found in cache/ directory")