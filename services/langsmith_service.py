"""
LangSmith Monitoring Service

Provides comprehensive monitoring, tracing, and analytics for LLM operations
using LangSmith. Tracks pricing, performance, token usage, and error rates.

Key Features:
- Automatic tracing of LLM calls
- Cost tracking and analysis
- Performance metrics (latency, throughput)
- Token usage monitoring
- Error tracking and analysis
- Project-based organization
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from langsmith import Client
# LangSmith imports - simplified approach
import json

logger = logging.getLogger(__name__)


class LangSmithMonitor:
    """
    LangSmith monitoring and analytics service.
    
    Handles project configuration, tracing setup, and data collection
    for comprehensive LLM monitoring.
    """
    
    def __init__(self, project_name: str = "power-agent", auto_trace: bool = True):
        """
        Initialize LangSmith monitoring.
        
        Args:
            project_name: Name of the LangSmith project
            auto_trace: Whether to automatically trace LLM calls
        """
        self.project_name = project_name
        self.auto_trace = auto_trace
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize LangSmith client with authentication."""
        try:
            # Set up environment variables for LangSmith
            api_key = os.getenv("LANGSMITH_API_KEY")
            if not api_key:
                logger.warning("LANGSMITH_API_KEY not found. LangSmith monitoring disabled.")
                return
                
            # Set project name in environment
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            
            # Initialize client
            self.client = Client(api_key=api_key)
            logger.info(f"LangSmith client initialized for project: {self.project_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}")
            self.client = None
    
    def is_enabled(self) -> bool:
        """Check if LangSmith monitoring is enabled."""
        return self.client is not None
    
    def get_project_info(self) -> Dict[str, Any]:
        """Get information about the current project."""
        if not self.is_enabled():
            return {"error": "LangSmith not enabled"}
            
        try:
            # Get project info
            projects = list(self.client.list_projects())
            current_project = next(
                (p for p in projects if p.name == self.project_name), 
                None
            )
            
            if current_project:
                return {
                    "name": current_project.name,
                    "id": str(current_project.id),
                    "created_at": getattr(current_project, 'created_at', None).isoformat() if hasattr(current_project, 'created_at') and current_project.created_at else None,
                    "description": getattr(current_project, 'description', None),
                }
            else:
                return {"error": f"Project {self.project_name} not found"}
                
        except Exception as e:
            logger.error(f"Error getting project info: {e}")
            return {"error": str(e)}
    
    def get_usage_metrics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get usage metrics for the specified time period.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary containing usage metrics
        """
        if not self.is_enabled():
            return {"error": "LangSmith not enabled"}
            
        try:
            # Calculate date range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Get runs for the project
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time,
                end_time=end_time
            ))
            
            # Calculate metrics
            total_runs = len(runs)
            successful_runs = sum(1 for run in runs if run.status == "success")
            failed_runs = sum(1 for run in runs if run.status == "error")
            
            # Calculate costs and tokens
            total_input_tokens = 0
            total_output_tokens = 0
            total_cost = 0.0
            latencies = []
            
            for run in runs:
                # Extract token usage and cost from multiple possible locations
                usage_found = False
                cost_found = False
                
                # Check run.extra (primary location)
                if run.extra and isinstance(run.extra, dict):
                    # Extract token usage
                    if "usage" in run.extra:
                        usage = run.extra["usage"]
                        total_input_tokens += usage.get("input_tokens", 0)
                        total_output_tokens += usage.get("output_tokens", 0)
                        usage_found = True
                    
                    # Extract cost information
                    if "cost" in run.extra:
                        total_cost += run.extra["cost"]
                        cost_found = True
                    
                    # Check for our cost tracker data
                    if "langsmith_cost_tracking" in run.extra:
                        tracking_data = run.extra["langsmith_cost_tracking"]
                        if not usage_found and "usage" in tracking_data:
                            usage = tracking_data["usage"]
                            total_input_tokens += usage.get("input_tokens", 0)
                            total_output_tokens += usage.get("output_tokens", 0)
                            usage_found = True
                        
                        if not cost_found and "cost" in tracking_data:
                            total_cost += tracking_data["cost"]
                            cost_found = True
                
                # Check run metadata (alternative location)
                if hasattr(run, 'metadata') and run.metadata:
                    if not usage_found and "usage" in run.metadata:
                        usage = run.metadata["usage"]
                        total_input_tokens += usage.get("input_tokens", 0)
                        total_output_tokens += usage.get("output_tokens", 0)
                    
                    if not cost_found and "cost" in run.metadata:
                        total_cost += run.metadata["cost"]
                
                # Calculate latency
                if run.start_time and run.end_time:
                    latency = (run.end_time - run.start_time).total_seconds()
                    latencies.append(latency)
            
            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
            error_rate = (failed_runs / total_runs * 100) if total_runs > 0 else 0
            
            return {
                "period_days": days,
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "error_rate_percent": round(error_rate, 2),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "estimated_cost_usd": round(total_cost, 4),
                "avg_latency_seconds": round(avg_latency, 3),
                "p95_latency_seconds": round(p95_latency, 3),
                "runs_per_day": round(total_runs / days, 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting usage metrics: {e}")
            return {"error": str(e)}
    
    def get_model_breakdown(self, days: int = 7) -> Dict[str, Any]:
        """
        Get usage breakdown by model.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary containing model-specific metrics
        """
        if not self.is_enabled():
            return {"error": "LangSmith not enabled"}
            
        try:
            # Calculate date range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Get runs for the project
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time,
                end_time=end_time
            ))
            
            # Group by model
            model_stats = {}
            
            for run in runs:
                # Extract model name from run metadata
                model_name = "unknown"
                if run.extra and isinstance(run.extra, dict):
                    model_name = run.extra.get("model", "unknown")
                elif hasattr(run, 'serialized') and run.serialized:
                    model_name = run.serialized.get("model", "unknown")
                
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        "runs": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost": 0.0,
                        "latencies": []
                    }
                
                stats = model_stats[model_name]
                stats["runs"] += 1
                
                # Extract metrics
                if run.extra and isinstance(run.extra, dict):
                    if "usage" in run.extra:
                        usage = run.extra["usage"]
                        stats["input_tokens"] += usage.get("input_tokens", 0)
                        stats["output_tokens"] += usage.get("output_tokens", 0)
                    
                    if "cost" in run.extra:
                        stats["cost"] += run.extra["cost"]
                
                # Calculate latency
                if run.start_time and run.end_time:
                    latency = (run.end_time - run.start_time).total_seconds()
                    stats["latencies"].append(latency)
            
            # Calculate final statistics
            result = {}
            for model, stats in model_stats.items():
                latencies = stats["latencies"]
                result[model] = {
                    "runs": stats["runs"],
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "total_tokens": stats["input_tokens"] + stats["output_tokens"],
                    "estimated_cost_usd": round(stats["cost"], 4),
                    "avg_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else 0,
                    "cost_per_1k_tokens": round(
                        (stats["cost"] / (stats["input_tokens"] + stats["output_tokens"]) * 1000), 4
                    ) if (stats["input_tokens"] + stats["output_tokens"]) > 0 else 0
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting model breakdown: {e}")
            return {"error": str(e)}
    
    def export_analytics_data(self, days: int = 30, format: str = "json") -> str:
        """
        Export comprehensive analytics data.
        
        Args:
            days: Number of days to export
            format: Export format ("json" or "csv")
            
        Returns:
            Path to exported file
        """
        if not self.is_enabled():
            raise ValueError("LangSmith not enabled")
            
        try:
            # Get comprehensive data
            usage_metrics = self.get_usage_metrics(days)
            model_breakdown = self.get_model_breakdown(days)
            project_info = self.get_project_info()
            
            # Create export data
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "project_info": project_info,
                "usage_metrics": usage_metrics,
                "model_breakdown": model_breakdown,
                "period_days": days
            }
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"langsmith_analytics_{self.project_name}_{timestamp}.{format}"
            filepath = os.path.join("reports", filename)
            
            # Ensure reports directory exists
            os.makedirs("reports", exist_ok=True)
            
            # Export data
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
            else:
                # For CSV, we'd need to flatten the data structure
                # This is a simplified version
                with open(filepath, 'w') as f:
                    f.write("metric,value\n")
                    for key, value in usage_metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"{key},{value}\n")
            
            logger.info(f"Analytics data exported to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting analytics data: {e}")
            raise
    
    def setup_environment_variables(self):
        """
        Helper method to print environment variable setup instructions.
        """
        instructions = """
        To enable LangSmith monitoring, set these environment variables:
        
        1. Get your API key from https://smith.langchain.com
        2. Add to your .env file:
           LANGSMITH_API_KEY=your_api_key_here
           LANGCHAIN_PROJECT=power-agent
           LANGCHAIN_TRACING_V2=true
        
        3. Restart your application
        """
        return instructions


def create_langsmith_monitor(project_name: str = "power-agent") -> LangSmithMonitor:
    """
    Factory function to create a LangSmith monitor instance.
    
    Args:
        project_name: Name of the LangSmith project
        
    Returns:
        Configured LangSmithMonitor instance
    """
    return LangSmithMonitor(project_name=project_name)