#!/usr/bin/env python3
"""
LangSmith Report Export Script

This script exports comprehensive monitoring and analytics reports from LangSmith
for your power outage analysis application.

Usage:
    python export_langsmith_report.py [days] [format]
    
Examples:
    python export_langsmith_report.py 30 json
    python export_langsmith_report.py 7 csv
"""

import sys
import json
import csv
from datetime import datetime
from services.llm_service import create_llm_manager


def export_detailed_report(days: int = 30, format: str = "json"):
    """
    Export a detailed LangSmith monitoring report.
    
    Args:
        days: Number of days to include in the report
        format: Export format ("json" or "csv")
    """
    print(f"📊 Exporting LangSmith report for last {days} days...")
    
    try:
        # Create LLM manager (connects to LangSmith)
        llm_manager = create_llm_manager()
        
        if not llm_manager.langsmith_monitor.is_enabled():
            print("❌ LangSmith monitoring not enabled. Check your API key.")
            return None
        
        # Get comprehensive monitoring data
        monitoring_data = llm_manager.get_monitoring_data(days)
        
        # Add export metadata
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "days_included": days,
                "application": "power-agent",
                "export_format": format
            },
            "langsmith_data": monitoring_data
        }
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"langsmith_export_{days}d_{timestamp}.{format}"
        filepath = f"reports/{filename}"
        
        # Export in requested format
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format.lower() == "csv":
            # Flatten data for CSV export
            csv_data = []
            
            # Usage metrics
            usage_metrics = monitoring_data.get("usage_metrics", {})
            if usage_metrics and "error" not in usage_metrics:
                csv_data.append({
                    "category": "usage",
                    "metric": "total_runs",
                    "value": usage_metrics.get("total_runs", 0),
                    "period_days": days
                })
                csv_data.append({
                    "category": "usage", 
                    "metric": "total_tokens",
                    "value": usage_metrics.get("total_tokens", 0),
                    "period_days": days
                })
                csv_data.append({
                    "category": "usage",
                    "metric": "estimated_cost_usd", 
                    "value": usage_metrics.get("estimated_cost_usd", 0),
                    "period_days": days
                })
                csv_data.append({
                    "category": "usage",
                    "metric": "error_rate_percent",
                    "value": usage_metrics.get("error_rate_percent", 0),
                    "period_days": days
                })
                csv_data.append({
                    "category": "usage",
                    "metric": "avg_latency_seconds",
                    "value": usage_metrics.get("avg_latency_seconds", 0),
                    "period_days": days
                })
            
            # Model breakdown
            model_breakdown = monitoring_data.get("model_breakdown", {})
            if model_breakdown and "error" not in model_breakdown:
                for model, stats in model_breakdown.items():
                    csv_data.append({
                        "category": "model",
                        "metric": f"{model}_runs", 
                        "value": stats.get("runs", 0),
                        "period_days": days
                    })
                    csv_data.append({
                        "category": "model",
                        "metric": f"{model}_cost_usd",
                        "value": stats.get("estimated_cost_usd", 0),
                        "period_days": days
                    })
                    csv_data.append({
                        "category": "model", 
                        "metric": f"{model}_total_tokens",
                        "value": stats.get("total_tokens", 0),
                        "period_days": days
                    })
            
            # Write CSV
            if csv_data:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=["category", "metric", "value", "period_days"])
                    writer.writeheader()
                    writer.writerows(csv_data)
            else:
                print("⚠️ No data available for CSV export")
                return None
        
        else:
            print(f"❌ Unsupported format: {format}. Use 'json' or 'csv'")
            return None
        
        print(f"✅ Report exported successfully!")
        print(f"📁 File: {filepath}")
        
        # Print summary
        usage = monitoring_data.get("usage_metrics", {})
        if usage and "error" not in usage:
            print(f"\n📊 Summary (Last {days} days):")
            print(f"   Total LLM calls: {usage.get('total_runs', 0):,}")
            print(f"   Total tokens: {usage.get('total_tokens', 0):,}")
            print(f"   Total cost: ${usage.get('estimated_cost_usd', 0):.4f}")
            print(f"   Error rate: {usage.get('error_rate_percent', 0):.1f}%")
            print(f"   Avg latency: {usage.get('avg_latency_seconds', 0):.2f}s")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return None


def export_evaluation_summary(days: int = 7):
    """Export evaluation results summary (if available)."""
    print(f"\n📋 Evaluation Summary (Last {days} days):")
    print("Note: Evaluation data export requires LangSmith Plus/Enterprise")
    print("For detailed evaluation exports, use the LangSmith web UI:")
    print("1. Go to https://smith.langchain.com")
    print("2. Navigate to Evaluations tab")
    print("3. Click Export button")


def main():
    """Main export function."""
    # Parse command line arguments
    days = 30
    format = "json"
    
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid days argument. Using default: 30")
    
    if len(sys.argv) > 2:
        format = sys.argv[2].lower()
        if format not in ["json", "csv"]:
            print("❌ Invalid format. Using default: json")
            format = "json"
    
    print("🚀 LangSmith Report Export Tool")
    print("=" * 40)
    
    # Export main report
    filepath = export_detailed_report(days, format)
    
    # Show evaluation summary
    export_evaluation_summary(days)
    
    if filepath:
        print(f"\n✨ Export complete! Check: {filepath}")
    else:
        print("\n❌ Export failed. Check LangSmith configuration.")


if __name__ == "__main__":
    main()