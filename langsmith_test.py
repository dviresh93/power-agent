#!/usr/bin/env python3
"""
LangSmith Integration Test Script

This script demonstrates and tests the LangSmith integration.
Run this to:
1. Test LangSmith connection
2. Make sample LLM calls with tracing
3. View monitoring data
4. Generate reports

Usage:
    python langsmith_test.py
"""

import os
import sys
import json
from datetime import datetime
from services.llm_service import create_llm_manager
from langchain_core.messages import HumanMessage

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def check_environment():
    """Check if LangSmith environment is properly configured."""
    print_section("Environment Check")
    
    api_key = os.getenv("LANGSMITH_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT", "power-agent")
    tracing = os.getenv("LANGCHAIN_TRACING_V2")
    
    print(f"LANGSMITH_API_KEY: {'✓ Set' if api_key else '✗ Not set'}")
    print(f"LANGCHAIN_PROJECT: {project}")
    print(f"LANGCHAIN_TRACING_V2: {'✓ Enabled' if tracing == 'true' else '✗ Not enabled'}")
    
    if not api_key:
        print("\n⚠️  LangSmith API key not found!")
        print("Please set LANGSMITH_API_KEY in your environment or .env file")
        print("Get your API key from: https://smith.langchain.com")
        return False
    
    return True

def test_llm_manager():
    """Test LLM manager with LangSmith integration."""
    print_section("Testing LLM Manager")
    
    try:
        # Create LLM manager
        llm_manager = create_llm_manager()
        
        # Get provider info
        provider_info = llm_manager.get_provider_info()
        print(f"Provider: {provider_info['provider']}")
        print(f"Model: {provider_info['model']}")
        print(f"LangSmith enabled: {llm_manager.langsmith_monitor.is_enabled()}")
        
        return llm_manager
    except Exception as e:
        print(f"❌ Error creating LLM manager: {e}")
        return None

def make_sample_calls(llm_manager):
    """Make sample LLM calls to generate tracing data."""
    print_section("Making Sample LLM Calls")
    
    sample_queries = [
        "What is the weather like today?",
        "Explain renewable energy in one sentence.",
        "What are the benefits of solar power?",
        "How do wind turbines work?",
        "What is carbon footprint?"
    ]
    
    responses = []
    
    for i, query in enumerate(sample_queries, 1):
        try:
            print(f"\n{i}. Query: {query}")
            
            # Create message
            message = HumanMessage(content=query)
            
            # Make LLM call (this will be traced by LangSmith)
            response = llm_manager.invoke([message])
            
            print(f"   Response: {response.content[:100]}...")
            responses.append({
                "query": query,
                "response": response.content,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return responses

def view_monitoring_data(llm_manager):
    """View monitoring data from LangSmith."""
    print_section("LangSmith Monitoring Data")
    
    try:
        # Get monitoring data
        monitoring_data = llm_manager.get_monitoring_data(days=1)
        
        if "error" in monitoring_data:
            print(f"❌ Error getting monitoring data: {monitoring_data['error']}")
            return
        
        # Display usage metrics
        usage = monitoring_data.get("usage_metrics", {})
        print("\n📊 Usage Metrics (Last 24 hours):")
        print(f"   Total runs: {usage.get('total_runs', 0)}")
        print(f"   Successful runs: {usage.get('successful_runs', 0)}")
        print(f"   Failed runs: {usage.get('failed_runs', 0)}")
        print(f"   Error rate: {usage.get('error_rate_percent', 0)}%")
        print(f"   Total tokens: {usage.get('total_tokens', 0):,}")
        print(f"   Estimated cost: ${usage.get('estimated_cost_usd', 0)}")
        print(f"   Avg latency: {usage.get('avg_latency_seconds', 0)}s")
        
        # Display model breakdown
        models = monitoring_data.get("model_breakdown", {})
        print("\n🤖 Model Breakdown:")
        for model, stats in models.items():
            print(f"   {model}:")
            print(f"     Runs: {stats.get('runs', 0)}")
            print(f"     Tokens: {stats.get('total_tokens', 0):,}")
            print(f"     Cost: ${stats.get('estimated_cost_usd', 0)}")
            print(f"     Avg latency: {stats.get('avg_latency_seconds', 0)}s")
        
        return monitoring_data
        
    except Exception as e:
        print(f"❌ Error viewing monitoring data: {e}")
        return None

def generate_report(llm_manager):
    """Generate and export monitoring report."""
    print_section("Generating Monitoring Report")
    
    try:
        # Export report
        report_path = llm_manager.export_monitoring_report(days=7)
        print(f"✓ Report exported to: {report_path}")
        
        # Read and display summary
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            print(f"\n📋 Report Summary:")
            print(f"   Export time: {report_data.get('export_timestamp')}")
            print(f"   Period: {report_data.get('period_days')} days")
            
            usage = report_data.get('usage_metrics', {})
            if usage and 'total_runs' in usage:
                print(f"   Total runs: {usage['total_runs']}")
                print(f"   Total cost: ${usage.get('estimated_cost_usd', 0)}")
        
        return report_path
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return None

def show_setup_instructions(llm_manager):
    """Show LangSmith setup instructions."""
    print_section("LangSmith Setup Instructions")
    
    instructions = llm_manager.get_langsmith_setup_instructions()
    print(instructions)

def main():
    """Main test function."""
    print("🧪 LangSmith Integration Test")
    print("This script will test the LangSmith integration with your LLM service.")
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please configure LangSmith first.")
        return 1
    
    # Test LLM manager
    llm_manager = test_llm_manager()
    if not llm_manager:
        return 1
    
    # Show setup instructions if not enabled
    if not llm_manager.langsmith_monitor.is_enabled():
        show_setup_instructions(llm_manager)
        print("\n⚠️  LangSmith not enabled. Please follow setup instructions above.")
        return 1
    
    # Make sample calls
    responses = make_sample_calls(llm_manager)
    
    # Wait a moment for data to be processed
    print("\n⏳ Waiting for LangSmith to process data...")
    import time
    time.sleep(5)
    
    # View monitoring data
    monitoring_data = view_monitoring_data(llm_manager)
    
    # Generate report
    report_path = generate_report(llm_manager)
    
    print_section("Test Complete")
    print("✅ LangSmith integration test completed!")
    print("\nNext steps:")
    print("1. Visit https://smith.langchain.com to view detailed analytics")
    print("2. Check the generated report for comprehensive metrics")
    print("3. Integrate monitoring into your production workflows")
    
    if report_path:
        print(f"4. Review the exported report: {report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())