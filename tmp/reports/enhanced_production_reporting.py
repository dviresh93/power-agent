#!/usr/bin/env python3
"""
Enhanced Production Reporting with LangSmith Integration

Combines existing DeepEval results with LangSmith monitoring data
to provide comprehensive production insights including:
- Performance metrics from DeepEval
- Cost analysis from LangSmith
- Token usage patterns
- Latency and throughput analysis
- Model performance comparison
"""

import json
import sys
import os
from datetime import datetime, timedelta
from services.llm_service import create_llm_manager
from typing import Dict, Any, Optional

def print_section(title: str, char: str = "=", width: int = 60):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")

def load_deepeval_results(filename: str) -> Optional[Dict[str, Any]]:
    """Load DeepEval results from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ DeepEval results file not found: {filename}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error parsing DeepEval results: {filename}")
        return None

def get_langsmith_data(days: int = 7) -> Optional[Dict[str, Any]]:
    """Get LangSmith monitoring data."""
    try:
        llm_manager = create_llm_manager()
        if not llm_manager.langsmith_monitor.is_enabled():
            print("⚠️  LangSmith monitoring not enabled")
            return None
        
        return llm_manager.get_monitoring_data(days)
    except Exception as e:
        print(f"❌ Error getting LangSmith data: {e}")
        return None

def display_deepeval_summary(data: Dict[str, Any]):
    """Display DeepEval performance summary."""
    print_section("🎯 DEEPEVAL PERFORMANCE SUMMARY")
    
    summary = data.get("summary", {})
    print(f"📊 Overall Performance:")
    print(f"   Tests Run: {summary.get('total_tests', 0)}")
    print(f"   Tests Passed: {summary.get('total_passed', 0)}")
    print(f"   Pass Rate: {summary.get('pass_rate', 0):.1%}")
    
    reliability = summary.get('agent_reliability', 'Unknown')
    emoji = "🟢" if reliability == "High" else "🟡" if reliability == "Medium" else "🔴"
    print(f"   Agent Reliability: {emoji} {reliability}")
    
    # Weather validation
    weather = data.get("weather_validation", {})
    if weather:
        print(f"\n🌤️  Weather Validation:")
        print(f"   Accuracy Rate: {weather.get('accuracy_rate', 0):.1%}")
        print(f"   Reasoning Quality: {weather.get('reasoning_rate', 0):.1%}")
    
    # Chat responses
    chat = data.get("chat_responses", {})
    if chat:
        print(f"\n💬 Chat Response Quality:")
        print(f"   Quality Rate: {chat.get('quality_rate', 0):.1%}")
        print(f"   Topic Coverage: {chat.get('average_topic_coverage', 0):.1%}")

def display_langsmith_metrics(data: Dict[str, Any], days: int):
    """Display LangSmith monitoring metrics."""
    print_section("📈 LANGSMITH MONITORING METRICS")
    
    usage = data.get("usage_metrics", {})
    if "error" in usage:
        print(f"❌ Error getting usage data: {usage['error']}")
        return
    
    print(f"📊 Usage Metrics (Last {days} days):")
    print(f"   Total LLM Calls: {usage.get('total_runs', 0):,}")
    print(f"   Successful Calls: {usage.get('successful_runs', 0):,}")
    print(f"   Failed Calls: {usage.get('failed_runs', 0):,}")
    print(f"   Error Rate: {usage.get('error_rate_percent', 0):.2f}%")
    print(f"   Average Daily Calls: {usage.get('runs_per_day', 0):.1f}")
    
    print(f"\n💰 Cost Analysis:")
    print(f"   Total Cost: ${usage.get('estimated_cost_usd', 0):.4f}")
    print(f"   Daily Average: ${(usage.get('estimated_cost_usd', 0) / days):.4f}")
    if usage.get('total_runs', 0) > 0:
        cost_per_call = usage.get('estimated_cost_usd', 0) / usage.get('total_runs', 1)
        print(f"   Cost per Call: ${cost_per_call:.6f}")
    
    print(f"\n🔤 Token Usage:")
    print(f"   Input Tokens: {usage.get('total_input_tokens', 0):,}")
    print(f"   Output Tokens: {usage.get('total_output_tokens', 0):,}")
    print(f"   Total Tokens: {usage.get('total_tokens', 0):,}")
    if usage.get('total_runs', 0) > 0:
        avg_tokens_per_call = usage.get('total_tokens', 0) / usage.get('total_runs', 1)
        print(f"   Avg Tokens per Call: {avg_tokens_per_call:.0f}")
    
    print(f"\n⚡ Performance Metrics:")
    print(f"   Average Latency: {usage.get('avg_latency_seconds', 0):.3f}s")
    print(f"   95th Percentile Latency: {usage.get('p95_latency_seconds', 0):.3f}s")
    
    # Model breakdown
    models = data.get("model_breakdown", {})
    if models and not isinstance(models, dict) or "error" not in models:
        print(f"\n🤖 Model Performance Breakdown:")
        for model, stats in models.items():
            if isinstance(stats, dict):
                print(f"   {model}:")
                print(f"     Calls: {stats.get('runs', 0):,}")
                print(f"     Tokens: {stats.get('total_tokens', 0):,}")
                print(f"     Cost: ${stats.get('estimated_cost_usd', 0):.4f}")
                print(f"     Avg Latency: {stats.get('avg_latency_seconds', 0):.3f}s")
                print(f"     Cost per 1K tokens: ${stats.get('cost_per_1k_tokens', 0):.4f}")

def calculate_efficiency_metrics(deepeval_data: Dict[str, Any], langsmith_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate efficiency metrics combining both data sources."""
    usage = langsmith_data.get("usage_metrics", {})
    summary = deepeval_data.get("summary", {})
    
    metrics = {}
    
    # Cost efficiency
    total_cost = usage.get('estimated_cost_usd', 0)
    pass_rate = summary.get('pass_rate', 0)
    
    if total_cost > 0 and pass_rate > 0:
        metrics['cost_per_successful_test'] = total_cost / (summary.get('total_passed', 1))
        metrics['cost_effectiveness_score'] = pass_rate / total_cost if total_cost > 0 else 0
    
    # Performance efficiency
    avg_latency = usage.get('avg_latency_seconds', 0)
    if avg_latency > 0:
        metrics['performance_efficiency'] = pass_rate / avg_latency
    
    # Token efficiency
    total_tokens = usage.get('total_tokens', 0)
    total_tests = summary.get('total_tests', 0)
    if total_tokens > 0 and total_tests > 0:
        metrics['tokens_per_test'] = total_tokens / total_tests
        metrics['successful_tokens_ratio'] = (total_tokens * pass_rate) / total_tokens if total_tokens > 0 else 0
    
    return metrics

def display_efficiency_analysis(efficiency_metrics: Dict[str, Any]):
    """Display efficiency analysis."""
    print_section("⚡ EFFICIENCY ANALYSIS")
    
    print("🎯 Cost Efficiency:")
    if 'cost_per_successful_test' in efficiency_metrics:
        print(f"   Cost per Successful Test: ${efficiency_metrics['cost_per_successful_test']:.6f}")
    if 'cost_effectiveness_score' in efficiency_metrics:
        score = efficiency_metrics['cost_effectiveness_score']
        rating = "Excellent" if score > 100 else "Good" if score > 50 else "Needs Improvement"
        print(f"   Cost Effectiveness Score: {score:.2f} ({rating})")
    
    print("\n⚡ Performance Efficiency:")
    if 'performance_efficiency' in efficiency_metrics:
        perf = efficiency_metrics['performance_efficiency']
        rating = "Excellent" if perf > 0.5 else "Good" if perf > 0.2 else "Needs Improvement"
        print(f"   Performance Score: {perf:.3f} ({rating})")
    
    print("\n🔤 Token Efficiency:")
    if 'tokens_per_test' in efficiency_metrics:
        print(f"   Tokens per Test: {efficiency_metrics['tokens_per_test']:.0f}")
    if 'successful_tokens_ratio' in efficiency_metrics:
        ratio = efficiency_metrics['successful_tokens_ratio']
        print(f"   Successful Token Utilization: {ratio:.1%}")

def generate_recommendations(deepeval_data: Dict[str, Any], langsmith_data: Dict[str, Any], efficiency_metrics: Dict[str, Any]):
    """Generate actionable recommendations."""
    print_section("💡 ACTIONABLE RECOMMENDATIONS")
    
    summary = deepeval_data.get("summary", {})
    usage = langsmith_data.get("usage_metrics", {})
    models = langsmith_data.get("model_breakdown", {})
    
    recommendations = []
    
    # Performance recommendations
    pass_rate = summary.get('pass_rate', 0)
    if pass_rate < 0.8:
        recommendations.append("🎯 Improve test pass rate - consider fine-tuning prompts or model selection")
    
    # Cost recommendations
    error_rate = usage.get('error_rate_percent', 0)
    if error_rate > 5:
        recommendations.append(f"💰 Reduce error rate ({error_rate:.1f}%) to minimize wasted API costs")
    
    # Latency recommendations
    avg_latency = usage.get('avg_latency_seconds', 0)
    if avg_latency > 3:
        recommendations.append(f"⚡ Optimize response time (currently {avg_latency:.1f}s) - consider smaller models or prompt optimization")
    
    # Model recommendations
    if isinstance(models, dict) and len(models) > 1:
        # Find most cost-effective model
        best_model = None
        best_efficiency = 0
        
        for model, stats in models.items():
            if isinstance(stats, dict):
                cost_per_token = stats.get('cost_per_1k_tokens', float('inf'))
                if cost_per_token > 0:
                    efficiency = 1 / cost_per_token
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_model = model
        
        if best_model:
            recommendations.append(f"🤖 Consider using {best_model} more frequently - appears most cost-effective")
    
    # Token efficiency recommendations
    if 'tokens_per_test' in efficiency_metrics:
        tokens_per_test = efficiency_metrics['tokens_per_test']
        if tokens_per_test > 2000:
            recommendations.append("🔤 Optimize prompt length - high token usage per test")
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("   🎉 System is performing optimally!")
    
    # Cost projection
    daily_cost = usage.get('estimated_cost_usd', 0) / max(usage.get('period_days', 1), 1)
    if daily_cost > 0:
        print(f"\n📊 Cost Projections:")
        print(f"   Weekly: ${daily_cost * 7:.2f}")
        print(f"   Monthly: ${daily_cost * 30:.2f}")
        print(f"   Yearly: ${daily_cost * 365:.2f}")

def export_combined_report(deepeval_data: Dict[str, Any], langsmith_data: Dict[str, Any], efficiency_metrics: Dict[str, Any]) -> str:
    """Export combined report to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/combined_production_report_{timestamp}.json"
    
    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    
    combined_report = {
        "report_timestamp": datetime.now().isoformat(),
        "report_type": "combined_production_analysis",
        "deepeval_results": deepeval_data,
        "langsmith_metrics": langsmith_data,
        "efficiency_analysis": efficiency_metrics,
        "summary": {
            "total_tests": deepeval_data.get("summary", {}).get("total_tests", 0),
            "pass_rate": deepeval_data.get("summary", {}).get("pass_rate", 0),
            "total_llm_calls": langsmith_data.get("usage_metrics", {}).get("total_runs", 0),
            "total_cost": langsmith_data.get("usage_metrics", {}).get("estimated_cost_usd", 0),
            "error_rate": langsmith_data.get("usage_metrics", {}).get("error_rate_percent", 0),
            "avg_latency": langsmith_data.get("usage_metrics", {}).get("avg_latency_seconds", 0)
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(combined_report, f, indent=2)
    
    return filename

def main():
    """Main reporting function."""
    print("📊 ENHANCED PRODUCTION REPORTING WITH LANGSMITH")
    print("Combining DeepEval results with LangSmith monitoring data")
    
    # Parse arguments
    deepeval_file = sys.argv[1] if len(sys.argv) > 1 else "production_evaluation_report.json"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    
    # Load data
    print(f"\n🔍 Loading data...")
    print(f"   DeepEval results: {deepeval_file}")
    print(f"   LangSmith period: {days} days")
    
    deepeval_data = load_deepeval_results(deepeval_file)
    if not deepeval_data:
        return 1
    
    langsmith_data = get_langsmith_data(days)
    if not langsmith_data:
        print("⚠️  Continuing with DeepEval data only...")
        langsmith_data = {"usage_metrics": {}, "model_breakdown": {}}
    
    # Display summaries
    display_deepeval_summary(deepeval_data)
    
    if langsmith_data and "usage_metrics" in langsmith_data:
        display_langsmith_metrics(langsmith_data, days)
        
        # Calculate and display efficiency metrics
        efficiency_metrics = calculate_efficiency_metrics(deepeval_data, langsmith_data)
        if efficiency_metrics:
            display_efficiency_analysis(efficiency_metrics)
        
        # Generate recommendations
        generate_recommendations(deepeval_data, langsmith_data, efficiency_metrics)
        
        # Export combined report
        try:
            report_file = export_combined_report(deepeval_data, langsmith_data, efficiency_metrics)
            print_section("📁 REPORT EXPORTED")
            print(f"Combined report saved to: {report_file}")
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
    
    print_section("✅ ANALYSIS COMPLETE")
    print("For detailed LangSmith analytics, visit: https://smith.langchain.com")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())