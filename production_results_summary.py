#!/usr/bin/env python3
"""
Production DeepEval Results Summary
Clean summary of production evaluation results
"""

import json
import sys

def main():
    """Display production evaluation summary"""
    filename = sys.argv[1] if len(sys.argv) > 1 else "production_evaluation_report.json"
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return
    
    print("🎯 PRODUCTION DEEPEVAL RESULTS SUMMARY")
    print("=" * 60)
    
    # Overall Summary
    summary = data.get("summary", {})
    print(f"\n📊 OVERALL PERFORMANCE")
    print(f"   Tests Run: {summary.get('total_tests', 0)}")
    print(f"   Tests Passed: {summary.get('total_passed', 0)}")
    print(f"   Pass Rate: {summary.get('pass_rate', 0):.1%}")
    
    reliability = summary.get('agent_reliability', 'Unknown')
    emoji = "🟢" if reliability == "High" else "🟡" if reliability == "Medium" else "🔴"
    print(f"   Agent Reliability: {emoji} {reliability}")
    
    # Weather Validation Results
    weather = data.get("weather_validation", {})
    print(f"\n🌤️  WEATHER THRESHOLD DECISION TESTING")
    print(f"   Total Scenarios: {weather.get('total_tests', 0)}")
    print(f"   ✅ Accuracy Tests Passed: {weather.get('accuracy_tests_passed', 0)}/{weather.get('total_tests', 0)} ({weather.get('accuracy_rate', 0):.1%})")
    print(f"   📝 Reasoning Tests Passed: {weather.get('reasoning_tests_passed', 0)}/{weather.get('total_tests', 0)} ({weather.get('reasoning_rate', 0):.1%})")
    
    # Weather Test Details
    weather_details = data.get("detailed_results", {}).get("weather_validation", [])
    if weather_details:
        print(f"\n   📋 Weather Test Scenarios:")
        for i, result in enumerate(weather_details, 1):
            name = result.get("scenario_name", "Unknown")
            expected = result.get("expected_classification", "?")
            accuracy = "✅" if result.get("accuracy_passed", False) else "❌"
            reasoning = "✅" if result.get("reasoning_passed", False) else "❌"
            print(f"      {i}. {name}")
            print(f"         Expected: {expected} | Accuracy: {accuracy} | Reasoning: {reasoning}")
    
    # Chat Response Results
    chat = data.get("chat_responses", {})
    print(f"\n💬 CHAT RESPONSE TESTING")
    print(f"   Total Questions: {chat.get('total_tests', 0)}")
    print(f"   ✅ Quality Tests Passed: {chat.get('quality_tests_passed', 0)}/{chat.get('total_tests', 0)} ({chat.get('quality_rate', 0):.1%})")
    print(f"   📊 Average Topic Coverage: {chat.get('average_topic_coverage', 0):.1%}")
    
    # Chat Test Details
    chat_details = data.get("detailed_results", {}).get("chat_responses", [])
    if chat_details:
        print(f"\n   📋 Chat Test Scenarios:")
        for i, result in enumerate(chat_details, 1):
            name = result.get("scenario_name", "Unknown")
            quality = "✅" if result.get("quality_passed", False) else "❌"
            coverage = result.get("topic_coverage_rate", 0)
            print(f"      {i}. {name}")
            print(f"         Quality: {quality} | Topic Coverage: {coverage:.1%}")
    
    # Key Findings
    print(f"\n🔍 KEY FINDINGS")
    
    if weather.get('accuracy_rate', 0) > 0.8:
        print(f"   ✅ Weather threshold decisions are highly accurate ({weather.get('accuracy_rate', 0):.1%})")
    else:
        print(f"   ⚠️  Weather threshold accuracy needs improvement ({weather.get('accuracy_rate', 0):.1%})")
    
    if weather.get('reasoning_rate', 0) > 0.8:
        print(f"   ✅ Agent provides excellent reasoning quality ({weather.get('reasoning_rate', 0):.1%})")
    else:
        print(f"   ⚠️  Reasoning quality needs improvement ({weather.get('reasoning_rate', 0):.1%})")
    
    if chat.get('quality_rate', 0) > 0.6:
        print(f"   ✅ Chat responses are helpful and accurate")
    else:
        print(f"   ⚠️  Chat response quality needs improvement ({chat.get('quality_rate', 0):.1%})")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    if summary.get('pass_rate', 0) > 0.8:
        print(f"   🎉 Agent performance is excellent - ready for production")
    elif summary.get('pass_rate', 0) > 0.6:
        print(f"   🔧 Agent performance is good with room for improvement")
        if chat.get('quality_rate', 0) < 0.6:
            print(f"   📝 Focus on improving chat response helpfulness")
    else:
        print(f"   🛠️  Agent needs significant improvements before production")
        if weather.get('accuracy_rate', 0) < 0.8:
            print(f"   🌤️  Review weather threshold logic")
        if chat.get('quality_rate', 0) < 0.6:
            print(f"   💬 Enhance chat response quality and relevance")
    
    print(f"\n📁 Full results available in: {filename}")
    print("=" * 60)

if __name__ == "__main__":
    main()