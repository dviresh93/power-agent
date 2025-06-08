#!/usr/bin/env python3
"""
Simple Results Viewer for DeepEval Reports
Displays evaluation results in a readable format
"""

import json
import sys
from datetime import datetime

def load_and_display_results(filename="memory_efficient_evaluation_report.json"):
    """Load and display evaluation results"""
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return
    
    # Header
    print("🎯 POWER OUTAGE AGENT EVALUATION RESULTS")
    print("=" * 60)
    
    # Summary
    summary = data.get("summary", {})
    print(f"\n📊 SUMMARY")
    print(f"   Timestamp: {data.get('timestamp', 'Unknown')}")
    print(f"   Total Tests: {summary.get('total_tests', 0)}")
    print(f"   Tests Passed: {summary.get('total_passed', 0)}")
    print(f"   Pass Rate: {summary.get('pass_rate', 0):.1%}")
    print(f"   Agent Reliability: {summary.get('agent_reliability', 'Unknown')}")
    
    # Decision Points Overview
    decision_points = data.get("decision_points", {})
    print(f"\n🎯 DECISION POINTS")
    
    # Weather Threshold
    weather = decision_points.get("weather_threshold", {})
    print(f"\n🌤️  Weather Threshold Decision:")
    print(f"    Tests Run: {weather.get('tests', 0)}")
    print(f"    Weather Accuracy: {weather.get('weather_accuracy', 0)}/{weather.get('tests', 0)}")
    print(f"    False Positive Detection: {weather.get('false_positive_detection', 0)}/{weather.get('tests', 0)}")
    
    # Temporal Analysis
    temporal = decision_points.get("temporal_analysis", {})
    print(f"\n⏰ Temporal Analysis Decision:")
    print(f"    Tests Run: {temporal.get('tests', 0)}")
    print(f"    Pattern Recognition: {temporal.get('pattern_recognition', 0)}/{temporal.get('tests', 0)}")
    
    # Chat Responses
    chat = decision_points.get("chat_responses", {})
    print(f"\n💬 Chat Response Decision:")
    print(f"    Tests Run: {chat.get('tests', 0)}")
    print(f"    Recommendation Quality: {chat.get('recommendation_quality', 0)}/{chat.get('tests', 0)}")
    
    # Detailed Results
    detailed = data.get("detailed_results", {})
    print(f"\n🔍 DETAILED RESULTS")
    
    # Weather Decisions
    weather_results = detailed.get("weather_decisions", [])
    if weather_results:
        print(f"\n🌤️  Weather Decision Details:")
        for i, result in enumerate(weather_results, 1):
            print(f"    Test {i}: {result.get('scenario', 'Unknown')}")
            if 'error' in result:
                print(f"    ❌ Error: {result['error']}")
            else:
                print(f"    ✅ Weather Score: {result.get('weather_threshold_score', 0):.2f}")
                print(f"    ✅ False Pos Score: {result.get('false_positive_detection_score', 0):.2f}")
                print(f"    📝 Response: {result.get('response_preview', 'No preview')[:100]}...")
    
    # Temporal Decisions
    temporal_results = detailed.get("temporal_decisions", [])
    if temporal_results:
        print(f"\n⏰ Temporal Decision Details:")
        for i, result in enumerate(temporal_results, 1):
            print(f"    Test {i}: {result.get('scenario', 'Unknown')}")
            if 'error' in result:
                print(f"    ❌ Error: {result['error']}")
            else:
                print(f"    ✅ Pattern Score: {result.get('temporal_pattern_score', 0):.2f}")
                print(f"    📝 Response: {result.get('response_preview', 'No preview')[:100]}...")
    
    # Chat Decisions
    chat_results = detailed.get("chat_decisions", [])
    if chat_results:
        print(f"\n💬 Chat Decision Details:")
        for i, result in enumerate(chat_results, 1):
            print(f"    Test {i}: {result.get('scenario', 'Unknown')}")
            print(f"    ❓ Question: {result.get('question', 'No question')}")
            if 'error' in result:
                print(f"    ❌ Error: {result['error']}")
            else:
                print(f"    ✅ Recommendation Score: {result.get('recommendation_score', 0):.2f}")
                print(f"    📝 Response: {result.get('response_preview', 'No preview')[:100]}...")
    
    # Recommendations
    if summary.get('pass_rate', 0) < 0.8:
        print(f"\n💡 RECOMMENDATIONS:")
        if weather.get('weather_accuracy', 0) == 0:
            print("   • Fix weather threshold decision logic")
        if temporal.get('tests', 0) == 0:
            print("   • Enable temporal analysis tests")
        if chat.get('tests', 0) == 0:
            print("   • Enable chat response tests")
        if any('error' in r for r in weather_results):
            print("   • Fix evaluation framework import/context issues")
    
    print(f"\n" + "=" * 60)

def main():
    """Main execution"""
    filename = sys.argv[1] if len(sys.argv) > 1 else "memory_efficient_evaluation_report.json"
    load_and_display_results(filename)

if __name__ == "__main__":
    main()