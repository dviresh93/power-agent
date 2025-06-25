#!/usr/bin/env python3
"""
Enhanced Results Viewer for DeepEval Reports
Multiple viewing options with detailed analysis
"""

import json
import sys
import argparse
from datetime import datetime

class ResultsViewer:
    """Enhanced results viewer with multiple display options"""
    
    def __init__(self, filename="memory_efficient_evaluation_report.json"):
        self.filename = filename
        self.data = self.load_data()
    
    def load_data(self):
        """Load evaluation results"""
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ File not found: {self.filename}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            sys.exit(1)
    
    def view_summary(self):
        """Display summary view"""
        print("📊 EVALUATION SUMMARY")
        print("=" * 50)
        
        summary = self.data.get("summary", {})
        print(f"🕐 Timestamp: {self.data.get('timestamp', 'Unknown')}")
        print(f"🧪 Total Tests: {summary.get('total_tests', 0)}")
        print(f"✅ Tests Passed: {summary.get('total_passed', 0)}")
        print(f"📈 Pass Rate: {summary.get('pass_rate', 0):.1%}")
        
        reliability = summary.get('agent_reliability', 'Unknown')
        emoji = "🟢" if reliability == "High" else "🟡" if reliability == "Medium" else "🔴"
        print(f"{emoji} Agent Reliability: {reliability}")
        
        return summary.get('pass_rate', 0) >= 0.8
    
    def view_decision_points(self):
        """Display decision points breakdown"""
        print("\n🎯 DECISION POINTS BREAKDOWN")
        print("=" * 50)
        
        decision_points = self.data.get("decision_points", {})
        
        # Weather Threshold Decision
        weather = decision_points.get("weather_threshold", {})
        weather_rate = weather.get('weather_accuracy', 0) / max(weather.get('tests', 1), 1)
        false_pos_rate = weather.get('false_positive_detection', 0) / max(weather.get('tests', 1), 1)
        
        print(f"\n🌤️  Weather Threshold Decision:")
        print(f"   📊 Tests: {weather.get('tests', 0)}")
        print(f"   🎯 Weather Accuracy: {weather.get('weather_accuracy', 0)}/{weather.get('tests', 0)} ({weather_rate:.1%})")
        print(f"   🔍 False Positive Detection: {weather.get('false_positive_detection', 0)}/{weather.get('tests', 0)} ({false_pos_rate:.1%})")
        
        # Temporal Analysis Decision
        temporal = decision_points.get("temporal_analysis", {})
        temporal_rate = temporal.get('pattern_recognition', 0) / max(temporal.get('tests', 1), 1) if temporal.get('tests', 0) > 0 else 0
        
        print(f"\n⏰ Temporal Analysis Decision:")
        print(f"   📊 Tests: {temporal.get('tests', 0)}")
        print(f"   🔍 Pattern Recognition: {temporal.get('pattern_recognition', 0)}/{temporal.get('tests', 0)} ({temporal_rate:.1%})")
        
        # Chat Response Decision
        chat = decision_points.get("chat_responses", {})
        chat_rate = chat.get('recommendation_quality', 0) / max(chat.get('tests', 1), 1) if chat.get('tests', 0) > 0 else 0
        
        print(f"\n💬 Chat Response Decision:")
        print(f"   📊 Tests: {chat.get('tests', 0)}")
        print(f"   💡 Recommendation Quality: {chat.get('recommendation_quality', 0)}/{chat.get('tests', 0)} ({chat_rate:.1%})")
    
    def view_detailed_results(self, show_responses=False):
        """Display detailed test results"""
        print("\n🔍 DETAILED TEST RESULTS")
        print("=" * 50)
        
        detailed = self.data.get("detailed_results", {})
        
        # Weather Decision Results
        weather_results = detailed.get("weather_decisions", [])
        if weather_results:
            print(f"\n🌤️  Weather Threshold Decision Tests:")
            for i, result in enumerate(weather_results, 1):
                scenario = result.get('scenario', 'Unknown')
                print(f"\n   Test {i}: {scenario}")
                
                if 'error' in result:
                    print(f"   ❌ Error: {result['error']}")
                else:
                    weather_score = result.get('weather_threshold_score', 0)
                    false_pos_score = result.get('false_positive_detection_score', 0)
                    weather_passed = result.get('weather_passed', False)
                    false_pos_passed = result.get('false_pos_passed', False)
                    
                    weather_icon = "✅" if weather_passed else "❌"
                    false_pos_icon = "✅" if false_pos_passed else "❌"
                    
                    print(f"   {weather_icon} Weather Threshold Score: {weather_score:.2f}")
                    print(f"   {false_pos_icon} False Positive Detection Score: {false_pos_score:.2f}")
                    
                    if show_responses:
                        response = result.get('response_preview', 'No response available')
                        print(f"   📝 Agent Response:")
                        print(f"      {response[:200]}{'...' if len(response) > 200 else ''}")
        
        # Temporal Analysis Results
        temporal_results = detailed.get("temporal_decisions", [])
        if temporal_results:
            print(f"\n⏰ Temporal Analysis Decision Tests:")
            for i, result in enumerate(temporal_results, 1):
                scenario = result.get('scenario', 'Unknown')
                print(f"\n   Test {i}: {scenario}")
                
                if 'error' in result:
                    print(f"   ❌ Error: {result['error']}")
                else:
                    temporal_score = result.get('temporal_pattern_score', 0)
                    temporal_passed = result.get('temporal_passed', False)
                    temporal_icon = "✅" if temporal_passed else "❌"
                    
                    print(f"   {temporal_icon} Temporal Pattern Score: {temporal_score:.2f}")
                    
                    if show_responses:
                        response = result.get('response_preview', 'No response available')
                        print(f"   📝 Agent Response:")
                        print(f"      {response[:200]}{'...' if len(response) > 200 else ''}")
        
        # Chat Response Results
        chat_results = detailed.get("chat_decisions", [])
        if chat_results:
            print(f"\n💬 Chat Response Decision Tests:")
            for i, result in enumerate(chat_results, 1):
                scenario = result.get('scenario', 'Unknown')
                question = result.get('question', 'No question')
                print(f"\n   Test {i}: {scenario}")
                print(f"   ❓ Question: {question}")
                
                if 'error' in result:
                    print(f"   ❌ Error: {result['error']}")
                else:
                    recommendation_score = result.get('recommendation_score', 0)
                    recommendation_passed = result.get('recommendation_passed', False)
                    recommendation_icon = "✅" if recommendation_passed else "❌"
                    
                    print(f"   {recommendation_icon} Recommendation Score: {recommendation_score:.2f}")
                    
                    if show_responses:
                        response = result.get('response_preview', 'No response available')
                        print(f"   📝 Agent Response:")
                        print(f"      {response[:200]}{'...' if len(response) > 200 else ''}")
    
    def view_recommendations(self):
        """Display recommendations based on results"""
        print("\n💡 RECOMMENDATIONS")
        print("=" * 50)
        
        summary = self.data.get("summary", {})
        decision_points = self.data.get("decision_points", {})
        detailed = self.data.get("detailed_results", {})
        
        recommendations = []
        
        # Overall performance
        if summary.get('pass_rate', 0) < 0.8:
            recommendations.append("🔧 Overall performance needs improvement")
        else:
            recommendations.append("✅ Agent performance is excellent")
        
        # Weather threshold analysis
        weather = decision_points.get("weather_threshold", {})
        if weather.get('tests', 0) > 0:
            if weather.get('weather_accuracy', 0) < weather.get('tests', 0):
                recommendations.append("🌤️  Improve weather threshold decision accuracy")
            if weather.get('false_positive_detection', 0) < weather.get('tests', 0):
                recommendations.append("🔍 Enhance false positive detection quality")
        
        # Temporal analysis
        temporal = decision_points.get("temporal_analysis", {})
        if temporal.get('tests', 0) == 0:
            recommendations.append("⏰ Enable temporal analysis testing")
        elif temporal.get('pattern_recognition', 0) < temporal.get('tests', 0):
            recommendations.append("⏰ Improve temporal pattern recognition")
        
        # Chat responses
        chat = decision_points.get("chat_responses", {})
        if chat.get('tests', 0) == 0:
            recommendations.append("💬 Enable chat response testing")
        elif chat.get('recommendation_quality', 0) < chat.get('tests', 0):
            recommendations.append("💬 Enhance chat response quality")
        
        # Error handling
        all_results = (detailed.get("weather_decisions", []) + 
                      detailed.get("temporal_decisions", []) + 
                      detailed.get("chat_decisions", []))
        
        if any('error' in result for result in all_results):
            recommendations.append("🛠️  Fix evaluation framework import/execution issues")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    def view_full_report(self, show_responses=False):
        """Display full comprehensive report"""
        print("🎯 POWER OUTAGE AGENT - COMPREHENSIVE EVALUATION REPORT")
        print("=" * 80)
        
        success = self.view_summary()
        self.view_decision_points()
        self.view_detailed_results(show_responses)
        self.view_recommendations()
        
        print("\n" + "=" * 80)
        
        return success

def main():
    """Main execution with command line arguments"""
    parser = argparse.ArgumentParser(description="View DeepEval results for Power Outage Agent")
    parser.add_argument("filename", nargs="?", default="memory_efficient_evaluation_report.json",
                       help="JSON file to view (default: memory_efficient_evaluation_report.json)")
    parser.add_argument("--summary", "-s", action="store_true", help="Show summary only")
    parser.add_argument("--decision-points", "-d", action="store_true", help="Show decision points only")
    parser.add_argument("--detailed", "-t", action="store_true", help="Show detailed results only")
    parser.add_argument("--recommendations", "-r", action="store_true", help="Show recommendations only")
    parser.add_argument("--responses", action="store_true", help="Include agent responses in detailed view")
    parser.add_argument("--full", "-f", action="store_true", help="Show full comprehensive report")
    
    args = parser.parse_args()
    
    viewer = ResultsViewer(args.filename)
    
    if args.summary:
        viewer.view_summary()
    elif args.decision_points:
        viewer.view_decision_points()
    elif args.detailed:
        viewer.view_detailed_results(args.responses)
    elif args.recommendations:
        viewer.view_recommendations()
    elif args.full:
        viewer.view_full_report(args.responses)
    else:
        # Default: show full report without responses
        viewer.view_full_report(False)

if __name__ == "__main__":
    main()