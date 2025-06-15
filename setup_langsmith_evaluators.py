#!/usr/bin/env python3
"""
LangSmith Online Evaluators Setup Guide

This script helps you set up online evaluators in LangSmith for automatic
quality assessment of your power outage analysis application.

Usage:
    python setup_langsmith_evaluators.py

This will:
1. Test evaluators locally to ensure they work
2. Provide instructions for setting up automation rules in LangSmith
3. Give you the evaluator code to paste into the LangSmith UI
"""

import json
import sys
from services.langsmith_evaluators import (
    evaluate_report_quality,
    evaluate_validation_accuracy,
    evaluate_response_formatting,
    evaluate_analysis_completeness
)


def test_local_evaluators():
    """Test all evaluators locally before deployment."""
    print("🧪 Testing LangSmith Evaluators Locally")
    print("=" * 50)
    
    # Sample test data
    sample_runs = [
        {
            "name": "Good Report",
            "run": {
                "inputs": {"question": "Analyze power outage patterns"},
                "outputs": {
                    "content": """
                    # Power Outage Analysis Summary
                    
                    Based on comprehensive data analysis, several key findings emerge:
                    
                    ## Weather Correlation Analysis
                    The data shows strong correlation between severe weather events and outages. 
                    Statistical analysis indicates 78% of outages occur during high wind conditions.
                    
                    ## Geographic Distribution
                    Outages concentrate in northeastern regions, with rural areas showing higher frequency.
                    This suggests infrastructure vulnerabilities in remote locations.
                    
                    ## Validation Results
                    After thorough analysis, 15 outages classified as real due to weather correlation,
                    while 8 identified as false positives lacking supporting weather evidence.
                    
                    ## Recommendations
                    1. Focus infrastructure improvements on high-risk areas
                    2. Implement predictive monitoring during weather events
                    3. Review validation criteria for improved accuracy
                    
                    Analysis indicates high confidence in findings based on comprehensive review.
                    """
                }
            }
        },
        {
            "name": "Poor Report",
            "run": {
                "inputs": {"question": "What about outages?"},
                "outputs": {
                    "content": "There are some outages. Weather might be involved. Not sure."
                }
            }
        }
    ]
    
    evaluators = [
        ("Report Quality", evaluate_report_quality),
        ("Validation Accuracy", evaluate_validation_accuracy),
        ("Response Formatting", evaluate_response_formatting),
        ("Analysis Completeness", evaluate_analysis_completeness)
    ]
    
    for sample in sample_runs:
        print(f"\n📊 Testing with: {sample['name']}")
        print("-" * 30)
        
        for eval_name, evaluator in evaluators:
            try:
                result = evaluator(sample['run'])
                # Get the primary score key
                score_keys = [k for k in result.keys() if k not in ['feedback', 'error', 'max_score']]
                primary_score = result.get(score_keys[0], 0.0) if score_keys else 0.0
                
                print(f"  {eval_name}: {primary_score:.2f}")
                if 'feedback' in result:
                    print(f"    Feedback: {', '.join(result['feedback'][:2])}")
                
            except Exception as e:
                print(f"  {eval_name}: ERROR - {e}")
    
    print("\n✅ Local testing complete!")


def print_evaluator_code():
    """Print the evaluator code for pasting into LangSmith UI."""
    print("\n🔧 Evaluator Code for LangSmith UI")
    print("=" * 50)
    
    evaluators = {
        "report_quality": {
            "name": "Report Quality Evaluator",
            "description": "Evaluates technical report quality, structure, and completeness",
            "code": '''
import json
import re

def perform_eval(run):
    """Evaluate report quality and completeness."""
    try:
        output = run.get('outputs', {})
        
        # Get response content
        response_content = ""
        if isinstance(output, dict):
            response_content = output.get('content', '') or str(output)
        else:
            response_content = str(output)
        
        response_lower = response_content.lower()
        
        # Quality scoring
        quality_score = 0
        max_score = 10
        feedback = []
        
        # Check report structure (2 points)
        structure_keywords = ['summary', 'analysis', 'findings', 'conclusion', 'recommendation']
        structure_found = sum(1 for keyword in structure_keywords if keyword in response_lower)
        if structure_found >= 3:
            quality_score += 2
            feedback.append("Good structure")
        elif structure_found >= 1:
            quality_score += 1
            feedback.append("Basic structure")
        
        # Check data-driven insights (2 points)
        data_indicators = ['data shows', 'analysis reveals', 'statistics', 'evidence']
        data_found = sum(1 for indicator in data_indicators if indicator in response_lower)
        if data_found >= 2:
            quality_score += 2
            feedback.append("Data-driven")
        elif data_found >= 1:
            quality_score += 1
        
        # Check technical vocabulary (2 points)
        technical_terms = ['outage', 'power', 'weather', 'correlation', 'validation']
        technical_found = sum(1 for term in technical_terms if term in response_lower)
        if technical_found >= 3:
            quality_score += 2
            feedback.append("Technical accuracy")
        elif technical_found >= 1:
            quality_score += 1
        
        # Check analysis depth (2 points)
        analysis_elements = ['false positive', 'real outage', 'weather correlation']
        analysis_found = sum(1 for element in analysis_elements if element in response_lower)
        if analysis_found >= 2:
            quality_score += 2
            feedback.append("Deep analysis")
        elif analysis_found >= 1:
            quality_score += 1
        
        # Check response length (2 points)
        word_count = len(response_content.split())
        if word_count >= 200:
            quality_score += 2
            feedback.append("Detailed")
        elif word_count >= 100:
            quality_score += 1
        
        return {
            "report_quality": quality_score / max_score,
            "feedback": feedback,
            "word_count": word_count
        }
    
    except Exception as e:
        return {"report_quality": 0.0, "error": str(e)}
'''
        },
        
        "validation_accuracy": {
            "name": "Validation Accuracy Evaluator", 
            "description": "Evaluates reasoning quality for outage validation decisions",
            "code": '''
def perform_eval(run):
    """Evaluate validation decision accuracy."""
    try:
        output = run.get('outputs', {})
        
        response_content = ""
        if isinstance(output, dict):
            response_content = output.get('content', '') or str(output)
        else:
            response_content = str(output)
        
        response_lower = response_content.lower()
        
        accuracy_score = 0
        max_score = 6
        feedback = []
        
        # Check reasoning (2 points)
        reasoning_indicators = ['because', 'due to', 'indicates', 'therefore']
        reasoning_found = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
        if reasoning_found >= 2:
            accuracy_score += 2
            feedback.append("Good reasoning")
        elif reasoning_found >= 1:
            accuracy_score += 1
        
        # Check weather analysis (2 points)
        weather_terms = ['weather', 'storm', 'wind', 'temperature']
        weather_found = sum(1 for term in weather_terms if term in response_lower)
        if weather_found >= 2:
            accuracy_score += 2
            feedback.append("Weather analysis")
        elif weather_found >= 1:
            accuracy_score += 1
        
        # Check confidence indicators (2 points)
        confidence_terms = ['likely', 'confident', 'certain', 'probable']
        confidence_found = sum(1 for term in confidence_terms if term in response_lower)
        if confidence_found >= 1:
            accuracy_score += 2
            feedback.append("Clear confidence")
        
        return {
            "validation_accuracy": accuracy_score / max_score,
            "feedback": feedback
        }
    
    except Exception as e:
        return {"validation_accuracy": 0.0, "error": str(e)}
'''
        }
    }
    
    for eval_id, eval_info in evaluators.items():
        print(f"\n📋 {eval_info['name']}")
        print(f"Description: {eval_info['description']}")
        print("Code to paste in LangSmith:")
        print("-" * 40)
        print(eval_info['code'])
        print("-" * 40)


def print_setup_instructions():
    """Print step-by-step setup instructions."""
    print("\n📋 LangSmith Setup Instructions")
    print("=" * 50)
    
    instructions = """
    1. 🌐 Go to https://smith.langchain.com
    2. 📁 Open your 'power-agent' project
    3. ⚙️  Click "Add Rules" button in the top right
    4. ➕ For each evaluator:
       
       a) Report Quality Evaluator:
          - Name: "Report Quality"
          - Filter: Leave empty (evaluates all runs)
          - Sampling Rate: 100% (or lower for cost savings)
          - Select "Custom Code Evaluator"
          - Paste the Report Quality code from above
          - Test code with sample data
          - Save evaluator
       
       b) Validation Accuracy Evaluator:
          - Name: "Validation Accuracy" 
          - Filter: Leave empty
          - Sampling Rate: 100%
          - Select "Custom Code Evaluator"
          - Paste the Validation Accuracy code from above
          - Test and save
    
    5. ✅ Once saved, evaluators will automatically run on new traces
    6. 📊 View results in the LangSmith dashboard under "Evaluations"
    7. 📈 Monitor quality trends over time
    
    💡 Tips:
    - Start with 10% sampling rate to test, then increase
    - Add filters if you want to evaluate specific trace types only
    - Monitor evaluation costs (each evaluation uses LLM credits)
    - Check evaluation results regularly for quality insights
    """
    
    print(instructions)


def main():
    """Main setup function."""
    print("🚀 LangSmith Online Evaluators Setup")
    print("=" * 50)
    
    # Test evaluators locally first
    print("\nStep 1: Testing evaluators locally...")
    test_local_evaluators()
    
    # Print evaluator code
    print_evaluator_code()
    
    # Print setup instructions
    print_setup_instructions()
    
    print("\n✨ Setup complete!")
    print("Next: Copy the evaluator code above and paste it into LangSmith UI")
    print("Visit: https://smith.langchain.com")


if __name__ == "__main__":
    main()