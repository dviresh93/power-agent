"""
LangSmith Online Evaluators for Power Outage Analysis

This module contains custom evaluators for LangSmith online evaluation that
assess the quality of LLM responses in the power outage analysis application.

These evaluators run automatically on every LLM trace to provide real-time
quality monitoring and feedback.

Evaluators:
- Report Quality: Evaluates technical report completeness and accuracy
- Validation Accuracy: Checks if outage validation decisions are well-reasoned
- Analysis Completeness: Ensures all required analysis components are present
- Response Formatting: Validates proper JSON structure and required fields
"""

import json
import re
from typing import Dict, Any, List


def evaluate_report_quality(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the quality of generated reports.
    
    Checks for:
    - Proper structure and formatting
    - Presence of key sections (summary, analysis, recommendations)
    - Technical accuracy indicators
    - Data-driven insights
    
    Returns:
        Dict with quality scores and feedback
    """
    try:
        output = run.get('outputs', {})
        
        # Get the actual response content
        response_content = ""
        if isinstance(output, dict):
            response_content = output.get('content', '') or str(output)
        else:
            response_content = str(output)
        
        response_lower = response_content.lower()
        
        # Quality indicators
        quality_score = 0
        max_score = 10
        feedback = []
        
        # Check for report structure (2 points)
        structure_keywords = ['summary', 'analysis', 'findings', 'conclusion', 'recommendation']
        structure_found = sum(1 for keyword in structure_keywords if keyword in response_lower)
        if structure_found >= 3:
            quality_score += 2
            feedback.append("Good report structure")
        elif structure_found >= 1:
            quality_score += 1
            feedback.append("Basic report structure")
        else:
            feedback.append("Missing report structure")
        
        # Check for data-driven insights (2 points)
        data_indicators = ['data shows', 'analysis reveals', 'statistics indicate', 'trends suggest', 'evidence']
        data_found = sum(1 for indicator in data_indicators if indicator in response_lower)
        if data_found >= 2:
            quality_score += 2
            feedback.append("Strong data-driven insights")
        elif data_found >= 1:
            quality_score += 1
            feedback.append("Some data-driven insights")
        else:
            feedback.append("Lacks data-driven insights")
        
        # Check for technical accuracy (2 points)
        technical_terms = ['outage', 'power', 'grid', 'electrical', 'utility', 'weather', 'correlation']
        technical_found = sum(1 for term in technical_terms if term in response_lower)
        if technical_found >= 4:
            quality_score += 2
            feedback.append("Good technical vocabulary")
        elif technical_found >= 2:
            quality_score += 1
            feedback.append("Basic technical vocabulary")
        else:
            feedback.append("Limited technical vocabulary")
        
        # Check for specific analysis elements (2 points)
        analysis_elements = ['false positive', 'real outage', 'validation', 'weather correlation', 'geographic']
        analysis_found = sum(1 for element in analysis_elements if element in response_lower)
        if analysis_found >= 3:
            quality_score += 2
            feedback.append("Comprehensive analysis")
        elif analysis_found >= 1:
            quality_score += 1
            feedback.append("Basic analysis")
        else:
            feedback.append("Limited analysis depth")
        
        # Check response length and detail (2 points)
        word_count = len(response_content.split())
        if word_count >= 200:
            quality_score += 2
            feedback.append("Detailed response")
        elif word_count >= 100:
            quality_score += 1
            feedback.append("Adequate detail")
        else:
            feedback.append("Response too brief")
        
        # Normalize score to 0-1 range
        normalized_score = quality_score / max_score
        
        return {
            "report_quality": normalized_score,
            "quality_score": quality_score,
            "max_score": max_score,
            "feedback": feedback,
            "word_count": word_count
        }
    
    except Exception as e:
        return {
            "report_quality": 0.0,
            "error": str(e),
            "feedback": ["Error evaluating report quality"]
        }


def evaluate_validation_accuracy(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the accuracy of outage validation decisions.
    
    Checks for:
    - Clear reasoning for validation decisions
    - Weather correlation analysis
    - Geographic considerations
    - Confidence indicators
    
    Returns:
        Dict with validation accuracy scores
    """
    try:
        output = run.get('outputs', {})
        inputs = run.get('inputs', {})
        
        # Get response content
        response_content = ""
        if isinstance(output, dict):
            response_content = output.get('content', '') or str(output)
        else:
            response_content = str(output)
        
        response_lower = response_content.lower()
        
        # Validation accuracy indicators
        accuracy_score = 0
        max_score = 8
        feedback = []
        
        # Check for clear decision reasoning (2 points)
        reasoning_indicators = ['because', 'due to', 'indicates', 'suggests', 'evidence shows', 'therefore']
        reasoning_found = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
        if reasoning_found >= 3:
            accuracy_score += 2
            feedback.append("Strong reasoning provided")
        elif reasoning_found >= 1:
            accuracy_score += 1
            feedback.append("Some reasoning provided")
        else:
            feedback.append("Lacks clear reasoning")
        
        # Check for weather correlation (2 points)
        weather_terms = ['weather', 'temperature', 'wind', 'storm', 'precipitation', 'climate']
        weather_found = sum(1 for term in weather_terms if term in response_lower)
        if weather_found >= 2:
            accuracy_score += 2
            feedback.append("Good weather analysis")
        elif weather_found >= 1:
            accuracy_score += 1
            feedback.append("Basic weather consideration")
        else:
            feedback.append("Missing weather analysis")
        
        # Check for geographic analysis (2 points)
        geo_terms = ['location', 'area', 'region', 'geographic', 'spatial', 'distance', 'proximity']
        geo_found = sum(1 for term in geo_terms if term in response_lower)
        if geo_found >= 2:
            accuracy_score += 2
            feedback.append("Good geographic analysis")
        elif geo_found >= 1:
            accuracy_score += 1
            feedback.append("Basic geographic consideration")
        else:
            feedback.append("Missing geographic analysis")
        
        # Check for confidence indicators (2 points)
        confidence_terms = ['likely', 'probable', 'confident', 'certain', 'uncertain', 'possible']
        confidence_found = sum(1 for term in confidence_terms if term in response_lower)
        if confidence_found >= 2:
            accuracy_score += 2
            feedback.append("Clear confidence indicators")
        elif confidence_found >= 1:
            accuracy_score += 1
            feedback.append("Some confidence indicators")
        else:
            feedback.append("Missing confidence indicators")
        
        # Normalize score
        normalized_score = accuracy_score / max_score
        
        return {
            "validation_accuracy": normalized_score,
            "accuracy_score": accuracy_score,
            "max_score": max_score,
            "feedback": feedback
        }
    
    except Exception as e:
        return {
            "validation_accuracy": 0.0,
            "error": str(e),
            "feedback": ["Error evaluating validation accuracy"]
        }


def evaluate_response_formatting(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate proper formatting of LLM responses.
    
    Checks for:
    - Valid JSON structure (where applicable)
    - Proper markdown formatting
    - Required fields present
    - No malformed content
    
    Returns:
        Dict with formatting scores
    """
    try:
        output = run.get('outputs', {})
        
        formatting_score = 0
        max_score = 6
        feedback = []
        
        # Get response content
        response_content = ""
        if isinstance(output, dict):
            response_content = output.get('content', '') or str(output)
        else:
            response_content = str(output)
        
        # Check if response is supposed to be JSON
        if response_content.strip().startswith('{') or response_content.strip().startswith('['):
            try:
                json.loads(response_content)
                formatting_score += 2
                feedback.append("Valid JSON structure")
            except json.JSONDecodeError:
                feedback.append("Invalid JSON structure")
        else:
            # Check for markdown formatting
            markdown_indicators = ['#', '##', '###', '*', '**', '-', '```']
            markdown_found = sum(1 for indicator in markdown_indicators if indicator in response_content)
            if markdown_found >= 3:
                formatting_score += 2
                feedback.append("Good markdown formatting")
            elif markdown_found >= 1:
                formatting_score += 1
                feedback.append("Basic formatting")
            else:
                formatting_score += 1
                feedback.append("Plain text formatting")
        
        # Check for complete sentences (2 points)
        sentences = response_content.split('.')
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
        if complete_sentences >= 5:
            formatting_score += 2
            feedback.append("Well-structured sentences")
        elif complete_sentences >= 2:
            formatting_score += 1
            feedback.append("Adequate sentence structure")
        else:
            feedback.append("Poor sentence structure")
        
        # Check for no obvious errors (2 points)
        error_indicators = ['error', 'failed', 'null', 'undefined', 'exception']
        errors_found = sum(1 for error in error_indicators if error in response_content.lower())
        if errors_found == 0:
            formatting_score += 2
            feedback.append("No obvious errors")
        elif errors_found <= 1:
            formatting_score += 1
            feedback.append("Minor errors present")
        else:
            feedback.append("Multiple errors present")
        
        # Normalize score
        normalized_score = formatting_score / max_score
        
        return {
            "formatting_quality": normalized_score,
            "formatting_score": formatting_score,
            "max_score": max_score,
            "feedback": feedback
        }
    
    except Exception as e:
        return {
            "formatting_quality": 0.0,
            "error": str(e),
            "feedback": ["Error evaluating formatting"]
        }


def evaluate_analysis_completeness(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate completeness of analysis responses.
    
    Checks for:
    - All required analysis components
    - Comprehensive coverage
    - Actionable insights
    - Proper conclusions
    
    Returns:
        Dict with completeness scores
    """
    try:
        output = run.get('outputs', {})
        inputs = run.get('inputs', {})
        
        # Get response content
        response_content = ""
        if isinstance(output, dict):
            response_content = output.get('content', '') or str(output)
        else:
            response_content = str(output)
        
        response_lower = response_content.lower()
        
        completeness_score = 0
        max_score = 8
        feedback = []
        
        # Check for comprehensive analysis (2 points)
        analysis_components = ['summary', 'details', 'patterns', 'trends', 'insights']
        components_found = sum(1 for comp in analysis_components if comp in response_lower)
        if components_found >= 4:
            completeness_score += 2
            feedback.append("Comprehensive analysis")
        elif components_found >= 2:
            completeness_score += 1
            feedback.append("Basic analysis coverage")
        else:
            feedback.append("Limited analysis coverage")
        
        # Check for actionable insights (2 points)
        actionable_terms = ['recommend', 'suggest', 'should', 'could', 'action', 'next steps']
        actionable_found = sum(1 for term in actionable_terms if term in response_lower)
        if actionable_found >= 3:
            completeness_score += 2
            feedback.append("Strong actionable insights")
        elif actionable_found >= 1:
            completeness_score += 1
            feedback.append("Some actionable insights")
        else:
            feedback.append("Lacks actionable insights")
        
        # Check for proper conclusions (2 points)
        conclusion_terms = ['conclusion', 'summary', 'in summary', 'overall', 'finally']
        conclusion_found = sum(1 for term in conclusion_terms if term in response_lower)
        if conclusion_found >= 2:
            completeness_score += 2
            feedback.append("Clear conclusions")
        elif conclusion_found >= 1:
            completeness_score += 1
            feedback.append("Basic conclusions")
        else:
            feedback.append("Missing conclusions")
        
        # Check for quantitative elements (2 points)
        quantitative_terms = ['number', 'count', 'percentage', 'rate', 'total', 'average', 'statistics']
        quant_found = sum(1 for term in quantitative_terms if term in response_lower)
        if quant_found >= 3:
            completeness_score += 2
            feedback.append("Good quantitative analysis")
        elif quant_found >= 1:
            completeness_score += 1
            feedback.append("Some quantitative elements")
        else:
            feedback.append("Lacks quantitative analysis")
        
        # Normalize score
        normalized_score = completeness_score / max_score
        
        return {
            "analysis_completeness": normalized_score,
            "completeness_score": completeness_score,
            "max_score": max_score,
            "feedback": feedback
        }
    
    except Exception as e:
        return {
            "analysis_completeness": 0.0,
            "error": str(e),
            "feedback": ["Error evaluating analysis completeness"]
        }


# Test functions for local evaluation
def test_evaluators():
    """Test all evaluators with sample data."""
    sample_run = {
        "inputs": {"question": "Analyze power outage patterns"},
        "outputs": {
            "content": """
            # Power Outage Analysis Summary
            
            Based on the data analysis, several key findings emerge:
            
            ## Weather Correlation Analysis
            The data shows a strong correlation between severe weather events and power outages. 
            Storm patterns indicate that 78% of outages occur during high wind conditions.
            
            ## Geographic Distribution
            Outages are concentrated in the northeastern region, with higher frequency in rural areas.
            This suggests infrastructure vulnerabilities in remote locations.
            
            ## Validation Results
            After careful analysis, 15 outages were classified as real outages due to weather correlation,
            while 8 were identified as false positives because they lacked supporting weather data.
            
            ## Recommendations
            1. Focus infrastructure improvements on high-risk areas
            2. Implement predictive monitoring during weather events
            3. Review validation criteria for better accuracy
            
            The analysis indicates high confidence in these findings based on comprehensive data review.
            """
        }
    }
    
    print("Testing evaluators...")
    
    # Test each evaluator
    evaluators = [
        ("Report Quality", evaluate_report_quality),
        ("Validation Accuracy", evaluate_validation_accuracy),
        ("Response Formatting", evaluate_response_formatting),
        ("Analysis Completeness", evaluate_analysis_completeness)
    ]
    
    for name, evaluator in evaluators:
        try:
            result = evaluator(sample_run)
            print(f"\n{name}:")
            print(f"  Score: {result.get('score', 'N/A')}")
            print(f"  Feedback: {result.get('feedback', [])}")
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")


if __name__ == "__main__":
    test_evaluators()