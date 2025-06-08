"""
Comprehensive DeepEvals for Power Outage Analysis Agent
Evaluates every decision point with advanced metrics and test scenarios
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
import asyncio
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# DeepEval imports
from deepeval import evaluate, assert_test
from deepeval.metrics import (
    HallucinationMetric,
    FactualConsistencyMetric, 
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    BiasMetric,
    ToxicityMetric,
    SummarizationMetric,
    GEval,
    BaseMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

# Import agent components
from main import (
    validate_outage_report,
    analyze_outage_reports_dataset,
    chat_about_validation_results,
    WeatherService,
    LLMManager,
    PromptManager
)

# ==================== CUSTOM DEEPEVAL METRICS ====================

class WeatherConsistencyMetric(BaseMetric):
    """Custom metric to evaluate weather-based decision consistency"""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.name = "Weather Consistency"
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Measure if weather analysis aligns with outage classification"""
        try:
            weather_data = json.loads(test_case.context) if test_case.context else {}
            response = test_case.actual_output.upper()
            
            # Define severe weather thresholds based on agent prompts
            severe_conditions = [
                weather_data.get('wind_speed', 0) > 40,  # > 25 mph = 40 km/h
                weather_data.get('wind_gusts', 0) > 56,  # > 35 mph = 56 km/h  
                weather_data.get('precipitation', 0) > 12.7,  # > 0.5 inches/hour
                weather_data.get('temperature', 20) < -12 or weather_data.get('temperature', 20) > 35,
                weather_data.get('snowfall', 0) > 5  # > 2 inches = 5 cm
            ]
            
            has_severe_weather = any(severe_conditions)
            classified_as_real = "REAL OUTAGE" in response
            
            # Perfect consistency: severe weather = real outage, mild = false positive
            if has_severe_weather and classified_as_real:
                self.score = 1.0
            elif not has_severe_weather and not classified_as_real:
                self.score = 1.0
            else:
                # Borderline cases get partial credit
                severity_count = sum(severe_conditions)
                if severity_count == 1:  # Borderline case
                    self.score = 0.7
                else:
                    self.score = 0.0
            
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Weather consistency measurement error: {e}")
            self.score = 0.0
            self.success = False
            return self.score
    
    def is_successful(self) -> bool:
        return self.success

class OutageClassificationAccuracyMetric(BaseMetric):
    """Custom metric for outage classification accuracy"""
    
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.name = "Classification Accuracy"
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Measure classification accuracy against expected result"""
        try:
            expected = test_case.expected_output.upper()
            actual = test_case.actual_output.upper()
            
            # Extract classification from responses
            expected_class = None
            actual_class = None
            
            if "REAL OUTAGE" in expected:
                expected_class = "REAL"
            elif "FALSE POSITIVE" in expected:
                expected_class = "FALSE"
            
            if "REAL OUTAGE" in actual:
                actual_class = "REAL"
            elif "FALSE POSITIVE" in actual:
                actual_class = "FALSE"
            
            if expected_class and actual_class:
                self.score = 1.0 if expected_class == actual_class else 0.0
            else:
                self.score = 0.0
            
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Classification accuracy measurement error: {e}")
            self.score = 0.0
            self.success = False
            return self.score
    
    def is_successful(self) -> bool:
        return self.success

class ReasoningQualityMetric(BaseMetric):
    """Custom metric for reasoning quality in responses"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.name = "Reasoning Quality"
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Measure quality of reasoning provided"""
        try:
            response = test_case.actual_output.lower()
            
            # Quality indicators
            quality_indicators = [
                any(word in response for word in ['wind', 'temperature', 'precipitation', 'weather']),
                any(word in response for word in ['because', 'due to', 'caused by', 'since']),
                any(word in response for word in ['mph', 'km/h', 'degrees', 'inches', 'mm']),
                len(response.split()) > 20,  # Sufficient detail
                any(word in response for word in ['severe', 'mild', 'extreme', 'moderate', 'heavy', 'light'])
            ]
            
            self.score = sum(quality_indicators) / len(quality_indicators)
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Reasoning quality measurement error: {e}")
            self.score = 0.0
            self.success = False
            return self.score
    
    def is_successful(self) -> bool:
        return self.success

# ==================== TEST DATA GENERATION ====================

@dataclass
class TestScenario:
    """Data class for test scenarios"""
    name: str
    outage_report: Dict
    weather_data: Dict
    expected_classification: str
    expected_reasoning_points: List[str]
    scenario_type: str  # "clear_false_positive", "clear_real_outage", "borderline"

class TestDataGenerator:
    """Generate comprehensive test scenarios"""
    
    def __init__(self):
        self.scenarios = []
        self._generate_scenarios()
    
    def _generate_scenarios(self):
        """Generate diverse test scenarios covering edge cases"""
        
        # Clear False Positives - Mild Weather
        self.scenarios.extend([
            TestScenario(
                name="Perfect Weather Single Customer",
                outage_report={
                    'datetime': '2022-01-01 00:00:00',
                    'latitude': 40.198069,
                    'longitude': -88.262575,
                    'customers': 1
                },
                weather_data={
                    'temperature': 15.0,
                    'precipitation': 0.0,
                    'wind_speed': 8.0,
                    'wind_gusts': 12.0,
                    'snowfall': 0.0
                },
                expected_classification="FALSE POSITIVE",
                expected_reasoning_points=['mild weather', 'low wind', 'no precipitation'],
                scenario_type="clear_false_positive"
            ),
            
            TestScenario(
                name="Sensor Malfunction Pattern",
                outage_report={
                    'datetime': '2022-01-01 02:30:00',
                    'latitude': 39.456789,
                    'longitude': -87.654321,
                    'customers': 3
                },
                weather_data={
                    'temperature': 18.5,
                    'precipitation': 0.1,
                    'wind_speed': 12.0,
                    'wind_gusts': 16.0,
                    'snowfall': 0.0
                },
                expected_classification="FALSE POSITIVE",
                expected_reasoning_points=['normal conditions', 'light wind', 'minimal precipitation'],
                scenario_type="clear_false_positive"
            )
        ])
        
        # Clear Real Outages - Severe Weather
        self.scenarios.extend([
            TestScenario(
                name="High Wind Storm",
                outage_report={
                    'datetime': '2022-01-01 01:45:00',
                    'latitude': 38.822124,
                    'longitude': -89.791293,
                    'customers': 43
                },
                weather_data={
                    'temperature': 5.0,
                    'precipitation': 2.5,
                    'wind_speed': 48.0,  # Strong wind > 40 km/h
                    'wind_gusts': 67.0,  # Strong gusts > 56 km/h
                    'snowfall': 0.5
                },
                expected_classification="REAL OUTAGE",
                expected_reasoning_points=['high wind', 'strong gusts', 'weather caused'],
                scenario_type="clear_real_outage"
            ),
            
            TestScenario(
                name="Ice Storm Conditions",
                outage_report={
                    'datetime': '2022-01-01 04:20:00',
                    'latitude': 40.123456,
                    'longitude': -88.987654,
                    'customers': 28
                },
                weather_data={
                    'temperature': -8.0,  # Below -12°C threshold when combined with precipitation
                    'precipitation': 5.2,  # Heavy precipitation
                    'wind_speed': 25.0,
                    'wind_gusts': 35.0,
                    'snowfall': 8.0  # Heavy snow > 5 cm
                },
                expected_classification="REAL OUTAGE",
                expected_reasoning_points=['freezing conditions', 'heavy snow', 'ice accumulation'],
                scenario_type="clear_real_outage"
            )
        ])
        
        # Borderline Cases
        self.scenarios.extend([
            TestScenario(
                name="Moderate Wind Edge Case",
                outage_report={
                    'datetime': '2022-01-01 03:15:00',
                    'latitude': 38.687404,
                    'longitude': -89.609473,
                    'customers': 15
                },
                weather_data={
                    'temperature': 2.3,
                    'precipitation': 0.8,
                    'wind_speed': 38.0,  # Just under severe threshold
                    'wind_gusts': 54.0,  # Just under severe threshold
                    'snowfall': 3.5
                },
                expected_classification="BORDERLINE",
                expected_reasoning_points=['moderate conditions', 'borderline wind speeds'],
                scenario_type="borderline"
            ),
            
            TestScenario(
                name="Temperature Extreme Low Customer Count",
                outage_report={
                    'datetime': '2022-01-01 05:00:00',
                    'latitude': 39.555555,
                    'longitude': -88.333333,
                    'customers': 2
                },
                weather_data={
                    'temperature': -15.0,  # Extreme cold
                    'precipitation': 0.0,
                    'wind_speed': 15.0,
                    'wind_gusts': 20.0,
                    'snowfall': 0.0
                },
                expected_classification="BORDERLINE",
                expected_reasoning_points=['extreme cold', 'low customer count', 'minimal other factors'],
                scenario_type="borderline"
            )
        ])
    
    def get_scenarios(self, scenario_type: str = None) -> List[TestScenario]:
        """Get scenarios filtered by type"""
        if scenario_type:
            return [s for s in self.scenarios if s.scenario_type == scenario_type]
        return self.scenarios

# ==================== TEST CASE CLASSES ====================

class OutageValidationTestCase(LLMTestCase):
    """Enhanced test case for outage validation"""
    
    def __init__(self, scenario: TestScenario):
        self.scenario = scenario
        
        super().__init__(
            input=json.dumps(scenario.outage_report),
            actual_output="",
            expected_output=scenario.expected_classification,
            context=json.dumps(scenario.weather_data)
        )
    
    def execute(self):
        """Execute the validation test"""
        try:
            result = validate_outage_report.invoke({
                "outage_report": self.scenario.outage_report,
                "weather_data": self.scenario.weather_data
            })
            self.actual_output = result
            return result
        except Exception as e:
            logger.error(f"Test execution error: {e}")
            self.actual_output = f"ERROR: {str(e)}"
            return self.actual_output

class DatasetAnalysisTestCase(LLMTestCase):
    """Test case for dataset analysis capabilities"""
    
    def __init__(self, dataset_summary: Dict, sample_reports: List[Dict], expected_insights: List[str]):
        self.dataset_summary = dataset_summary
        self.sample_reports = sample_reports
        self.expected_insights = expected_insights
        
        super().__init__(
            input=json.dumps({"summary": dataset_summary, "samples": sample_reports}),
            actual_output="",
            expected_output="Comprehensive dataset analysis with patterns and insights",
            context=json.dumps({"expected_insights": expected_insights})
        )
    
    def execute(self):
        """Execute dataset analysis"""
        try:
            result = analyze_outage_reports_dataset.invoke({
                "dataset_summary": self.dataset_summary,
                "sample_reports": self.sample_reports
            })
            self.actual_output = result
            return result
        except Exception as e:
            logger.error(f"Dataset analysis error: {e}")
            self.actual_output = f"ERROR: {str(e)}"
            return self.actual_output

class ChatResponseTestCase(LLMTestCase):
    """Test case for chat interface responses"""
    
    def __init__(self, question: str, context: Dict, expected_topics: List[str]):
        self.question = question
        self.context = context
        self.expected_topics = expected_topics
        
        super().__init__(
            input=question,
            actual_output="",
            expected_output=f"Response covering: {', '.join(expected_topics)}",
            context=json.dumps(context)
        )
    
    def execute(self):
        """Execute chat response test"""
        try:
            result = chat_about_validation_results.invoke({
                "question": self.question,
                "context": self.context
            })
            self.actual_output = result
            return result
        except Exception as e:
            logger.error(f"Chat response error: {e}")
            self.actual_output = f"ERROR: {str(e)}"
            return self.actual_output

# ==================== TEST SUITE RUNNERS ====================

class OutageValidationTestSuite:
    """Comprehensive test suite for outage validation"""
    
    def __init__(self):
        self.data_generator = TestDataGenerator()
        self.results = []
    
    def run_basic_accuracy_tests(self):
        """Test basic classification accuracy"""
        logger.info("Running basic accuracy tests...")
        
        # Test clear cases first
        clear_scenarios = (
            self.data_generator.get_scenarios("clear_false_positive") +
            self.data_generator.get_scenarios("clear_real_outage")
        )
        
        results = []
        for scenario in clear_scenarios:
            test_case = OutageValidationTestCase(scenario)
            test_case.execute()
            
            # Apply metrics
            accuracy_metric = OutageClassificationAccuracyMetric(threshold=0.9)
            weather_metric = WeatherConsistencyMetric(threshold=0.8)
            reasoning_metric = ReasoningQualityMetric(threshold=0.7)
            hallucination_metric = HallucinationMetric(threshold=0.3)
            
            accuracy_metric.measure(test_case)
            weather_metric.measure(test_case)
            reasoning_metric.measure(test_case)
            hallucination_metric.measure(test_case)
            
            results.append({
                "scenario_name": scenario.name,
                "scenario_type": scenario.scenario_type,
                "expected": scenario.expected_classification,
                "actual_response": test_case.actual_output,
                "accuracy_score": accuracy_metric.score,
                "weather_consistency": weather_metric.score,
                "reasoning_quality": reasoning_metric.score,
                "hallucination_score": hallucination_metric.score,
                "accuracy_passed": accuracy_metric.is_successful(),
                "weather_passed": weather_metric.is_successful(),
                "reasoning_passed": reasoning_metric.is_successful(),
                "hallucination_passed": hallucination_metric.is_successful()
            })
        
        return results
    
    def run_edge_case_tests(self):
        """Test edge cases and borderline scenarios"""
        logger.info("Running edge case tests...")
        
        borderline_scenarios = self.data_generator.get_scenarios("borderline")
        results = []
        
        for scenario in borderline_scenarios:
            test_case = OutageValidationTestCase(scenario)
            test_case.execute()
            
            # For borderline cases, focus on reasoning quality and consistency
            reasoning_metric = ReasoningQualityMetric(threshold=0.6)  # Lower threshold
            weather_metric = WeatherConsistencyMetric(threshold=0.6)
            factual_metric = FactualConsistencyMetric(threshold=0.7)
            
            reasoning_metric.measure(test_case)
            weather_metric.measure(test_case)
            factual_metric.measure(test_case)
            
            # For borderline cases, either classification could be correct
            classification_reasonable = any(keyword in test_case.actual_output.upper() 
                                          for keyword in ["REAL OUTAGE", "FALSE POSITIVE"])
            
            results.append({
                "scenario_name": scenario.name,
                "scenario_type": scenario.scenario_type,
                "actual_response": test_case.actual_output,
                "classification_reasonable": classification_reasonable,
                "reasoning_quality": reasoning_metric.score,
                "weather_consistency": weather_metric.score,
                "factual_consistency": factual_metric.score,
                "reasoning_passed": reasoning_metric.is_successful(),
                "weather_passed": weather_metric.is_successful(),
                "factual_passed": factual_metric.is_successful()
            })
        
        return results
    
    def run_stress_tests(self):
        """Test system behavior under stress conditions"""
        logger.info("Running stress tests...")
        
        # Test with malformed data
        stress_scenarios = [
            {
                "name": "Missing Weather Data",
                "outage_report": {'datetime': '2022-01-01 12:00:00', 'latitude': 40.0, 'longitude': -88.0, 'customers': 10},
                "weather_data": {},  # Empty weather data
            },
            {
                "name": "Extreme Values",
                "outage_report": {'datetime': '2022-01-01 12:00:00', 'latitude': 40.0, 'longitude': -88.0, 'customers': 9999},
                "weather_data": {'temperature': -50, 'wind_speed': 200, 'precipitation': 100},
            },
            {
                "name": "Invalid Coordinates",
                "outage_report": {'datetime': '2022-01-01 12:00:00', 'latitude': 999, 'longitude': -999, 'customers': 5},
                "weather_data": {'temperature': 20, 'wind_speed': 10, 'precipitation': 0},
            }
        ]
        
        results = []
        for stress_test in stress_scenarios:
            try:
                test_case = OutageValidationTestCase(
                    TestScenario(
                        name=stress_test["name"],
                        outage_report=stress_test["outage_report"],
                        weather_data=stress_test["weather_data"],
                        expected_classification="ERROR_HANDLING",
                        expected_reasoning_points=[],
                        scenario_type="stress_test"
                    )
                )
                test_case.execute()
                
                # Check if system handles errors gracefully
                error_handled = not test_case.actual_output.startswith("ERROR:")
                response_length = len(test_case.actual_output)
                
                results.append({
                    "stress_test_name": stress_test["name"],
                    "response": test_case.actual_output[:200] + "..." if response_length > 200 else test_case.actual_output,
                    "error_handled_gracefully": error_handled,
                    "response_length": response_length,
                    "provides_reasoning": len(test_case.actual_output.split()) > 10
                })
                
            except Exception as e:
                results.append({
                    "stress_test_name": stress_test["name"],
                    "response": f"Exception: {str(e)}",
                    "error_handled_gracefully": False,
                    "response_length": 0,
                    "provides_reasoning": False
                })
        
        return results

class DatasetAnalysisTestSuite:
    """Test suite for dataset analysis capabilities"""
    
    def run_analysis_tests(self):
        """Test dataset analysis functionality"""
        logger.info("Running dataset analysis tests...")
        
        # Create mock dataset summaries
        test_datasets = [
            {
                "name": "High False Positive Dataset",
                "summary": {
                    "total_records": 100,
                    "date_range": {"start": "2022-01-01", "end": "2022-01-02"},
                    "customer_impact": {"total_affected": 1000, "avg_per_outage": 10, "max_single_outage": 50},
                    "temporal_patterns": {"peak_hour": 2, "hourly_distribution": {"0": 20, "1": 30, "2": 50}}
                },
                "sample_reports": [
                    {"datetime": "2022-01-01 00:00:00", "customers": 1, "latitude": 40.0, "longitude": -88.0},
                    {"datetime": "2022-01-01 02:00:00", "customers": 50, "latitude": 39.0, "longitude": -87.0}
                ],
                "expected_insights": ["peak activity", "suspicious patterns", "validation needed"]
            }
        ]
        
        results = []
        for dataset in test_datasets:
            test_case = DatasetAnalysisTestCase(
                dataset["summary"],
                dataset["sample_reports"],
                dataset["expected_insights"]
            )
            test_case.execute()
            
            # Evaluate analysis quality
            relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
            summarization_metric = SummarizationMetric(threshold=0.7)
            
            relevancy_metric.measure(test_case)
            summarization_metric.measure(test_case)
            
            # Check for key insights
            response_lower = test_case.actual_output.lower()
            insights_found = sum(1 for insight in dataset["expected_insights"] 
                               if insight.lower() in response_lower)
            
            results.append({
                "dataset_name": dataset["name"],
                "insights_found": insights_found,
                "total_expected_insights": len(dataset["expected_insights"]),
                "relevancy_score": relevancy_metric.score,
                "summarization_score": summarization_metric.score,
                "response_length": len(test_case.actual_output),
                "relevancy_passed": relevancy_metric.is_successful(),
                "summarization_passed": summarization_metric.is_successful()
            })
        
        return results

class ChatInterfaceTestSuite:
    """Test suite for chat interface"""
    
    def run_chat_tests(self):
        """Test chat interface responses"""
        logger.info("Running chat interface tests...")
        
        # Mock validation context
        validation_context = {
            "validation_results": {
                "total_reports": 100,
                "real_outages": [
                    {"datetime": "2022-01-01 01:45:00", "customers": 43, "weather_data": {"wind_speed": 48.0}}
                ],
                "false_positives": [
                    {"datetime": "2022-01-01 00:00:00", "customers": 1, "weather_data": {"wind_speed": 8.0}}
                ],
                "statistics": {
                    "false_positive_rate": 40.0,
                    "total_customers_actually_affected": 300
                }
            }
        }
        
        test_questions = [
            {
                "question": "Why were these reports classified as false positives?",
                "expected_topics": ["weather conditions", "mild weather", "false positive criteria"],
                "test_type": "explanation"
            },
            {
                "question": "What weather conditions caused the real outages?",
                "expected_topics": ["severe weather", "wind speed", "precipitation"],
                "test_type": "analysis"
            },
            {
                "question": "Provide a summary of the validation results.",
                "expected_topics": ["false positive rate", "customer impact", "statistics"],
                "test_type": "summary"
            },
            {
                "question": "How can we reduce false positives in the future?",
                "expected_topics": ["recommendations", "improvements", "operational"],
                "test_type": "recommendation"
            }
        ]
        
        results = []
        for question_test in test_questions:
            test_case = ChatResponseTestCase(
                question_test["question"],
                validation_context,
                question_test["expected_topics"]
            )
            test_case.execute()
            
            # Evaluate response quality
            relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
            contextual_metric = ContextualRelevancyMetric(threshold=0.7)
            bias_metric = BiasMetric(threshold=0.8)
            
            relevancy_metric.measure(test_case)
            contextual_metric.measure(test_case)
            bias_metric.measure(test_case)
            
            # Check topic coverage
            response_lower = test_case.actual_output.lower()
            topics_covered = sum(1 for topic in question_test["expected_topics"] 
                               if topic.lower() in response_lower)
            
            results.append({
                "question": question_test["question"],
                "test_type": question_test["test_type"],
                "topics_covered": topics_covered,
                "total_expected_topics": len(question_test["expected_topics"]),
                "relevancy_score": relevancy_metric.score,
                "contextual_score": contextual_metric.score,
                "bias_score": bias_metric.score,
                "response_length": len(test_case.actual_output),
                "relevancy_passed": relevancy_metric.is_successful(),
                "contextual_passed": contextual_metric.is_successful(),
                "bias_passed": bias_metric.is_successful()
            })
        
        return results

# ==================== COMPREHENSIVE TEST RUNNER ====================

class ComprehensiveEvaluationRunner:
    """Main test runner that orchestrates all evaluations"""
    
    def __init__(self):
        self.validation_suite = OutageValidationTestSuite()
        self.dataset_suite = DatasetAnalysisTestSuite()
        self.chat_suite = ChatInterfaceTestSuite()
        self.results = {}
    
    def run_all_evaluations(self):
        """Run all evaluation suites"""
        logger.info("Starting comprehensive evaluation of Power Outage Analysis Agent...")
        
        try:
            # Run all test suites
            self.results["validation_basic"] = self.validation_suite.run_basic_accuracy_tests()
            self.results["validation_edge_cases"] = self.validation_suite.run_edge_case_tests()
            self.results["validation_stress"] = self.validation_suite.run_stress_tests()
            self.results["dataset_analysis"] = self.dataset_suite.run_analysis_tests()
            self.results["chat_interface"] = self.chat_suite.run_chat_tests()
            
            # Generate comprehensive report
            report = self._generate_comprehensive_report()
            
            # Save results
            self._save_results(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {"error": str(e)}
    
    def _generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        
        # Calculate metrics for each test suite
        validation_basic = self.results.get("validation_basic", [])
        validation_edges = self.results.get("validation_edge_cases", [])
        validation_stress = self.results.get("validation_stress", [])
        dataset_analysis = self.results.get("dataset_analysis", [])
        chat_interface = self.results.get("chat_interface", [])
        
        # Validation metrics
        validation_metrics = {
            "basic_tests": {
                "total": len(validation_basic),
                "accuracy_passed": sum(1 for r in validation_basic if r.get("accuracy_passed", False)),
                "weather_consistency_passed": sum(1 for r in validation_basic if r.get("weather_passed", False)),
                "reasoning_quality_passed": sum(1 for r in validation_basic if r.get("reasoning_passed", False)),
                "hallucination_passed": sum(1 for r in validation_basic if r.get("hallucination_passed", False)),
                "avg_accuracy": np.mean([r.get("accuracy_score", 0) for r in validation_basic]) if validation_basic else 0,
                "avg_weather_consistency": np.mean([r.get("weather_consistency", 0) for r in validation_basic]) if validation_basic else 0
            },
            "edge_case_tests": {
                "total": len(validation_edges),
                "reasonable_classifications": sum(1 for r in validation_edges if r.get("classification_reasonable", False)),
                "reasoning_passed": sum(1 for r in validation_edges if r.get("reasoning_passed", False)),
                "avg_reasoning_quality": np.mean([r.get("reasoning_quality", 0) for r in validation_edges]) if validation_edges else 0
            },
            "stress_tests": {
                "total": len(validation_stress),
                "graceful_error_handling": sum(1 for r in validation_stress if r.get("error_handled_gracefully", False)),
                "provides_reasoning": sum(1 for r in validation_stress if r.get("provides_reasoning", False))
            }
        }
        
        # Dataset analysis metrics
        dataset_metrics = {
            "total": len(dataset_analysis),
            "relevancy_passed": sum(1 for r in dataset_analysis if r.get("relevancy_passed", False)),
            "summarization_passed": sum(1 for r in dataset_analysis if r.get("summarization_passed", False)),
            "avg_relevancy": np.mean([r.get("relevancy_score", 0) for r in dataset_analysis]) if dataset_analysis else 0
        }
        
        # Chat interface metrics
        chat_metrics = {
            "total": len(chat_interface),
            "relevancy_passed": sum(1 for r in chat_interface if r.get("relevancy_passed", False)),
            "contextual_passed": sum(1 for r in chat_interface if r.get("contextual_passed", False)),
            "bias_passed": sum(1 for r in chat_interface if r.get("bias_passed", False)),
            "avg_topic_coverage": np.mean([r.get("topics_covered", 0) / max(r.get("total_expected_topics", 1), 1) 
                                         for r in chat_interface]) if chat_interface else 0
        }
        
        # Overall assessment
        total_tests = (validation_metrics["basic_tests"]["total"] + 
                      validation_metrics["edge_case_tests"]["total"] +
                      validation_metrics["stress_tests"]["total"] +
                      dataset_metrics["total"] + 
                      chat_metrics["total"])
        
        total_passed = (validation_metrics["basic_tests"]["accuracy_passed"] +
                       validation_metrics["edge_case_tests"]["reasonable_classifications"] +
                       validation_metrics["stress_tests"]["graceful_error_handling"] +
                       dataset_metrics["relevancy_passed"] +
                       chat_metrics["relevancy_passed"])
        
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "overall_pass_rate": overall_pass_rate,
                "agent_reliability": "High" if overall_pass_rate > 0.8 else "Medium" if overall_pass_rate > 0.6 else "Low"
            },
            "validation_metrics": validation_metrics,
            "dataset_metrics": dataset_metrics,
            "chat_metrics": chat_metrics,
            "detailed_results": self.results,
            "recommendations": self._generate_recommendations(overall_pass_rate, validation_metrics, dataset_metrics, chat_metrics)
        }
    
    def _generate_recommendations(self, overall_rate: float, validation: Dict, dataset: Dict, chat: Dict) -> List[str]:
        """Generate improvement recommendations based on test results"""
        recommendations = []
        
        if overall_rate < 0.8:
            recommendations.append("Overall performance needs improvement - consider fine-tuning prompts")
        
        if validation["basic_tests"]["avg_accuracy"] < 0.9:
            recommendations.append("Improve classification accuracy by enhancing weather threshold definitions")
        
        if validation["basic_tests"]["avg_weather_consistency"] < 0.8:
            recommendations.append("Strengthen weather-to-outage correlation logic in prompts")
        
        if validation["edge_case_tests"]["avg_reasoning_quality"] < 0.7:
            recommendations.append("Enhance reasoning quality for borderline cases")
        
        if validation["stress_tests"]["graceful_error_handling"] < validation["stress_tests"]["total"]:
            recommendations.append("Improve error handling for malformed or missing data")
        
        if dataset["avg_relevancy"] < 0.7:
            recommendations.append("Improve dataset analysis relevancy and insight generation")
        
        if chat["avg_topic_coverage"] < 0.8:
            recommendations.append("Enhance chat responses to better cover expected topics")
        
        if not recommendations:
            recommendations.append("Agent performance is excellent - maintain current prompt quality")
        
        return recommendations
    
    def _save_results(self, report: Dict):
        """Save evaluation results to files"""
        
        # Save comprehensive JSON report
        with open("comprehensive_evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate and save markdown report
        markdown_report = self._generate_markdown_report(report)
        with open("evaluation_report.md", "w") as f:
            f.write(markdown_report)
        
        logger.info("Evaluation results saved to comprehensive_evaluation_report.json and evaluation_report.md")
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate human-readable markdown report"""
        
        md = f"""# Power Outage Analysis Agent - Comprehensive Evaluation Report

**Generated:** {report['timestamp']}
**Agent Reliability:** {report['overall']['agent_reliability']}

## Executive Summary

- **Total Tests:** {report['overall']['total_tests']}
- **Overall Pass Rate:** {report['overall']['overall_pass_rate']:.1%}
- **Tests Passed:** {report['overall']['total_passed']}/{report['overall']['total_tests']}

## Detailed Results

### Outage Validation Tests

#### Basic Accuracy Tests
- **Total Tests:** {report['validation_metrics']['basic_tests']['total']}
- **Classification Accuracy:** {report['validation_metrics']['basic_tests']['accuracy_passed']}/{report['validation_metrics']['basic_tests']['total']} ({report['validation_metrics']['basic_tests']['avg_accuracy']:.2f} avg score)
- **Weather Consistency:** {report['validation_metrics']['basic_tests']['weather_consistency_passed']}/{report['validation_metrics']['basic_tests']['total']} ({report['validation_metrics']['basic_tests']['avg_weather_consistency']:.2f} avg score)
- **Reasoning Quality:** {report['validation_metrics']['basic_tests']['reasoning_quality_passed']}/{report['validation_metrics']['basic_tests']['total']}
- **Hallucination Prevention:** {report['validation_metrics']['basic_tests']['hallucination_passed']}/{report['validation_metrics']['basic_tests']['total']}

#### Edge Case Tests
- **Total Tests:** {report['validation_metrics']['edge_case_tests']['total']}
- **Reasonable Classifications:** {report['validation_metrics']['edge_case_tests']['reasonable_classifications']}/{report['validation_metrics']['edge_case_tests']['total']}
- **Average Reasoning Quality:** {report['validation_metrics']['edge_case_tests']['avg_reasoning_quality']:.2f}

#### Stress Tests
- **Total Tests:** {report['validation_metrics']['stress_tests']['total']}
- **Graceful Error Handling:** {report['validation_metrics']['stress_tests']['graceful_error_handling']}/{report['validation_metrics']['stress_tests']['total']}
- **Provides Reasoning:** {report['validation_metrics']['stress_tests']['provides_reasoning']}/{report['validation_metrics']['stress_tests']['total']}

### Dataset Analysis Tests
- **Total Tests:** {report['dataset_metrics']['total']}
- **Relevancy Passed:** {report['dataset_metrics']['relevancy_passed']}/{report['dataset_metrics']['total']}
- **Average Relevancy Score:** {report['dataset_metrics']['avg_relevancy']:.2f}

### Chat Interface Tests
- **Total Tests:** {report['chat_metrics']['total']}
- **Relevancy Passed:** {report['chat_metrics']['relevancy_passed']}/{report['chat_metrics']['total']}
- **Contextual Relevancy:** {report['chat_metrics']['contextual_passed']}/{report['chat_metrics']['total']}
- **Bias Prevention:** {report['chat_metrics']['bias_passed']}/{report['chat_metrics']['total']}
- **Average Topic Coverage:** {report['chat_metrics']['avg_topic_coverage']:.1%}

## Recommendations

"""
        
        for i, rec in enumerate(report['recommendations'], 1):
            md += f"{i}. {rec}\n"
        
        md += "\n## Conclusion\n\n"
        
        if report['overall']['overall_pass_rate'] > 0.8:
            md += "The Power Outage Analysis Agent demonstrates high reliability and accuracy across all decision points."
        elif report['overall']['overall_pass_rate'] > 0.6:
            md += "The agent shows good performance but has areas for improvement, particularly in edge case handling."
        else:
            md += "The agent requires significant improvements to meet production reliability standards."
        
        return md

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive DeepEvals for Power Outage Analysis Agent")
    print("=" * 80)
    
    # Initialize and run evaluations
    evaluator = ComprehensiveEvaluationRunner()
    report = evaluator.run_all_evaluations()
    
    if "error" in report:
        print(f"❌ Evaluation failed: {report['error']}")
        return
    
    # Display summary
    print(f"\n✅ Evaluation Complete!")
    print(f"📊 Overall Pass Rate: {report['overall']['overall_pass_rate']:.1%}")
    print(f"🎯 Agent Reliability: {report['overall']['agent_reliability']}")
    print(f"📁 Results saved to: comprehensive_evaluation_report.json")
    print(f"📄 Report saved to: evaluation_report.md")
    
    print(f"\n🔍 Key Findings:")
    for rec in report['recommendations'][:3]:  # Show top 3 recommendations
        print(f"  • {rec}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()