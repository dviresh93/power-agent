"""
Production-Ready DeepEvals for Power Outage Analysis Agent
Follows the exact same pattern as tests.py for proper LangChain tool invocation
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

# Import tools using the same pattern as tests.py
from main import validate_outage_report, chat_about_results

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DEEPEVAL METRICS ====================

class WeatherThresholdAccuracyMetric(BaseMetric):
    """Evaluates weather threshold decision accuracy"""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.name = "Weather Threshold Accuracy"
    
    def measure(self, test_case: LLMTestCase) -> float:
        try:
            # Parse weather data from context (context is a list of strings)
            weather_data = {}
            if test_case.context and len(test_case.context) > 0:
                weather_data = json.loads(test_case.context[0])
            response = test_case.actual_output.upper()
            
            # Apply the same weather thresholds as defined in prompts.json
            severe_wind = weather_data.get('wind_speed', 0) > 40 or weather_data.get('wind_gusts', 0) > 56
            heavy_precip = weather_data.get('precipitation', 0) > 12.7
            extreme_temp = weather_data.get('temperature', 20) < -12 or weather_data.get('temperature', 20) > 35
            heavy_snow = weather_data.get('snowfall', 0) > 5
            
            severe_weather = any([severe_wind, heavy_precip, extreme_temp, heavy_snow])
            classified_as_real = "REAL OUTAGE" in response
            
            # Score based on correct threshold application
            if severe_weather and classified_as_real:
                self.score = 1.0  # Correctly identified severe weather as real outage
            elif not severe_weather and not classified_as_real:
                self.score = 1.0  # Correctly identified mild weather as false positive
            else:
                self.score = 0.0  # Incorrect classification
            
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Weather threshold accuracy measurement error: {e}")
            self.score = 0.0
            self.success = False
            return self.score
    
    def is_successful(self) -> bool:
        return self.success

class ReasoningQualityMetric(BaseMetric):
    """Evaluates quality of reasoning in agent responses"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.name = "Reasoning Quality"
    
    def measure(self, test_case: LLMTestCase) -> float:
        try:
            response = test_case.actual_output.lower()
            
            # Quality indicators for detailed reasoning
            reasoning_indicators = [
                any(word in response for word in ['wind', 'temperature', 'precipitation', 'weather']),
                any(word in response for word in ['because', 'due to', 'caused by', 'analysis', 'based on']),
                any(word in response for word in ['mph', 'km/h', 'degrees', 'celsius', 'fahrenheit']),
                len(response.split()) > 30,  # Sufficient detail
                any(word in response for word in ['severe', 'mild', 'extreme', 'moderate'])
            ]
            
            self.score = sum(reasoning_indicators) / len(reasoning_indicators)
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Reasoning quality measurement error: {e}")
            self.score = 0.0
            self.success = False
            return self.score
    
    def is_successful(self) -> bool:
        return self.success

class ChatResponseQualityMetric(BaseMetric):
    """Evaluates chat response helpfulness and accuracy"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.name = "Chat Response Quality"
    
    def measure(self, test_case: LLMTestCase) -> float:
        try:
            response = test_case.actual_output.lower()
            
            # Quality indicators for helpful chat responses
            quality_indicators = [
                any(word in response for word in ['recommend', 'suggest', 'should', 'could']),
                any(word in response for word in ['threshold', 'criteria', 'analysis', 'system']),
                any(word in response for word in ['improve', 'enhance', 'reduce', 'prevent']),
                len(response.split()) > 20,  # Sufficient detail
                any(phrase in response for phrase in ['false positive', 'outage detection', 'weather conditions'])
            ]
            
            self.score = sum(quality_indicators) / len(quality_indicators)
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Chat response quality measurement error: {e}")
            self.score = 0.0
            self.success = False
            return self.score
    
    def is_successful(self) -> bool:
        return self.success

# ==================== TEST CASE CLASSES ====================

class OutageValidationTestCase(LLMTestCase):
    """Test case for outage validation - follows same pattern as tests.py"""
    
    def __init__(self, outage_report: Dict, weather_data: Dict, expected_classification: str, scenario_name: str):
        self.outage_report = outage_report
        self.weather_data = weather_data
        self.scenario_name = scenario_name
        
        # Set required LLMTestCase fields
        super().__init__(
            input=json.dumps(outage_report),
            actual_output="",
            expected_output=expected_classification,
            context=[json.dumps(weather_data)]  # Context must be list of strings
        )
    
    def run(self):
        """Run the test case using the exact same pattern as tests.py"""
        try:
            # Call the validation function using the same pattern as tests.py line 85-88
            self.actual_output = validate_outage_report.invoke({
                "outage_report": self.outage_report,
                "weather_data": self.weather_data
            })
            return self.actual_output
        except Exception as e:
            logger.error(f"Error running validation test case {self.scenario_name}: {str(e)}")
            self.actual_output = f"ERROR: {str(e)}"
            return self.actual_output

class ChatResponseTestCase(LLMTestCase):
    """Test case for chat responses - follows same pattern as tests.py"""
    
    def __init__(self, question: str, context: Dict, expected_topics: List[str], scenario_name: str):
        self.question = question
        self.context_data = context
        self.expected_topics = expected_topics
        self.scenario_name = scenario_name
        
        # Set required LLMTestCase fields
        super().__init__(
            input=question,
            actual_output="",
            expected_output=f"Response covering: {', '.join(expected_topics)}",
            context=[json.dumps(context)]  # Context must be list of strings
        )
    
    def run(self):
        """Run the test case using the exact same pattern as tests.py"""
        try:
            # Call the chat function using the same pattern as tests.py line 157-160
            self.actual_output = chat_about_results.invoke({
                "question": self.question,
                "context": self.context_data
            })
            return self.actual_output
        except Exception as e:
            logger.error(f"Error running chat test case {self.scenario_name}: {str(e)}")
            self.actual_output = f"ERROR: {str(e)}"
            return self.actual_output

# ==================== TEST SCENARIO DEFINITIONS ====================

class ProductionTestRunner:
    """Production-ready test runner following tests.py patterns"""
    
    def __init__(self):
        self.results = {}
        self.weather_test_scenarios = self._create_weather_test_scenarios()
        self.chat_test_scenarios = self._create_chat_test_scenarios()
    
    def _create_weather_test_scenarios(self) -> List[Dict]:
        """Create comprehensive weather test scenarios"""
        return [
            {
                "name": "Clear False Positive - Perfect Conditions",
                "outage_report": {
                    "datetime": "2022-01-01 00:00:00",
                    "latitude": 40.198069,
                    "longitude": -88.262575,
                    "customers": 1
                },
                "weather_data": {
                    "temperature": 18.0,
                    "precipitation": 0.0,
                    "wind_speed": 8.0,
                    "wind_gusts": 12.0,
                    "snowfall": 0.0
                },
                "expected": "FALSE POSITIVE"
            },
            {
                "name": "Clear Real Outage - High Wind Event",
                "outage_report": {
                    "datetime": "2022-01-01 01:30:00",
                    "latitude": 39.456789,
                    "longitude": -87.654321,
                    "customers": 45
                },
                "weather_data": {
                    "temperature": 2.0,
                    "precipitation": 3.2,
                    "wind_speed": 52.0,  # Above 40 km/h threshold
                    "wind_gusts": 68.0,  # Above 56 km/h threshold
                    "snowfall": 1.0
                },
                "expected": "REAL OUTAGE"
            },
            {
                "name": "Extreme Temperature Event",
                "outage_report": {
                    "datetime": "2022-01-01 02:15:00",
                    "latitude": 38.822124,
                    "longitude": -89.791293,
                    "customers": 23
                },
                "weather_data": {
                    "temperature": -18.0,  # Below -12°C threshold
                    "precipitation": 0.5,
                    "wind_speed": 20.0,
                    "wind_gusts": 28.0,
                    "snowfall": 2.0
                },
                "expected": "REAL OUTAGE"
            },
            {
                "name": "Heavy Precipitation Event",
                "outage_report": {
                    "datetime": "2022-01-01 03:45:00",
                    "latitude": 40.123456,
                    "longitude": -88.987654,
                    "customers": 18
                },
                "weather_data": {
                    "temperature": 15.0,
                    "precipitation": 18.5,  # Above 12.7 mm/h threshold
                    "wind_speed": 25.0,
                    "wind_gusts": 35.0,
                    "snowfall": 0.0
                },
                "expected": "REAL OUTAGE"
            },
            {
                "name": "Borderline Wind Conditions",
                "outage_report": {
                    "datetime": "2022-01-01 04:20:00",
                    "latitude": 39.555555,
                    "longitude": -88.333333,
                    "customers": 8
                },
                "weather_data": {
                    "temperature": 10.0,
                    "precipitation": 0.2,
                    "wind_speed": 38.0,  # Just below 40 km/h threshold
                    "wind_gusts": 54.0,  # Just below 56 km/h threshold
                    "snowfall": 0.0
                },
                "expected": "FALSE POSITIVE"  # Should lean toward false positive for borderline
            }
        ]
    
    def _create_chat_test_scenarios(self) -> List[Dict]:
        """Create chat response test scenarios"""
        return [
            {
                "name": "Technical Threshold Inquiry",
                "question": "What wind speed threshold does the system use to classify outages as real?",
                "context": {
                    "validation_results": {
                        "thresholds": {
                            "wind_speed": 40,
                            "wind_gusts": 56,
                            "precipitation": 12.7,
                            "temperature_low": -12,
                            "temperature_high": 35
                        }
                    }
                },
                "expected_topics": ["40", "wind", "threshold", "km/h"]
            },
            {
                "name": "False Positive Reduction Strategy",
                "question": "How can we reduce false positives in our outage detection system?",
                "context": {
                    "validation_results": {
                        "false_positive_rate": 35.5,
                        "main_causes": ["mild weather conditions", "sensor errors", "system testing"]
                    }
                },
                "expected_topics": ["sensor", "threshold", "improve", "accuracy"]
            },
            {
                "name": "Statistical Summary Request",
                "question": "What was the overall false positive rate in this analysis?",
                "context": {
                    "validation_results": {
                        "total_reports": 110,
                        "real_outages": 68,
                        "false_positives": 42,
                        "false_positive_rate": 38.2
                    }
                },
                "expected_topics": ["38.2", "rate", "reports", "analysis"]
            }
        ]
    
    def run_weather_validation_tests(self):
        """Run weather validation tests using production patterns"""
        logger.info("Running weather validation tests...")
        
        results = []
        
        for scenario in self.weather_test_scenarios:
            # Create test case using the same pattern as tests.py
            test_case = OutageValidationTestCase(
                outage_report=scenario["outage_report"],
                weather_data=scenario["weather_data"],
                expected_classification=scenario["expected"],
                scenario_name=scenario["name"]
            )
            
            # Run the test case
            response = test_case.run()
            
            # Apply DeepEval metrics
            accuracy_metric = WeatherThresholdAccuracyMetric()
            reasoning_metric = ReasoningQualityMetric()
            
            accuracy_score = accuracy_metric.measure(test_case)
            reasoning_score = reasoning_metric.measure(test_case)
            
            results.append({
                "scenario_name": scenario["name"],
                "expected_classification": scenario["expected"],
                "actual_output": response,
                "accuracy_score": accuracy_score,
                "reasoning_score": reasoning_score,
                "accuracy_passed": accuracy_metric.is_successful(),
                "reasoning_passed": reasoning_metric.is_successful(),
                "weather_conditions": scenario["weather_data"],
                "response_preview": response[:250] + "..." if len(response) > 250 else response
            })
        
        return results
    
    def run_chat_response_tests(self):
        """Run chat response tests using production patterns"""
        logger.info("Running chat response tests...")
        
        results = []
        
        for scenario in self.chat_test_scenarios:
            # Create test case using the same pattern as tests.py
            test_case = ChatResponseTestCase(
                question=scenario["question"],
                context=scenario["context"],
                expected_topics=scenario["expected_topics"],
                scenario_name=scenario["name"]
            )
            
            # Run the test case
            response = test_case.run()
            
            # Apply DeepEval metrics
            quality_metric = ChatResponseQualityMetric()
            quality_score = quality_metric.measure(test_case)
            
            # Check topic coverage
            response_lower = response.lower()
            topics_covered = sum(1 for topic in scenario["expected_topics"] 
                               if topic.lower() in response_lower)
            
            results.append({
                "scenario_name": scenario["name"],
                "question": scenario["question"],
                "actual_output": response,
                "quality_score": quality_score,
                "quality_passed": quality_metric.is_successful(),
                "topics_covered": topics_covered,
                "total_topics": len(scenario["expected_topics"]),
                "topic_coverage_rate": topics_covered / len(scenario["expected_topics"]),
                "response_preview": response[:250] + "..." if len(response) > 250 else response
            })
        
        return results
    
    def run_all_tests(self):
        """Run all production-ready tests"""
        logger.info("Starting production-ready DeepEval testing...")
        
        self.results = {
            "weather_validation": self.run_weather_validation_tests(),
            "chat_responses": self.run_chat_response_tests()
        }
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        
        # Save results
        with open("production_evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        
        weather_results = self.results.get("weather_validation", [])
        chat_results = self.results.get("chat_responses", [])
        
        # Calculate weather validation metrics
        weather_accuracy_passed = sum(1 for r in weather_results if r.get("accuracy_passed", False))
        weather_reasoning_passed = sum(1 for r in weather_results if r.get("reasoning_passed", False))
        weather_total = len([r for r in weather_results if not r["actual_output"].startswith("ERROR:")])
        
        # Calculate chat response metrics
        chat_quality_passed = sum(1 for r in chat_results if r.get("quality_passed", False))
        chat_total = len([r for r in chat_results if not r["actual_output"].startswith("ERROR:")])
        
        total_tests = weather_total + chat_total
        total_core_passed = weather_accuracy_passed + chat_quality_passed
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "total_passed": total_core_passed,
                "pass_rate": total_core_passed / total_tests if total_tests > 0 else 0,
                "agent_reliability": "High" if total_core_passed / total_tests > 0.8 else "Medium" if total_core_passed / total_tests > 0.6 else "Low"
            },
            "weather_validation": {
                "total_tests": weather_total,
                "accuracy_tests_passed": weather_accuracy_passed,
                "reasoning_tests_passed": weather_reasoning_passed,
                "accuracy_rate": weather_accuracy_passed / weather_total if weather_total > 0 else 0,
                "reasoning_rate": weather_reasoning_passed / weather_total if weather_total > 0 else 0
            },
            "chat_responses": {
                "total_tests": chat_total,
                "quality_tests_passed": chat_quality_passed,
                "quality_rate": chat_quality_passed / chat_total if chat_total > 0 else 0,
                "average_topic_coverage": sum(r.get("topic_coverage_rate", 0) for r in chat_results if not r["actual_output"].startswith("ERROR:")) / chat_total if chat_total > 0 else 0
            },
            "detailed_results": self.results
        }

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution for production-ready DeepEvals"""
    print("🎯 Production-Ready DeepEvals for Power Outage Analysis Agent")
    print("=" * 70)
    
    runner = ProductionTestRunner()
    report = runner.run_all_tests()
    
    print(f"\n✅ Evaluation Complete!")
    print(f"📊 Pass Rate: {report['summary']['pass_rate']:.1%}")
    print(f"🎯 Agent Reliability: {report['summary']['agent_reliability']}")
    
    weather = report['weather_validation']
    print(f"\n🌤️  Weather Validation:")
    print(f"   Accuracy: {weather['accuracy_tests_passed']}/{weather['total_tests']} ({weather['accuracy_rate']:.1%})")
    print(f"   Reasoning: {weather['reasoning_tests_passed']}/{weather['total_tests']} ({weather['reasoning_rate']:.1%})")
    
    chat = report['chat_responses']
    print(f"\n💬 Chat Responses:")
    print(f"   Quality: {chat['quality_tests_passed']}/{chat['total_tests']} ({chat['quality_rate']:.1%})")
    print(f"   Topic Coverage: {chat['average_topic_coverage']:.1%}")
    
    print(f"\n📁 Results saved to: production_evaluation_report.json")

if __name__ == "__main__":
    main()