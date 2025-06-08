"""
Simplified DeepEvals for Power Outage Analysis Agent
Tests the core logic directly without tool interface complications
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== SIMPLIFIED METRICS ====================

class WeatherDecisionAccuracyMetric(BaseMetric):
    """Simplified weather decision accuracy metric"""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.name = "Weather Decision Accuracy"
    
    def measure(self, test_case: LLMTestCase) -> float:
        try:
            expected = test_case.expected_output.upper()
            actual = test_case.actual_output.upper()
            
            # Simple classification check
            expected_real = "REAL OUTAGE" in expected
            actual_real = "REAL OUTAGE" in actual
            
            self.score = 1.0 if expected_real == actual_real else 0.0
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Weather decision accuracy error: {e}")
            self.score = 0.0
            self.success = False
            return self.score
    
    def is_successful(self) -> bool:
        return self.success

class ReasoningQualityMetric(BaseMetric):
    """Simplified reasoning quality metric"""
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.name = "Reasoning Quality"
    
    def measure(self, test_case: LLMTestCase) -> float:
        try:
            response = test_case.actual_output.lower()
            
            # Basic quality indicators
            quality_indicators = [
                "wind" in response or "weather" in response,
                "temperature" in response or "precipitation" in response,
                len(response.split()) > 20,  # Sufficient detail
                any(word in response for word in ['because', 'due to', 'analysis', 'based on'])
            ]
            
            self.score = sum(quality_indicators) / len(quality_indicators)
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Reasoning quality error: {e}")
            self.score = 0.0
            self.success = False
            return self.score
    
    def is_successful(self) -> bool:
        return self.success

# ==================== DIRECT AGENT TESTING ====================

class DirectAgentTester:
    """Test agent by calling core LLM chains directly"""
    
    def __init__(self):
        self.results = {}
        self.test_scenarios = self._create_test_scenarios()
    
    def _create_test_scenarios(self):
        """Create test scenarios"""
        return [
            {
                "name": "Clear False Positive - Perfect Weather",
                "outage_report": {
                    "datetime": "2022-01-01 00:00:00",
                    "latitude": 40.0,
                    "longitude": -88.0,
                    "customers": 1
                },
                "weather_data": {
                    "temperature": 20,
                    "precipitation": 0,
                    "wind_speed": 5,
                    "wind_gusts": 8,
                    "snowfall": 0
                },
                "expected": "FALSE POSITIVE"
            },
            {
                "name": "Clear Real Outage - Severe Storm",
                "outage_report": {
                    "datetime": "2022-01-01 02:00:00", 
                    "latitude": 39.0,
                    "longitude": -87.0,
                    "customers": 50
                },
                "weather_data": {
                    "temperature": -5,
                    "precipitation": 15,
                    "wind_speed": 60,
                    "wind_gusts": 85,
                    "snowfall": 8
                },
                "expected": "REAL OUTAGE"
            },
            {
                "name": "Borderline Case - Moderate Conditions",
                "outage_report": {
                    "datetime": "2022-01-01 03:00:00",
                    "latitude": 38.5,
                    "longitude": -88.5, 
                    "customers": 15
                },
                "weather_data": {
                    "temperature": 0,
                    "precipitation": 8,
                    "wind_speed": 35,
                    "wind_gusts": 50,
                    "snowfall": 3
                },
                "expected": "BORDERLINE"
            }
        ]
    
    def test_weather_decisions_directly(self):
        """Test weather decisions by directly calling LLM chains"""
        logger.info("Testing weather decisions directly...")
        
        try:
            # Import the core LLM components
            from main import LLMManager, PromptManager
            
            # Initialize components
            llm_manager = LLMManager()
            llm = llm_manager.get_llm()
            
            with open('prompts.json', 'r') as f:
                prompts = json.load(f)
            
            # Create the validation chain manually
            from langchain_core.prompts import ChatPromptTemplate
            
            false_positive_prompt = ChatPromptTemplate.from_messages([
                ("system", prompts["false_positive_detection"]["system"]),
                ("human", prompts["false_positive_detection"]["human"])
            ])
            
            validation_chain = false_positive_prompt | llm
            
            results = []
            
            for scenario in self.test_scenarios:
                try:
                    # Format the prompt
                    response = validation_chain.invoke({
                        "outage_report": json.dumps(scenario["outage_report"], indent=2),
                        "weather_data": json.dumps(scenario["weather_data"], indent=2)
                    })
                    
                    # Extract response content
                    if hasattr(response, 'content'):
                        response_text = response.content
                    else:
                        response_text = str(response)
                    
                    # Create test case for evaluation
                    test_case = LLMTestCase(
                        input=json.dumps(scenario["outage_report"]),
                        actual_output=response_text,
                        expected_output=scenario["expected"],
                        context=[json.dumps(scenario["weather_data"])]
                    )
                    
                    # Apply metrics
                    accuracy_metric = WeatherDecisionAccuracyMetric()
                    reasoning_metric = ReasoningQualityMetric()
                    
                    accuracy_score = accuracy_metric.measure(test_case)
                    reasoning_score = reasoning_metric.measure(test_case)
                    
                    results.append({
                        "scenario": scenario["name"],
                        "expected": scenario["expected"],
                        "accuracy_score": accuracy_score,
                        "reasoning_score": reasoning_score,
                        "accuracy_passed": accuracy_metric.is_successful(),
                        "reasoning_passed": reasoning_metric.is_successful(),
                        "response_preview": response_text[:300] + "..." if len(response_text) > 300 else response_text
                    })
                    
                except Exception as e:
                    logger.error(f"Direct test error for {scenario['name']}: {e}")
                    results.append({
                        "scenario": scenario["name"],
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Direct testing setup error: {e}")
            return [{"error": f"Setup failed: {str(e)}"}]
    
    def test_chat_responses_directly(self):
        """Test chat responses directly"""
        logger.info("Testing chat responses directly...")
        
        try:
            from main import LLMManager
            from langchain_core.prompts import ChatPromptTemplate
            
            llm_manager = LLMManager()
            llm = llm_manager.get_llm()
            
            with open('prompts.json', 'r') as f:
                prompts = json.load(f)
            
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", prompts["chatbot_assistant"]["system"]),
                ("human", prompts["chatbot_assistant"]["human"])
            ])
            
            chat_chain = chat_prompt | llm
            
            # Simple chat test scenarios
            chat_tests = [
                {
                    "question": "What wind speed threshold is used for classification?",
                    "context": {"thresholds": {"wind_speed": 40, "wind_gusts": 56}},
                    "expected_topics": ["40", "wind", "threshold"]
                },
                {
                    "question": "How can we improve false positive detection?",
                    "context": {"false_positive_rate": 30},
                    "expected_topics": ["improve", "sensor", "threshold"]
                }
            ]
            
            results = []
            
            for test in chat_tests:
                try:
                    response = chat_chain.invoke({
                        "user_question": test["question"],
                        "analysis_context": json.dumps(test["context"])
                    })
                    
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Check if expected topics are covered
                    topics_covered = sum(1 for topic in test["expected_topics"] 
                                       if topic.lower() in response_text.lower())
                    
                    results.append({
                        "question": test["question"],
                        "topics_covered": topics_covered,
                        "total_topics": len(test["expected_topics"]),
                        "coverage_rate": topics_covered / len(test["expected_topics"]),
                        "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text
                    })
                    
                except Exception as e:
                    logger.error(f"Chat test error: {e}")
                    results.append({
                        "question": test["question"],
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Chat testing error: {e}")
            return [{"error": f"Chat testing failed: {str(e)}"}]
    
    def run_all_tests(self):
        """Run all simplified tests"""
        logger.info("Starting simplified agent testing...")
        
        self.results = {
            "weather_decisions": self.test_weather_decisions_directly(),
            "chat_responses": self.test_chat_responses_directly()
        }
        
        # Generate report
        report = self._generate_report()
        
        # Save results
        with open("simplified_evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_report(self):
        """Generate evaluation report"""
        weather_results = self.results.get("weather_decisions", [])
        chat_results = self.results.get("chat_responses", [])
        
        # Calculate weather metrics
        weather_accuracy = sum(1 for r in weather_results if r.get("accuracy_passed", False))
        weather_reasoning = sum(1 for r in weather_results if r.get("reasoning_passed", False))
        weather_total = len([r for r in weather_results if "error" not in r])
        
        # Calculate chat metrics
        chat_coverage = sum(r.get("coverage_rate", 0) for r in chat_results if "error" not in r)
        chat_total = len([r for r in chat_results if "error" not in r])
        
        total_tests = weather_total + chat_total
        total_passed = weather_accuracy + (chat_total if chat_coverage / chat_total > 0.7 else 0)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
                "agent_reliability": "High" if total_passed / total_tests > 0.8 else "Medium" if total_passed / total_tests > 0.6 else "Low"
            },
            "weather_decisions": {
                "total_tests": weather_total,
                "accuracy_passed": weather_accuracy,
                "reasoning_passed": weather_reasoning,
                "accuracy_rate": weather_accuracy / weather_total if weather_total > 0 else 0,
                "reasoning_rate": weather_reasoning / weather_total if weather_total > 0 else 0
            },
            "chat_responses": {
                "total_tests": chat_total,
                "average_coverage": chat_coverage / chat_total if chat_total > 0 else 0
            },
            "detailed_results": self.results
        }

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution"""
    print("🎯 Simplified DeepEvals for Power Outage Analysis Agent")
    print("=" * 60)
    
    tester = DirectAgentTester()
    report = tester.run_all_tests()
    
    print(f"\n✅ Evaluation Complete!")
    print(f"📊 Pass Rate: {report['summary']['pass_rate']:.1%}")
    print(f"🎯 Agent Reliability: {report['summary']['agent_reliability']}")
    
    weather = report['weather_decisions']
    print(f"\n🌤️  Weather Decisions:")
    print(f"   Accuracy: {weather['accuracy_passed']}/{weather['total_tests']} ({weather['accuracy_rate']:.1%})")
    print(f"   Reasoning: {weather['reasoning_passed']}/{weather['total_tests']} ({weather['reasoning_rate']:.1%})")
    
    chat = report['chat_responses']
    print(f"\n💬 Chat Responses:")
    print(f"   Average Coverage: {chat['average_coverage']:.1%}")
    
    print(f"\n📁 Results saved to: simplified_evaluation_report.json")

if __name__ == "__main__":
    main()