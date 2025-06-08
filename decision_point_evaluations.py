"""
Decision Point Specific Evaluations for Power Outage Analysis Agent
Focused evaluation on each critical decision the agent makes
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DECISION POINT SPECIFIC METRICS ====================

class WeatherThresholdAccuracyMetric(BaseMetric):
    """Evaluates accuracy of weather threshold application"""
    
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.name = "Weather Threshold Accuracy"
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Measure how accurately weather thresholds are applied"""
        try:
            weather_data = json.loads(test_case.context)
            response = test_case.actual_output.upper()
            
            # Extract weather values
            wind_speed = weather_data.get('wind_speed', 0)
            wind_gusts = weather_data.get('wind_gusts', 0) 
            precipitation = weather_data.get('precipitation', 0)
            temperature = weather_data.get('temperature', 20)
            snowfall = weather_data.get('snowfall', 0)
            
            # Check threshold adherence based on agent's criteria
            severe_wind = wind_speed > 40 or wind_gusts > 56  # km/h thresholds
            heavy_precip = precipitation > 12.7  # mm/h
            extreme_temp = temperature < -12 or temperature > 35  # Celsius
            heavy_snow = snowfall > 5  # cm
            
            severe_weather = severe_wind or heavy_precip or extreme_temp or heavy_snow
            classified_as_real = "REAL OUTAGE" in response
            
            # Score based on threshold adherence
            if severe_weather and classified_as_real:
                self.score = 1.0  # Correct severe weather -> real outage
            elif not severe_weather and not classified_as_real:
                self.score = 1.0  # Correct mild weather -> false positive
            elif severe_weather and not classified_as_real:
                # Severe weather but classified as false positive - check reasoning
                if any(phrase in response for phrase in ["BORDERLINE", "MODERATE", "UNCERTAIN"]):
                    self.score = 0.7  # Acknowledged uncertainty
                else:
                    self.score = 0.0  # Missed severe weather
            else:
                # Mild weather but classified as real - check for other factors
                customer_count = json.loads(test_case.input).get('customers', 0)
                if customer_count > 20:  # Large outage despite mild weather
                    self.score = 0.5  # Partial credit for considering scale
                else:
                    self.score = 0.0  # Incorrect classification
            
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Weather threshold accuracy error: {e}")
            self.score = 0.0
            self.success = False
            return self.score
    
    def is_successful(self) -> bool:
        return self.success

class CustomerImpactConsistencyMetric(BaseMetric):
    """Evaluates consistency of customer impact assessment"""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.name = "Customer Impact Consistency"
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Measure if customer impact is properly considered"""
        try:
            outage_data = json.loads(test_case.input)
            response = test_case.actual_output.lower()
            
            customer_count = outage_data.get('customers', 0)
            
            # Check if customer impact is mentioned appropriately
            mentions_customers = any(word in response for word in ['customer', 'affected', 'impact', 'scale'])
            mentions_numbers = any(str(i) in response for i in range(max(1, customer_count-2), customer_count+3))
            
            # Score based on appropriate consideration
            if customer_count >= 50:  # Large outage
                expects_emphasis = any(word in response for word in ['large', 'significant', 'major', 'substantial'])
                self.score = 1.0 if expects_emphasis else 0.5
            elif customer_count >= 10:  # Medium outage
                expects_mention = mentions_customers
                self.score = 1.0 if expects_mention else 0.6
            else:  # Small outage
                # For small outages, mentioning size helps with false positive detection
                self.score = 1.0 if mentions_customers else 0.8
            
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Customer impact consistency error: {e}")
            self.score = 0.0
            self.success = False
            return self.score
    
    def is_successful(self) -> bool:
        return self.success

class TemporalPatternRecognitionMetric(BaseMetric):
    """Evaluates recognition of temporal patterns in dataset analysis"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.name = "Temporal Pattern Recognition"
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Measure quality of temporal pattern analysis"""
        try:
            response = test_case.actual_output.lower()
            
            # Expected temporal analysis elements
            time_elements = [
                any(word in response for word in ['hour', 'time', 'temporal', 'when']),
                any(word in response for word in ['peak', 'pattern', 'trend', 'distribution']),
                any(word in response for word in ['night', 'day', 'morning', 'evening', 'midnight']),
                any(word in response for word in ['frequency', 'cluster', 'concentration']),
                any(number in response for number in ['0', '1', '2', '3', '4', '5'])  # Specific hours
            ]
            
            # Score based on temporal analysis depth
            self.score = sum(time_elements) / len(time_elements)
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Temporal pattern recognition error: {e}")
            self.score = 0.0
            self.success = False
            return self.score
    
    def is_successful(self) -> bool:
        return self.success

class RecommendationQualityMetric(BaseMetric):
    """Evaluates quality of operational recommendations"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.name = "Recommendation Quality"
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Measure quality and actionability of recommendations"""
        try:
            response = test_case.actual_output.lower()
            
            # Quality indicators for recommendations
            quality_elements = [
                any(word in response for word in ['recommend', 'suggest', 'should', 'could', 'improve']),
                any(word in response for word in ['sensor', 'equipment', 'monitoring', 'system']),
                any(word in response for word in ['threshold', 'criteria', 'validation', 'detection']),
                any(word in response for word in ['reduce', 'prevent', 'minimize', 'avoid']),
                any(word in response for word in ['operational', 'maintenance', 'upgrade', 'training'])
            ]
            
            # Check for specific, actionable recommendations
            actionable_indicators = [
                any(phrase in response for phrase in ['check', 'review', 'update', 'implement']),
                any(phrase in response for phrase in ['false positive', 'accuracy', 'reliability']),
                len([word for word in response.split() if word in ['should', 'must', 'need', 'require']]) > 0
            ]
            
            quality_score = sum(quality_elements) / len(quality_elements)
            actionable_score = sum(actionable_indicators) / len(actionable_indicators)
            
            self.score = (quality_score + actionable_score) / 2
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Recommendation quality error: {e}")
            self.score = 0.0
            self.success = False
            return self.score
    
    def is_successful(self) -> bool:
        return self.success

# ==================== DECISION POINT TEST SCENARIOS ====================

@dataclass
class DecisionPointTest:
    """Test case for specific decision points"""
    decision_point: str
    test_name: str
    inputs: Dict
    expected_behavior: str
    evaluation_criteria: List[str]

class DecisionPointTestGenerator:
    """Generate tests for each decision point"""
    
    def generate_weather_threshold_tests(self) -> List[DecisionPointTest]:
        """Tests for weather threshold decision point"""
        return [
            DecisionPointTest(
                decision_point="weather_threshold_evaluation",
                test_name="Exact Threshold Boundary - Wind Speed",
                inputs={
                    "outage_report": {"datetime": "2022-01-01 12:00:00", "latitude": 40.0, "longitude": -88.0, "customers": 10},
                    "weather_data": {"temperature": 15, "precipitation": 0, "wind_speed": 40.0, "wind_gusts": 35, "snowfall": 0}
                },
                expected_behavior="Should classify as borderline or real outage due to wind speed exactly at threshold",
                evaluation_criteria=["threshold_adherence", "reasoning_quality", "uncertainty_acknowledgment"]
            ),
            
            DecisionPointTest(
                decision_point="weather_threshold_evaluation", 
                test_name="Multiple Moderate Factors",
                inputs={
                    "outage_report": {"datetime": "2022-01-01 12:00:00", "latitude": 40.0, "longitude": -88.0, "customers": 25},
                    "weather_data": {"temperature": 0, "precipitation": 8.0, "wind_speed": 35, "wind_gusts": 50, "snowfall": 3}
                },
                expected_behavior="Should consider cumulative effect of multiple moderate weather factors",
                evaluation_criteria=["cumulative_assessment", "factor_weighting", "decision_reasoning"]
            ),
            
            DecisionPointTest(
                decision_point="weather_threshold_evaluation",
                test_name="Extreme Single Factor",
                inputs={
                    "outage_report": {"datetime": "2022-01-01 12:00:00", "latitude": 40.0, "longitude": -88.0, "customers": 5},
                    "weather_data": {"temperature": -25, "precipitation": 0, "wind_speed": 10, "wind_gusts": 15, "snowfall": 0}
                },
                expected_behavior="Should classify as real outage due to extreme temperature despite other mild factors",
                evaluation_criteria=["single_factor_dominance", "extreme_value_recognition", "threshold_override"]
            )
        ]
    
    def generate_customer_impact_tests(self) -> List[DecisionPointTest]:
        """Tests for customer impact assessment"""
        return [
            DecisionPointTest(
                decision_point="customer_impact_assessment",
                test_name="High Customer Count Mild Weather",
                inputs={
                    "outage_report": {"datetime": "2022-01-01 12:00:00", "latitude": 40.0, "longitude": -88.0, "customers": 100},
                    "weather_data": {"temperature": 20, "precipitation": 0, "wind_speed": 15, "wind_gusts": 20, "snowfall": 0}
                },
                expected_behavior="Should question large customer impact with mild weather conditions",
                evaluation_criteria=["impact_weather_correlation", "anomaly_detection", "investigation_suggestion"]
            ),
            
            DecisionPointTest(
                decision_point="customer_impact_assessment",
                test_name="Low Customer Count Severe Weather", 
                inputs={
                    "outage_report": {"datetime": "2022-01-01 12:00:00", "latitude": 40.0, "longitude": -88.0, "customers": 2},
                    "weather_data": {"temperature": -15, "precipitation": 15, "wind_speed": 60, "wind_gusts": 80, "snowfall": 10}
                },
                expected_behavior="Should classify as real outage despite low customer count due to severe weather",
                evaluation_criteria=["weather_priority", "localized_impact", "severity_recognition"]
            )
        ]
    
    def generate_temporal_analysis_tests(self) -> List[DecisionPointTest]:
        """Tests for temporal pattern analysis"""
        return [
            DecisionPointTest(
                decision_point="temporal_pattern_analysis",
                test_name="Midnight Hour Clustering",
                inputs={
                    "dataset_summary": {
                        "temporal_patterns": {
                            "hourly_distribution": {"0": 50, "1": 30, "2": 15, "3": 5, "4": 5, "5": 5},
                            "peak_hour": 0
                        }
                    },
                    "sample_reports": [
                        {"datetime": "2022-01-01 00:00:00", "customers": 1},
                        {"datetime": "2022-01-01 00:15:00", "customers": 1}, 
                        {"datetime": "2022-01-01 00:30:00", "customers": 1}
                    ]
                },
                expected_behavior="Should identify suspicious clustering at midnight as potential system issue",
                evaluation_criteria=["pattern_detection", "anomaly_identification", "system_issue_hypothesis"]
            )
        ]
    
    def generate_chat_response_tests(self) -> List[DecisionPointTest]:
        """Tests for chat response quality"""
        return [
            DecisionPointTest(
                decision_point="chat_response_generation",
                test_name="Technical Question About Thresholds",
                inputs={
                    "question": "What wind speed threshold does the system use to classify outages as real?",
                    "context": {"validation_results": {"thresholds": {"wind_speed": 40, "wind_gusts": 56}}}
                },
                expected_behavior="Should provide specific technical details with units and context",
                evaluation_criteria=["technical_accuracy", "unit_specification", "context_awareness"]
            ),
            
            DecisionPointTest(
                decision_point="chat_response_generation",
                test_name="Operational Improvement Question", 
                inputs={
                    "question": "How can we reduce false positives in our outage detection system?",
                    "context": {"validation_results": {"false_positive_rate": 40, "main_causes": ["mild weather", "sensor errors"]}}
                },
                expected_behavior="Should provide actionable recommendations based on analysis results",
                evaluation_criteria=["actionability", "relevance_to_results", "implementation_feasibility"]
            )
        ]

# ==================== DECISION POINT EVALUATOR ====================

class DecisionPointEvaluator:
    """Evaluate specific decision points of the agent"""
    
    def __init__(self):
        self.test_generator = DecisionPointTestGenerator()
        self.results = {}
    
    def evaluate_weather_threshold_decisions(self):
        """Evaluate weather threshold decision making"""
        logger.info("Evaluating weather threshold decisions...")
        
        from main import validate_outage_report
        
        tests = self.test_generator.generate_weather_threshold_tests()
        results = []
        
        for test in tests:
            try:
                # Execute the test
                response = validate_outage_report.invoke({
                    "outage_report": test.inputs["outage_report"],
                    "weather_data": test.inputs["weather_data"]
                })
                
                # Create test case for evaluation
                test_case = LLMTestCase(
                    input=json.dumps(test.inputs["outage_report"]),
                    actual_output=response,
                    expected_output=test.expected_behavior,
                    context=json.dumps(test.inputs["weather_data"])
                )
                
                # Apply decision-point specific metrics
                threshold_metric = WeatherThresholdAccuracyMetric()
                impact_metric = CustomerImpactConsistencyMetric()
                
                threshold_metric.measure(test_case)
                impact_metric.measure(test_case)
                
                results.append({
                    "test_name": test.test_name,
                    "decision_point": test.decision_point,
                    "response": response,
                    "threshold_accuracy": threshold_metric.score,
                    "impact_consistency": impact_metric.score,
                    "threshold_passed": threshold_metric.is_successful(),
                    "impact_passed": impact_metric.is_successful(),
                    "evaluation_criteria": test.evaluation_criteria
                })
                
            except Exception as e:
                logger.error(f"Weather threshold test error: {e}")
                results.append({
                    "test_name": test.test_name,
                    "error": str(e)
                })
        
        return results
    
    def evaluate_temporal_analysis_decisions(self):
        """Evaluate temporal pattern analysis"""
        logger.info("Evaluating temporal analysis decisions...")
        
        from main import analyze_outage_reports_dataset
        
        tests = self.test_generator.generate_temporal_analysis_tests()
        results = []
        
        for test in tests:
            try:
                # Execute the test
                response = analyze_outage_reports_dataset.invoke({
                    "dataset_summary": test.inputs["dataset_summary"],
                    "sample_reports": test.inputs["sample_reports"]
                })
                
                # Create test case for evaluation
                test_case = LLMTestCase(
                    input=json.dumps(test.inputs),
                    actual_output=response,
                    expected_output=test.expected_behavior,
                    context=json.dumps(test.inputs["dataset_summary"])
                )
                
                # Apply temporal analysis metrics
                temporal_metric = TemporalPatternRecognitionMetric()
                temporal_metric.measure(test_case)
                
                results.append({
                    "test_name": test.test_name,
                    "decision_point": test.decision_point,
                    "response": response[:500] + "..." if len(response) > 500 else response,
                    "temporal_recognition": temporal_metric.score,
                    "temporal_passed": temporal_metric.is_successful(),
                    "evaluation_criteria": test.evaluation_criteria
                })
                
            except Exception as e:
                logger.error(f"Temporal analysis test error: {e}")
                results.append({
                    "test_name": test.test_name,
                    "error": str(e)
                })
        
        return results
    
    def evaluate_chat_response_decisions(self):
        """Evaluate chat response decisions"""
        logger.info("Evaluating chat response decisions...")
        
        from main import chat_about_validation_results
        
        tests = self.test_generator.generate_chat_response_tests()
        results = []
        
        for test in tests:
            try:
                # Execute the test
                response = chat_about_validation_results.invoke({
                    "question": test.inputs["question"],
                    "context": test.inputs["context"]
                })
                
                # Create test case for evaluation
                test_case = LLMTestCase(
                    input=test.inputs["question"],
                    actual_output=response,
                    expected_output=test.expected_behavior,
                    context=json.dumps(test.inputs["context"])
                )
                
                # Apply chat response metrics
                recommendation_metric = RecommendationQualityMetric()
                recommendation_metric.measure(test_case)
                
                results.append({
                    "test_name": test.test_name,
                    "decision_point": test.decision_point,
                    "question": test.inputs["question"],
                    "response": response[:500] + "..." if len(response) > 500 else response,
                    "recommendation_quality": recommendation_metric.score,
                    "recommendation_passed": recommendation_metric.is_successful(),
                    "evaluation_criteria": test.evaluation_criteria
                })
                
            except Exception as e:
                logger.error(f"Chat response test error: {e}")
                results.append({
                    "test_name": test.test_name,
                    "error": str(e)
                })
        
        return results
    
    def run_all_decision_point_evaluations(self):
        """Run all decision point evaluations"""
        logger.info("Running comprehensive decision point evaluations...")
        
        self.results = {
            "weather_threshold": self.evaluate_weather_threshold_decisions(),
            "temporal_analysis": self.evaluate_temporal_analysis_decisions(), 
            "chat_responses": self.evaluate_chat_response_decisions()
        }
        
        # Generate summary report
        report = self._generate_decision_point_report()
        
        # Save results
        with open("decision_point_evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_decision_point_report(self) -> Dict:
        """Generate decision point evaluation report"""
        
        # Calculate metrics for each decision point
        weather_results = self.results.get("weather_threshold", [])
        temporal_results = self.results.get("temporal_analysis", [])
        chat_results = self.results.get("chat_responses", [])
        
        weather_metrics = {
            "total_tests": len(weather_results),
            "threshold_accuracy_passed": sum(1 for r in weather_results if r.get("threshold_passed", False)),
            "impact_consistency_passed": sum(1 for r in weather_results if r.get("impact_passed", False)),
            "avg_threshold_accuracy": sum(r.get("threshold_accuracy", 0) for r in weather_results) / len(weather_results) if weather_results else 0,
            "avg_impact_consistency": sum(r.get("impact_consistency", 0) for r in weather_results) / len(weather_results) if weather_results else 0
        }
        
        temporal_metrics = {
            "total_tests": len(temporal_results),
            "temporal_recognition_passed": sum(1 for r in temporal_results if r.get("temporal_passed", False)),
            "avg_temporal_recognition": sum(r.get("temporal_recognition", 0) for r in temporal_results) / len(temporal_results) if temporal_results else 0
        }
        
        chat_metrics = {
            "total_tests": len(chat_results),
            "recommendation_quality_passed": sum(1 for r in chat_results if r.get("recommendation_passed", False)),
            "avg_recommendation_quality": sum(r.get("recommendation_quality", 0) for r in chat_results) / len(chat_results) if chat_results else 0
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "decision_point_summary": {
                "weather_threshold_decision": weather_metrics,
                "temporal_analysis_decision": temporal_metrics,
                "chat_response_decision": chat_metrics
            },
            "detailed_results": self.results,
            "overall_decision_reliability": self._calculate_overall_reliability(weather_metrics, temporal_metrics, chat_metrics)
        }
    
    def _calculate_overall_reliability(self, weather: Dict, temporal: Dict, chat: Dict) -> Dict:
        """Calculate overall decision reliability"""
        
        total_tests = weather["total_tests"] + temporal["total_tests"] + chat["total_tests"]
        total_passed = (weather["threshold_accuracy_passed"] + 
                       temporal["temporal_recognition_passed"] + 
                       chat["recommendation_quality_passed"])
        
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0
        
        return {
            "total_decision_tests": total_tests,
            "total_passed": total_passed,
            "overall_pass_rate": overall_pass_rate,
            "reliability_level": "High" if overall_pass_rate > 0.8 else "Medium" if overall_pass_rate > 0.6 else "Low",
            "critical_decision_points": {
                "weather_threshold_reliability": weather["avg_threshold_accuracy"],
                "temporal_analysis_reliability": temporal["avg_temporal_recognition"],
                "chat_response_reliability": chat["avg_recommendation_quality"]
            }
        }

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution for decision point evaluations"""
    print("🎯 Decision Point Specific Evaluations")
    print("=" * 60)
    
    evaluator = DecisionPointEvaluator()
    report = evaluator.run_all_decision_point_evaluations()
    
    print(f"\n✅ Decision Point Evaluation Complete!")
    print(f"📊 Overall Reliability: {report['overall_decision_reliability']['reliability_level']}")
    print(f"🎯 Pass Rate: {report['overall_decision_reliability']['overall_pass_rate']:.1%}")
    
    print(f"\n🔍 Critical Decision Points:")
    critical = report['overall_decision_reliability']['critical_decision_points']
    print(f"  🌤️  Weather Threshold: {critical['weather_threshold_reliability']:.2f}")
    print(f"  ⏰ Temporal Analysis: {critical['temporal_analysis_reliability']:.2f}")
    print(f"  💬 Chat Response: {critical['chat_response_reliability']:.2f}")
    
    print(f"\n📁 Results saved to: decision_point_evaluation_report.json")

if __name__ == "__main__":
    main()