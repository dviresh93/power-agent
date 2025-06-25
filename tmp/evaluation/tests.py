"""
DeepEval Test Suite for Power Outage Analysis Agent
- Tests LLM accuracy in outage validation
- Evaluates weather data retrieval
- Validates chat responses
- Tests end-to-end validation pipeline
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Dict, List, Any
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import DeepEval
from deepeval import evaluate
from deepeval.metrics import (
    HallucinationMetric,
    FactualConsistencyMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    BiasMetric,
    ToxicityMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.integrations.langchain_integration import load_langchain_run

# Import from main application
from main import (
    validate_outage_report,
    chat_about_results,
    WeatherService,
    LLMManager
)

# ==================== TEST DATA ====================
def load_test_data() -> pd.DataFrame:
    """Load sample test data"""
    try:
        return pd.read_csv("data/raw_data.csv")
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        # Create minimal test data if file not found
        return pd.DataFrame({
            'DATETIME': ['2022-01-01 00:00:00', '2022-01-01 01:00:00'],
            'LATITUDE': [40.198069, 41.591122],
            'LONGITUDE': [-88.262575, -88.440995],
            'CUSTOMERS': [1, 5]
        })

# ==================== TEST CASES ====================
class OutageValidationTestCase(LLMTestCase):
    """Test case for outage validation"""
    
    def __init__(self, outage_report: Dict, weather_data: Dict, expected_output: str, **kwargs):
        self.outage_report = outage_report
        self.weather_data = weather_data
        self.expected_output = expected_output
        
        # LLMTestCase requires these fields
        self.input = json.dumps(outage_report)
        self.actual_output = None
        self.expected_output = expected_output
        self.context = json.dumps(weather_data)
        
        super().__init__(**kwargs)
    
    def run(self):
        """Run the test case"""
        try:
            # Call the validation function
            self.actual_output = validate_outage_report.invoke({
                "outage_report": self.outage_report,
                "weather_data": self.weather_data
            })
            return self.actual_output
        except Exception as e:
            logger.error(f"Error running test case: {str(e)}")
            self.actual_output = f"ERROR: {str(e)}"
            return self.actual_output

class WeatherServiceTestCase(LLMTestCase):
    """Test case for weather service"""
    
    def __init__(self, lat: float, lon: float, date: datetime, **kwargs):
        self.lat = lat
        self.lon = lon
        self.date = date
        
        # LLMTestCase requires these fields
        self.input = f"Weather for {lat}, {lon} on {date}"
        self.actual_output = None
        self.expected_output = "Weather data retrieved successfully"
        self.context = None
        
        super().__init__(**kwargs)
    
    def run(self):
        """Run the test case"""
        try:
            weather_service = WeatherService()
            self.weather_data = weather_service.get_historical_weather(
                lat=self.lat,
                lon=self.lon,
                date=self.date
            )
            
            # Check if weather data is complete
            required_fields = ['temperature', 'precipitation', 'wind_speed', 'wind_gusts']
            missing_fields = [field for field in required_fields if field not in self.weather_data]
            
            if missing_fields:
                self.actual_output = f"Missing weather data fields: {', '.join(missing_fields)}"
            elif self.weather_data.get('api_status') == 'failed':
                self.actual_output = f"API failure: {self.weather_data.get('error', 'Unknown error')}"
            else:
                self.actual_output = "Weather data retrieved successfully"
                
            return self.actual_output
        except Exception as e:
            logger.error(f"Error running weather test: {str(e)}")
            self.actual_output = f"ERROR: {str(e)}"
            return self.actual_output

class ChatInterfaceTestCase(LLMTestCase):
    """Test case for chat interface"""
    
    def __init__(self, question: str, context: Dict, expected_topics: List[str], **kwargs):
        self.question = question
        self.context = context
        self.expected_topics = expected_topics
        
        # LLMTestCase requires these fields
        self.input = question
        self.actual_output = None
        self.expected_output = f"Response covering: {', '.join(expected_topics)}"
        self.context = json.dumps(context)
        
        super().__init__(**kwargs)
    
    def run(self):
        """Run the test case"""
        try:
            self.actual_output = chat_about_results.invoke({
                "question": self.question,
                "context": self.context
            })
            return self.actual_output
        except Exception as e:
            logger.error(f"Error running chat test: {str(e)}")
            self.actual_output = f"ERROR: {str(e)}"
            return self.actual_output

# ==================== TEST METRICS ====================
def run_outage_validation_tests():
    """Run tests for outage validation"""
    logger.info("Running outage validation tests...")
    
    # Create test cases with various weather conditions
    test_cases = []
    
    # Case 1: Clear false positive - mild weather
    test_cases.append(OutageValidationTestCase(
        outage_report={
            'datetime': '2022-01-01 00:00:00',
            'latitude': 40.198069,
            'longitude': -88.262575,
            'customers': 1
        },
        weather_data={
            'temperature': 15.5,  # Moderate temperature
            'precipitation': 0.0,  # No rain
            'wind_speed': 5.2,    # Light wind
            'wind_gusts': 8.3,    # Light gusts
            'snowfall': 0.0       # No snow
        },
        expected_output="FALSE POSITIVE"
    ))
    
    # Case 2: Clear real outage - severe weather
    test_cases.append(OutageValidationTestCase(
        outage_report={
            'datetime': '2022-01-01 01:45:00',
            'latitude': 38.822124,
            'longitude': -89.791293,
            'customers': 43
        },
        weather_data={
            'temperature': -5.2,   # Very cold
            'precipitation': 0.7,  # Heavy rain
            'wind_speed': 32.5,    # Strong wind
            'wind_gusts': 45.8,    # Strong gusts
            'snowfall': 3.2        # Heavy snow
        },
        expected_output="REAL OUTAGE"
    ))
    
    # Case 3: Borderline case - moderate conditions
    test_cases.append(OutageValidationTestCase(
        outage_report={
            'datetime': '2022-01-01 03:15:00',
            'latitude': 38.687404,
            'longitude': -89.609473,
            'customers': 40
        },
        weather_data={
            'temperature': 2.3,    # Cold but not extreme
            'precipitation': 0.3,  # Light rain
            'wind_speed': 22.5,    # Moderate wind
            'wind_gusts': 28.7,    # Moderate gusts
            'snowfall': 1.5        # Light snow
        },
        expected_output="BORDERLINE"  # This could go either way
    ))
    
    results = []
    for i, test_case in enumerate(test_cases):
        logger.info(f"Running test case {i+1}/{len(test_cases)}")
        output = test_case.run()
        
        # Evaluate hallucination
        hallucination_metric = HallucinationMetric(threshold=0.5)
        factual_consistency_metric = FactualConsistencyMetric(threshold=0.7)
        
        # Evaluate the response
        hallucination_metric.measure(test_case)
        factual_consistency_metric.measure(test_case)
        
        # Determine if the classification matches expected output
        expected_class = test_case.expected_output
        if expected_class == "BORDERLINE":
            # For borderline cases, either classification is acceptable
            classification_correct = True
        else:
            classification_correct = expected_class in output.upper()
        
        results.append({
            "case_id": i+1,
            "outage_report": test_case.outage_report,
            "weather_data": test_case.weather_data,
            "expected_class": expected_class,
            "llm_response": output,
            "classification_correct": classification_correct,
            "hallucination_score": hallucination_metric.score,
            "factual_consistency_score": factual_consistency_metric.score,
            "passed_hallucination": hallucination_metric.passed,
            "passed_factual": factual_consistency_metric.passed
        })
    
    return results

def run_weather_service_tests():
    """Run tests for weather service"""
    logger.info("Running weather service tests...")
    
    # Get test data
    df = load_test_data()
    
    # Create test cases
    test_cases = []
    for i, row in df.head(5).iterrows():  # Test first 5 records
        date_time = datetime.strptime(row['DATETIME'], "%Y-%m-%d %H:%M:%S")
        test_cases.append(WeatherServiceTestCase(
            lat=row['LATITUDE'],
            lon=row['LONGITUDE'],
            date=date_time
        ))
    
    results = []
    for i, test_case in enumerate(test_cases):
        logger.info(f"Running weather test case {i+1}/{len(test_cases)}")
        output = test_case.run()
        
        # Record results
        results.append({
            "case_id": i+1,
            "location": f"{test_case.lat}, {test_case.lon}",
            "date": test_case.date,
            "result": output,
            "success": output == "Weather data retrieved successfully"
        })
    
    return results

def run_chat_interface_tests():
    """Run tests for chat interface"""
    logger.info("Running chat interface tests...")
    
    # Sample validation results for context
    context = {
        "raw_summary": {
            "total_reports": 100,
            "date_range": {"start": "2022-01-01", "end": "2022-01-02"},
            "raw_customer_claims": {
                "total_claimed": 500,
                "avg_per_report": 5.0,
                "max_single_report": 44
            }
        },
        "validation_results": {
            "total_reports": 100,
            "real_outages": [
                {"datetime": "2022-01-01 01:45:00", "customers": 43, 
                 "latitude": 38.8, "longitude": -89.7,
                 "weather_data": {"temperature": -5.2, "wind_speed": 32.5}}
            ],
            "false_positives": [
                {"datetime": "2022-01-01 00:00:00", "customers": 1, 
                 "latitude": 40.1, "longitude": -88.2,
                 "weather_data": {"temperature": 15.5, "wind_speed": 5.2}}
            ],
            "statistics": {
                "real_count": 60,
                "false_positive_count": 40,
                "false_positive_rate": 40.0,
                "total_customers_actually_affected": 300,
                "total_customers_claimed": 500,
                "customer_impact_reduction": 200
            }
        }
    }
    
    test_cases = []
    
    # Question about false positives
    test_cases.append(ChatInterfaceTestCase(
        question="Why were these reports classified as false positives?",
        context=context,
        expected_topics=["weather conditions", "mild weather", "false positive criteria"]
    ))
    
    # Question about real outages
    test_cases.append(ChatInterfaceTestCase(
        question="What weather conditions caused the real outages?",
        context=context,
        expected_topics=["severe weather", "wind speed", "temperature"]
    ))
    
    # Question about statistical summary
    test_cases.append(ChatInterfaceTestCase(
        question="Provide a summary of the validation results.",
        context=context,
        expected_topics=["false positive rate", "customer impact", "validation statistics"]
    ))
    
    results = []
    for i, test_case in enumerate(test_cases):
        logger.info(f"Running chat test case {i+1}/{len(test_cases)}")
        output = test_case.run()
        
        # Evaluate relevancy of response
        relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
        contextual_metric = ContextualRelevancyMetric(threshold=0.7)
        bias_metric = BiasMetric(threshold=0.7)
        
        relevancy_metric.measure(test_case)
        contextual_metric.measure(test_case)
        bias_metric.measure(test_case)
        
        # Check if response covers expected topics
        topics_covered = all(topic.lower() in output.lower() for topic in test_case.expected_topics)
        
        results.append({
            "case_id": i+1,
            "question": test_case.question,
            "response": output[:200] + "..." if len(output) > 200 else output,  # Truncate for display
            "expected_topics": test_case.expected_topics,
            "topics_covered": topics_covered,
            "relevancy_score": relevancy_metric.score,
            "contextual_score": contextual_metric.score,
            "bias_score": bias_metric.score,
            "passed_relevancy": relevancy_metric.passed,
            "passed_contextual": contextual_metric.passed,
            "passed_bias": bias_metric.passed
        })
    
    return results

# ==================== TEST RUNNER AND REPORTING ====================
def generate_test_report(results: Dict[str, List[Dict]]) -> Dict:
    """Generate a comprehensive test report"""
    
    # Calculate overall metrics
    validation_results = results.get("validation_tests", [])
    weather_results = results.get("weather_tests", [])
    chat_results = results.get("chat_tests", [])
    
    # Validation metrics
    validation_metrics = {
        "total_tests": len(validation_results),
        "passed_classification": sum(1 for r in validation_results if r.get("classification_correct", False)),
        "passed_hallucination": sum(1 for r in validation_results if r.get("passed_hallucination", False)),
        "passed_factual": sum(1 for r in validation_results if r.get("passed_factual", False)),
        "avg_hallucination_score": np.mean([r.get("hallucination_score", 0) for r in validation_results]) if validation_results else 0,
        "avg_factual_score": np.mean([r.get("factual_consistency_score", 0) for r in validation_results]) if validation_results else 0
    }
    
    # Weather service metrics
    weather_metrics = {
        "total_tests": len(weather_results),
        "successful_requests": sum(1 for r in weather_results if r.get("success", False)),
        "success_rate": sum(1 for r in weather_results if r.get("success", False)) / len(weather_results) if weather_results else 0
    }
    
    # Chat interface metrics
    chat_metrics = {
        "total_tests": len(chat_results),
        "topics_covered": sum(1 for r in chat_results if r.get("topics_covered", False)),
        "passed_relevancy": sum(1 for r in chat_results if r.get("passed_relevancy", False)),
        "passed_contextual": sum(1 for r in chat_results if r.get("passed_contextual", False)),
        "passed_bias": sum(1 for r in chat_results if r.get("passed_bias", False)),
        "avg_relevancy_score": np.mean([r.get("relevancy_score", 0) for r in chat_results]) if chat_results else 0,
        "avg_contextual_score": np.mean([r.get("contextual_score", 0) for r in chat_results]) if chat_results else 0
    }
    
    # Overall metrics
    total_tests = validation_metrics["total_tests"] + weather_metrics["total_tests"] + chat_metrics["total_tests"]
    passed_tests = (validation_metrics["passed_classification"] + 
                   weather_metrics["successful_requests"] + 
                   chat_metrics["topics_covered"])
    
    overall_metrics = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "overall_pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
        "validation_pass_rate": validation_metrics["passed_classification"] / validation_metrics["total_tests"] if validation_metrics["total_tests"] > 0 else 0,
        "weather_service_pass_rate": weather_metrics["success_rate"],
        "chat_interface_pass_rate": chat_metrics["topics_covered"] / chat_metrics["total_tests"] if chat_metrics["total_tests"] > 0 else 0
    }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "overall": overall_metrics,
        "validation": validation_metrics,
        "weather_service": weather_metrics,
        "chat_interface": chat_metrics,
        "detailed_results": results
    }

def display_console_report(report: Dict):
    """Display a formatted report in the console"""
    print("\n" + "="*80)
    print(f"POWER OUTAGE ANALYSIS AGENT - DEEPEVAL TEST REPORT")
    print(f"Generated: {report['timestamp']}")
    print("="*80)
    
    # Overall results
    overall = report["overall"]
    print(f"\nOVERALL RESULTS: {overall['passed_tests']}/{overall['total_tests']} tests passed ({overall['overall_pass_rate']*100:.1f}%)")
    print(f"- Validation: {overall['validation_pass_rate']*100:.1f}% pass rate")
    print(f"- Weather Service: {overall['weather_service_pass_rate']*100:.1f}% pass rate")
    print(f"- Chat Interface: {overall['chat_interface_pass_rate']*100:.1f}% pass rate")
    
    # Validation results
    validation = report["validation"]
    print("\n" + "-"*80)
    print(f"OUTAGE VALIDATION TESTS: {validation['passed_classification']}/{validation['total_tests']} correct classifications")
    print(f"- Hallucination Metric: {validation['passed_hallucination']}/{validation['total_tests']} passed (avg score: {validation['avg_hallucination_score']:.2f})")
    print(f"- Factual Consistency: {validation['passed_factual']}/{validation['total_tests']} passed (avg score: {validation['avg_factual_score']:.2f})")
    
    # Sample validation results
    print("\nSample Validation Results:")
    for i, result in enumerate(report["detailed_results"]["validation_tests"][:2]):  # Show first 2
        print(f"  Test {result['case_id']}: {'✅' if result['classification_correct'] else '❌'} " + 
              f"Expected: {result['expected_class']}, " +
              f"Hallucination: {result['hallucination_score']:.2f}, " +
              f"Factual: {result['factual_consistency_score']:.2f}")
    
    # Weather service results
    weather = report["weather_service"]
    print("\n" + "-"*80)
    print(f"WEATHER SERVICE TESTS: {weather['successful_requests']}/{weather['total_tests']} successful requests ({weather['success_rate']*100:.1f}%)")
    
    # Sample weather results
    print("\nSample Weather Results:")
    for i, result in enumerate(report["detailed_results"]["weather_tests"][:2]):  # Show first 2
        print(f"  Test {result['case_id']}: {'✅' if result['success'] else '❌'} " + 
              f"Location: {result['location']}, Result: {result['result']}")
    
    # Chat interface results
    chat = report["chat_interface"]
    print("\n" + "-"*80)
    print(f"CHAT INTERFACE TESTS: {chat['topics_covered']}/{chat['total_tests']} covered expected topics")
    print(f"- Answer Relevancy: {chat['passed_relevancy']}/{chat['total_tests']} passed (avg score: {chat['avg_relevancy_score']:.2f})")
    print(f"- Contextual Relevancy: {chat['passed_contextual']}/{chat['total_tests']} passed (avg score: {chat['avg_contextual_score']:.2f})")
    
    # Sample chat results
    print("\nSample Chat Results:")
    for i, result in enumerate(report["detailed_results"]["chat_tests"][:2]):  # Show first 2
        print(f"  Test {result['case_id']}: {'✅' if result['topics_covered'] else '❌'} " + 
              f"Question: \"{result['question']}\"")
        print(f"    Topics: {', '.join(result['expected_topics'])}")
        print(f"    Relevancy: {result['relevancy_score']:.2f}, Contextual: {result['contextual_score']:.2f}")
    
    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80 + "\n")

def generate_streamlit_dashboard(report: Dict):
    """Generate code for a Streamlit dashboard to visualize test results"""
    dashboard_code = """
import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load test report
def load_report():
    try:
        with open('test_report.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading test report: {str(e)}")
        return None

# Main dashboard
def main():
    st.set_page_config(
        page_title="Power Agent DeepEval Dashboard",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🤖⚡ Power Outage Analysis - DeepEval Test Dashboard")
    
    report = load_report()
    if not report:
        st.warning("No test report found. Please run tests first.")
        return
    
    st.info(f"Test report generated: {report['timestamp']}")
    
    # Overall metrics
    st.header("📊 Overall Test Results")
    
    overall = report["overall"]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Pass Rate", 
            f"{overall['overall_pass_rate']*100:.1f}%",
            help=f"{overall['passed_tests']}/{overall['total_tests']} tests passed"
        )
    
    with col2:
        st.metric(
            "Validation Pass Rate", 
            f"{overall['validation_pass_rate']*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Weather Service", 
            f"{overall['weather_service_pass_rate']*100:.1f}%"
        )
    
    with col4:
        st.metric(
            "Chat Interface", 
            f"{overall['chat_interface_pass_rate']*100:.1f}%"
        )
    
    # Create tabs for detailed results
    tab1, tab2, tab3 = st.tabs(["Validation Tests", "Weather Service", "Chat Interface"])
    
    with tab1:
        st.subheader("🔍 Outage Validation Tests")
        
        validation = report["validation"]
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Classification Accuracy", f"{validation['passed_classification']}/{validation['total_tests']}")
        with col2:
            st.metric("Hallucination Score", f"{validation['avg_hallucination_score']:.2f}")
        with col3:
            st.metric("Factual Consistency", f"{validation['avg_factual_score']:.2f}")
        
        # Detailed results
        if "validation_tests" in report["detailed_results"]:
            df = pd.DataFrame(report["detailed_results"]["validation_tests"])
            st.dataframe(df[["case_id", "expected_class", "classification_correct", 
                            "hallucination_score", "factual_consistency_score"]])
            
            # Expandable section for LLM responses
            with st.expander("View LLM Responses"):
                for i, result in enumerate(report["detailed_results"]["validation_tests"]):
                    st.write(f"**Test {result['case_id']}:** {'✅' if result['classification_correct'] else '❌'}")
                    st.write(f"**Expected:** {result['expected_class']}")
                    st.text_area(f"LLM Response {i+1}", value=result['llm_response'], height=150, key=f"valid_{i}")
    
    with tab2:
        st.subheader("☁️ Weather Service Tests")
        
        weather = report["weather_service"]
        
        # Metrics
        st.metric("Success Rate", f"{weather['success_rate']*100:.1f}%", 
                 help=f"{weather['successful_requests']}/{weather['total_tests']} successful requests")
        
        # Detailed results
        if "weather_tests" in report["detailed_results"]:
            df = pd.DataFrame(report["detailed_results"]["weather_tests"])
            st.dataframe(df[["case_id", "location", "date", "result", "success"]])
    
    with tab3:
        st.subheader("💬 Chat Interface Tests")
        
        chat = report["chat_interface"]
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Topics Covered", f"{chat['topics_covered']}/{chat['total_tests']}")
        with col2:
            st.metric("Relevancy Score", f"{chat['avg_relevancy_score']:.2f}")
        with col3:
            st.metric("Contextual Score", f"{chat['avg_contextual_score']:.2f}")
        
        # Detailed results
        if "chat_tests" in report["detailed_results"]:
            df = pd.DataFrame(report["detailed_results"]["chat_tests"])
            st.dataframe(df[["case_id", "question", "topics_covered", 
                            "relevancy_score", "contextual_score", "bias_score"]])
            
            # Expandable section for chat responses
            with st.expander("View Chat Responses"):
                for i, result in enumerate(report["detailed_results"]["chat_tests"]):
                    st.write(f"**Question {i+1}:** {result['question']}")
                    st.write(f"**Expected Topics:** {', '.join(result['expected_topics'])}")
                    st.text_area(f"Response {i+1}", value=result['response'], height=150, key=f"chat_{i}")

if __name__ == "__main__":
    main()
"""
    
    # Save the dashboard code
    with open("dashboard.py", "w") as f:
        f.write(dashboard_code)
    
    return "dashboard.py created successfully"

def save_report_to_json(report: Dict):
    """Save the report to a JSON file"""
    try:
        with open("test_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        return "test_report.json created successfully"
    except Exception as e:
        logger.error(f"Error saving report to JSON: {str(e)}")
        return f"Error saving report: {str(e)}"

# ==================== MAIN FUNCTION ====================
def run_all_tests():
    """Run all tests and generate reports"""
    try:
        print("Starting DeepEval tests for Power Outage Analysis Agent...")
        
        # Run all test categories
        validation_results = run_outage_validation_tests()
        weather_results = run_weather_service_tests()
        chat_results = run_chat_interface_tests()
        
        # Collect all results
        all_results = {
            "validation_tests": validation_results,
            "weather_tests": weather_results,
            "chat_tests": chat_results
        }
        
        # Generate report
        report = generate_test_report(all_results)
        
        # Display console report
        display_console_report(report)
        
        # Save report to JSON
        json_result = save_report_to_json(report)
        print(f"JSON Report: {json_result}")
        
        # Generate Streamlit dashboard
        dashboard_result = generate_streamlit_dashboard(report)
        print(f"Dashboard: {dashboard_result}")
        
        print("\nTo view the dashboard, run: streamlit run dashboard.py")
        
        return report
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        print(f"Error running tests: {str(e)}")
        return None

if __name__ == "__main__":
    run_all_tests()