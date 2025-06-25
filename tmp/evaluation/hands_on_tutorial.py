#!/usr/bin/env python3
"""
Hands-On Tutorial: How to Evaluate Your LLM Agent
Step-by-step interactive guide
"""

import json
import os

def step_1_run_evaluation():
    """Step 1: Run your first evaluation"""
    print("🎯 STEP 1: RUN YOUR FIRST EVALUATION")
    print("=" * 50)
    
    print("\n💡 What we're going to do:")
    print("   1. Test your agent with 8 different scenarios")
    print("   2. Check if it makes good decisions")
    print("   3. See how well it explains its thinking")
    
    print(f"\n📝 Commands to run:")
    print(f"   python production_deepevals.py")
    print(f"   (This will take 1-2 minutes)")
    
    if os.path.exists('production_evaluation_report.json'):
        print(f"\n✅ Results file already exists!")
        print(f"   📁 File: production_evaluation_report.json")
    else:
        print(f"\n⏳ Run the command above first, then come back")

def step_2_view_simple_results():
    """Step 2: View results in simple terms"""
    print("\n🎯 STEP 2: VIEW YOUR RESULTS")
    print("=" * 50)
    
    print(f"\n📊 Command to see simple summary:")
    print(f"   python production_results_summary.py")
    
    if os.path.exists('production_evaluation_report.json'):
        try:
            with open('production_evaluation_report.json', 'r') as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            print(f"\n🎉 YOUR AGENT'S REPORT CARD:")
            print(f"   📊 Overall Score: {summary.get('pass_rate', 0):.1%}")
            
            weather = data.get('weather_validation', {})
            print(f"   🌤️  Weather Decisions: {weather.get('accuracy_rate', 0):.1%} accurate")
            print(f"   📝 Reasoning Quality: {weather.get('reasoning_rate', 0):.1%} good")
            
            chat = data.get('chat_responses', {})
            print(f"   💬 Chat Helpfulness: {chat.get('quality_rate', 0):.1%} helpful")
            
        except Exception as e:
            print(f"   ❌ Error reading results: {e}")

def step_3_understand_one_test():
    """Step 3: Understand one test in detail"""
    print("\n🎯 STEP 3: UNDERSTAND ONE TEST IN DETAIL")
    print("=" * 50)
    
    if os.path.exists('production_evaluation_report.json'):
        try:
            with open('production_evaluation_report.json', 'r') as f:
                data = json.load(f)
            
            weather_tests = data.get('detailed_results', {}).get('weather_validation', [])
            if weather_tests:
                test = weather_tests[0]  # Show first test
                
                print(f"\n🧪 LET'S LOOK AT ONE TEST:")
                print(f"   Test Name: {test['scenario_name']}")
                
                print(f"\n📥 INPUT (What we gave the agent):")
                weather = test['weather_conditions']
                for key, value in weather.items():
                    print(f"   {key}: {value}")
                
                print(f"\n🎯 EXPECTED ANSWER: {test['expected_classification']}")
                print(f"   Why? Weather is too mild to cause real outages")
                
                print(f"\n🤖 AGENT'S ANSWER:")
                response = test['actual_output']
                first_line = response.split('\n')[0]
                print(f"   {first_line}")
                
                print(f"\n📊 SCORES:")
                print(f"   Accuracy: {test['accuracy_score']:.1%} ({'✅ PASS' if test['accuracy_passed'] else '❌ FAIL'})")
                print(f"   Reasoning: {test['reasoning_score']:.1%} ({'✅ PASS' if test['reasoning_passed'] else '❌ FAIL'})")
                
                print(f"\n💭 WHAT THIS MEANS:")
                if test['accuracy_passed']:
                    print(f"   ✅ Your agent correctly identified this as a false positive!")
                    print(f"   ✅ It knows that mild weather doesn't cause outages")
                else:
                    print(f"   ❌ Your agent got this wrong - needs improvement")
                    
                if test['reasoning_passed']:
                    print(f"   ✅ Your agent explained its decision well")
                    print(f"   ✅ It mentioned specific weather conditions and thresholds")
                else:
                    print(f"   ❌ Your agent needs to explain its reasoning better")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"   ⏳ Run Step 1 first to generate results")

def step_4_create_your_own_test():
    """Step 4: Create your own test"""
    print(f"\n🎯 STEP 4: CREATE YOUR OWN TEST")
    print("=" * 50)
    
    print(f"\n🛠️  Let's create a simple test together!")
    print(f"\nHere's the pattern:")
    
    test_code = '''
# Your Test Scenario
weather_data = {
    "temperature": 25,    # Hot day
    "wind_speed": 5,      # Very light wind  
    "precipitation": 0,   # No rain
    "customers": 2        # Only 2 customers affected
}

# What should the agent decide?
# Hot weather (25°C) + light wind (5 km/h) + no rain = NOT severe enough
# Expected answer: FALSE POSITIVE

# Why?
# - Wind speed 5 km/h is way below 40 km/h threshold
# - No precipitation
# - Temperature is normal, not extreme
# - Only 2 customers affected
'''
    
    print(test_code)
    
    print(f"\n💡 TO ADD THIS TEST:")
    print(f"   1. Open: production_deepevals.py")
    print(f"   2. Find: _create_weather_test_scenarios()")
    print(f"   3. Add your scenario to the list")
    print(f"   4. Run: python production_deepevals.py")

def step_5_interpret_results():
    """Step 5: How to interpret results"""
    print(f"\n🎯 STEP 5: HOW TO INTERPRET RESULTS")
    print("=" * 50)
    
    print(f"\n📊 UNDERSTANDING SCORES:")
    print(f"\n🌤️  WEATHER DECISION ACCURACY:")
    print(f"   100% = Perfect! Agent always gets right answer")
    print(f"   80-99% = Very good, minor improvements needed")
    print(f"   60-79% = Okay, needs some work")
    print(f"   <60% = Needs significant improvement")
    
    print(f"\n📝 REASONING QUALITY:")
    print(f"   100% = Agent explains everything clearly")
    print(f"   80-99% = Good explanations, minor gaps")
    print(f"   60-79% = Explanations need improvement")
    print(f"   <60% = Agent barely explains its decisions")
    
    print(f"\n💬 CHAT HELPFULNESS:")
    print(f"   100% = Super helpful, answers all questions well")
    print(f"   80-99% = Very helpful")
    print(f"   60-79% = Somewhat helpful")
    print(f"   <60% = Not very helpful, needs work")

def step_6_what_to_do_next():
    """Step 6: What to do with results"""
    print(f"\n🎯 STEP 6: WHAT TO DO WITH RESULTS")
    print("=" * 50)
    
    print(f"\n🔧 IF YOUR AGENT SCORED LOW:")
    
    print(f"\n📉 Accuracy Problems (Agent gives wrong answers):")
    print(f"   1. Check prompts.json file")
    print(f"   2. Look at 'false_positive_detection' section")
    print(f"   3. Make sure weather thresholds are clear:")
    print(f"      - Wind > 25 mph (40 km/h)")
    print(f"      - Precipitation > 0.5 inches/hour")
    print(f"      - Temperature < 10°F or > 95°F")
    
    print(f"\n📝 Reasoning Problems (Agent doesn't explain well):")
    print(f"   1. Update system prompt in prompts.json")
    print(f"   2. Add phrases like:")
    print(f"      'Explain your reasoning with specific numbers'")
    print(f"      'Reference the weather thresholds in your analysis'")
    
    print(f"\n💬 Chat Problems (Agent not helpful):")
    print(f"   1. Improve 'chatbot_assistant' prompt")
    print(f"   2. Add instructions like:")
    print(f"      'Provide specific, actionable recommendations'")
    print(f"      'Include relevant statistics and data'")

def main():
    """Run the complete hands-on tutorial"""
    print("🎓 HANDS-ON TUTORIAL: EVALUATING YOUR LLM AGENT")
    print("=" * 60)
    print("👋 Welcome! This tutorial will teach you everything step-by-step")
    
    step_1_run_evaluation()
    step_2_view_simple_results()
    step_3_understand_one_test()
    step_4_create_your_own_test()
    step_5_interpret_results()
    step_6_what_to_do_next()
    
    print(f"\n🎉 CONGRATULATIONS!")
    print("=" * 60)
    print("🏆 You now know how to evaluate LLM agents!")
    print("🚀 You can:")
    print("   ✅ Run evaluations")
    print("   ✅ Understand the results")
    print("   ✅ Create your own tests")
    print("   ✅ Improve your agent based on results")
    
    print(f"\n📚 QUICK REFERENCE:")
    print(f"   🧪 Run tests: python production_deepevals.py")
    print(f"   📊 View results: python production_results_summary.py")
    print(f"   📖 This guide: python hands_on_tutorial.py")

if __name__ == "__main__":
    main()