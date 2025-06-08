#!/usr/bin/env python3
"""
Beginner's Guide to LLM Agent Evaluation
Shows exactly what each test does and why
"""

import json

def explain_evaluation_results():
    """Explain evaluation results in simple terms"""
    
    print("🎓 BEGINNER'S GUIDE TO LLM AGENT EVALUATION")
    print("=" * 60)
    
    print("\n🤔 WHAT ARE WE TESTING?")
    print("Think of your LLM agent like a smart assistant that:")
    print("1. Looks at weather data")
    print("2. Decides if power outages are real or fake") 
    print("3. Answers questions about its decisions")
    print("\nWe want to make sure it's doing a good job!")
    
    print("\n📊 THE 3 TYPES OF TESTS:")
    
    print("\n🌤️  TEST 1: WEATHER DECISION ACCURACY")
    print("   What: Give agent weather data + outage report")
    print("   Ask: 'Is this outage real or fake?'")
    print("   Check: Did it get the right answer?")
    print("\n   Example:")
    print("   Input: Light wind (8 km/h), 1 customer affected")
    print("   Expected: 'FALSE POSITIVE' (fake outage)")
    print("   Why: Weather too mild to cause real outages")
    
    print("\n📝 TEST 2: REASONING QUALITY") 
    print("   What: Look at HOW the agent explains its answer")
    print("   Check: Does it give good reasons?")
    print("\n   Good Response:")
    print("   'FALSE POSITIVE - Wind speed 8 km/h is below 40 km/h threshold...'")
    print("\n   Bad Response:")
    print("   'FALSE POSITIVE' (no explanation)")
    
    print("\n💬 TEST 3: CHAT HELPFULNESS")
    print("   What: Ask agent questions about its work")
    print("   Check: Are the answers helpful and accurate?")
    print("\n   Question: 'What wind speed threshold do you use?'")
    print("   Good Answer: 'The system uses 40 km/h as the wind speed threshold...'")
    print("   Bad Answer: 'I use thresholds.' (too vague)")

def show_real_test_example():
    """Show a real test example"""
    
    print("\n" + "=" * 60)
    print("🔬 REAL TEST EXAMPLE FROM YOUR AGENT")
    print("=" * 60)
    
    # Try to load actual results
    try:
        with open('production_evaluation_report.json', 'r') as f:
            data = json.load(f)
        
        weather_tests = data.get('detailed_results', {}).get('weather_validation', [])
        if weather_tests:
            test = weather_tests[0]  # First test
            
            print(f"\n📋 TEST SCENARIO: {test['scenario_name']}")
            print("\n📥 INPUT GIVEN TO AGENT:")
            weather = test['weather_conditions']
            print(f"   🌡️ Temperature: {weather['temperature']}°C")
            print(f"   💨 Wind Speed: {weather['wind_speed']} km/h")
            print(f"   🌧️ Precipitation: {weather['precipitation']} mm/h")
            
            print(f"\n🎯 EXPECTED ANSWER: {test['expected_classification']}")
            
            print(f"\n🤖 AGENT'S ACTUAL RESPONSE:")
            response = test['actual_output']
            print(f"   {response[:200]}...")
            
            print(f"\n📊 EVALUATION SCORES:")
            accuracy_icon = "✅" if test['accuracy_passed'] else "❌"
            reasoning_icon = "✅" if test['reasoning_passed'] else "❌"
            print(f"   {accuracy_icon} Accuracy Score: {test['accuracy_score']:.1%}")
            print(f"   {reasoning_icon} Reasoning Score: {test['reasoning_score']:.1%}")
            
            print(f"\n💭 WHAT THIS MEANS:")
            if test['accuracy_passed']:
                print("   ✅ Agent got the right answer!")
            else:
                print("   ❌ Agent got the wrong answer")
                
            if test['reasoning_passed']:
                print("   ✅ Agent explained its thinking well")
            else:
                print("   ❌ Agent needs to explain better")
    
    except FileNotFoundError:
        print("\n❌ No evaluation results found yet!")
        print("💡 Run this first: python production_deepevals.py")

def show_how_to_improve():
    """Show how to improve the agent"""
    
    print("\n" + "=" * 60)
    print("🛠️  HOW TO IMPROVE YOUR AGENT")
    print("=" * 60)
    
    print("\n🎯 IF ACCURACY IS LOW:")
    print("   Problem: Agent gives wrong answers")
    print("   Solution: Check prompts.json weather thresholds")
    print("   Example: Make sure wind threshold is clear (40 km/h)")
    
    print("\n📝 IF REASONING IS LOW:")
    print("   Problem: Agent doesn't explain well")
    print("   Solution: Update system prompt to require detailed explanations")
    print("   Example: Add 'Explain your reasoning with specific numbers'")
    
    print("\n💬 IF CHAT QUALITY IS LOW:")
    print("   Problem: Agent gives unhelpful answers")
    print("   Solution: Improve chatbot_assistant prompt in prompts.json")
    print("   Example: Add 'Provide specific, actionable recommendations'")

def show_next_steps():
    """Show what to do next"""
    
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS")
    print("=" * 60)
    
    print("\n📋 TO RUN MORE TESTS:")
    print("   1. Add more weather scenarios to production_deepevals.py")
    print("   2. Test edge cases (exactly at thresholds)")
    print("   3. Test with different customer counts")
    
    print("\n🔧 TO IMPROVE PERFORMANCE:")
    print("   1. Look at failed tests in detail")
    print("   2. Update prompts.json based on failures")
    print("   3. Re-run tests to see improvement")
    
    print("\n📈 TO EXPAND TESTING:")
    print("   1. Test all decision points from prompts.json")
    print("   2. Add temporal analysis tests")
    print("   3. Add integration tests")

def main():
    """Main guide"""
    explain_evaluation_results()
    show_real_test_example()
    show_how_to_improve()
    show_next_steps()
    
    print("\n" + "=" * 60)
    print("🎉 YOU'RE NOW READY TO EVALUATE LLM AGENTS!")
    print("💡 Start with: python production_deepevals.py")
    print("📊 View results: python production_results_summary.py")
    print("=" * 60)

if __name__ == "__main__":
    main()