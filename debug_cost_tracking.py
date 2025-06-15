#!/usr/bin/env python3
"""
Debug Cost Tracking Issues

This script helps debug why cost tracking isn't working by:
1. Testing LLM responses and token extraction
2. Checking callback integration
3. Verifying model pricing data
"""

import os
import json
from services.llm_service import create_llm_manager
from langchain_core.messages import HumanMessage


def test_llm_response_structure():
    """Test what the LLM response structure looks like."""
    print("🔍 Testing LLM Response Structure")
    print("=" * 40)
    
    try:
        # Create LLM manager
        llm_manager = create_llm_manager()
        
        # Get provider info
        provider_info = llm_manager.get_provider_info()
        print(f"Provider: {provider_info['provider']}")
        print(f"Model: {provider_info['model']}")
        
        # Make a simple test call
        message = HumanMessage(content="Hello, respond with just 'Hi'")
        print(f"\nSending test message: {message.content}")
        
        # Get response
        response = llm_manager.invoke([message])
        
        print(f"\nResponse type: {type(response)}")
        print(f"Response content: {response.content}")
        
        # Check response attributes
        print(f"\nResponse attributes:")
        for attr in dir(response):
            if not attr.startswith('_'):
                try:
                    value = getattr(response, attr)
                    if not callable(value):
                        print(f"  {attr}: {value}")
                except:
                    print(f"  {attr}: <error accessing>")
        
        # Check for usage data specifically
        print(f"\nChecking for usage data:")
        if hasattr(response, 'usage_metadata'):
            print(f"  usage_metadata: {response.usage_metadata}")
        if hasattr(response, 'response_metadata'):
            print(f"  response_metadata: {response.response_metadata}")
        if hasattr(response, 'additional_kwargs'):
            print(f"  additional_kwargs: {response.additional_kwargs}")
        
        return response
        
    except Exception as e:
        print(f"❌ Error testing LLM: {e}")
        return None


def test_pricing_data():
    """Test if pricing data is loaded correctly."""
    print("\n💰 Testing Pricing Data")
    print("=" * 25)
    
    try:
        with open("pricing.json", "r") as f:
            pricing_data = json.load(f)
        
        print(f"✅ Pricing data loaded: {len(pricing_data)} models")
        
        # Check for common models
        test_models = ["claude-3-sonnet-20240229", "gpt-4-turbo", "llama3"]
        for model in test_models:
            if model in pricing_data:
                pricing = pricing_data[model]
                print(f"  {model}: ${pricing['input_cost_per_million']}/1M input, ${pricing['output_cost_per_million']}/1M output")
            else:
                print(f"  {model}: ❌ Not found in pricing data")
        
        return pricing_data
        
    except Exception as e:
        print(f"❌ Error loading pricing data: {e}")
        return None


def test_callback_integration():
    """Test if callbacks are properly attached."""
    print("\n🔗 Testing Callback Integration") 
    print("=" * 30)
    
    try:
        llm_manager = create_llm_manager()
        llm = llm_manager.get_llm()
        
        print(f"LLM type: {type(llm)}")
        
        # Check if callbacks are attached
        if hasattr(llm, 'callbacks'):
            callbacks = llm.callbacks or []
            print(f"Number of callbacks: {len(callbacks)}")
            
            for i, callback in enumerate(callbacks):
                print(f"  Callback {i+1}: {type(callback).__name__}")
                
                # Check if it's our cost tracker
                if 'Cost' in type(callback).__name__ or 'Usage' in type(callback).__name__:
                    print(f"    ✅ Cost tracking callback found")
                    
                    # Check if pricing data is loaded
                    if hasattr(callback, 'pricing_map'):
                        print(f"    Pricing models loaded: {len(callback.pricing_map)}")
                    if hasattr(callback, '_pricing_map'):
                        print(f"    Pricing models loaded: {len(callback._pricing_map)}")
        else:
            print("❌ No callbacks attribute found")
        
        return llm
        
    except Exception as e:
        print(f"❌ Error testing callbacks: {e}")
        return None


def test_token_counting():
    """Test manual token counting methods."""
    print("\n🔢 Testing Token Counting")
    print("=" * 25)
    
    test_text = "Hello, this is a test message for token counting."
    
    # Method 1: Rough estimation (4 chars per token)
    char_estimate = len(test_text) // 4
    print(f"Text: '{test_text}'")
    print(f"Character count: {len(test_text)}")
    print(f"Estimated tokens (chars/4): {char_estimate}")
    
    # Method 2: Word-based estimation  
    word_estimate = len(test_text.split()) * 1.3  # ~1.3 tokens per word
    print(f"Word count: {len(test_text.split())}")
    print(f"Estimated tokens (words*1.3): {word_estimate:.1f}")
    
    try:
        # Method 3: Try tiktoken if available
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        actual_tokens = len(encoding.encode(test_text))
        print(f"Actual tokens (tiktoken): {actual_tokens}")
    except ImportError:
        print("tiktoken not available for precise counting")
    except Exception as e:
        print(f"Error with tiktoken: {e}")


def propose_fixes():
    """Propose fixes for cost tracking issues."""
    print("\n🔧 Proposed Fixes")
    print("=" * 15)
    
    fixes = [
        "1. Add manual token counting fallback",
        "2. Install tiktoken for accurate token counting", 
        "3. Add debug logging in callbacks",
        "4. Check LLM provider response format",
        "5. Verify callback execution order",
        "6. Test with different LLM providers"
    ]
    
    for fix in fixes:
        print(fix)


def main():
    """Main debugging function."""
    print("🐛 Cost Tracking Debug Tool")
    print("=" * 40)
    
    # Test each component
    response = test_llm_response_structure()
    pricing_data = test_pricing_data() 
    llm = test_callback_integration()
    test_token_counting()
    
    # Propose fixes
    propose_fixes()
    
    print("\n📋 Debug Summary:")
    print(f"  LLM Response: {'✅' if response else '❌'}")
    print(f"  Pricing Data: {'✅' if pricing_data else '❌'}")
    print(f"  Callbacks: {'✅' if llm else '❌'}")
    
    print("\n💡 Next Steps:")
    print("1. Check the debug output above")
    print("2. Fix any issues found")
    print("3. Test cost tracking again")


if __name__ == "__main__":
    main()