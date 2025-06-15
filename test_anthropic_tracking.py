#!/usr/bin/env python3
"""
Test Anthropic cost tracking specifically
"""

import os
from services.llm_service import create_llm_manager
from langchain_core.messages import HumanMessage

def test_anthropic_tracking():
    """Test cost tracking with Anthropic/Claude"""
    print("🧪 Testing Anthropic Cost Tracking")
    print("=" * 40)
    
    # Set to use Anthropic
    test_config = {
        'via': 'claude',  # Force Claude usage
        'claude_model': 'claude-3-haiku-20240307'  # Use cheapest model for testing
    }
    
    try:
        llm_manager = create_llm_manager(model_config=test_config)
        
        # Get provider info
        provider_info = llm_manager.get_provider_info()
        print(f"Provider: {provider_info['provider']}")
        print(f"Model: {provider_info['model']}")
        
        if provider_info['provider'] != 'anthropic':
            print("❌ Not using Anthropic - check API key")
            return
        
        # Make a test call
        message = HumanMessage(content="Say 'Hello world' in exactly 2 words.")
        print(f"\nSending test message...")
        
        response = llm_manager.invoke([message])
        print(f"Response: {response.content}")
        
        # Check response structure
        print(f"\nResponse analysis:")
        print(f"Type: {type(response)}")
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            print(f"usage_metadata: {response.usage_metadata}")
        
        if hasattr(response, 'response_metadata'):
            print(f"response_metadata keys: {list(response.response_metadata.keys())}")
            
        print("\n✅ Test completed - check llm_usage.log for results")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Check if API key is set
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("💡 ANTHROPIC_API_KEY not set - using fallback model")
        else:
            print("💡 Anthropic API key is set but call failed")

if __name__ == "__main__":
    test_anthropic_tracking()