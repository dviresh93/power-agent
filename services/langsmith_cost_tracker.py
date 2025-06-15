"""
LangSmith Cost Tracking Callback

This callback handler integrates with LangSmith to provide real-time cost tracking
for LLM operations. It calculates costs based on token usage and model pricing,
then adds this data to LangSmith traces for monitoring and analytics.

Features:
- Automatic cost calculation based on token usage
- Model-specific pricing support
- Real-time cost tracking in LangSmith traces
- Latency and performance metrics
"""

import json
import logging
import time
from uuid import UUID
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class LangSmithCostTracker(BaseCallbackHandler):
    """
    LangSmith-specific callback handler for cost and usage tracking.
    
    This callback automatically calculates costs and adds them to LangSmith traces
    so they appear in the LangSmith dashboard with real pricing information.
    """
    
    def __init__(self, pricing_file: str = "pricing.json"):
        """
        Initialize the cost tracker.
        
        Args:
            pricing_file: Path to the JSON file containing model pricing information
        """
        super().__init__()
        self.call_start_times: Dict[UUID, float] = {}
        self.pricing_map = self._load_pricing_map(pricing_file)
        
    def _load_pricing_map(self, pricing_file: str) -> Dict[str, Dict[str, float]]:
        """Load model pricing information from JSON file."""
        try:
            with open(pricing_file, "r") as f:
                pricing_data = json.load(f)
                logger.info(f"✅ Loaded pricing data for {len(pricing_data)} models")
                return pricing_data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"❌ Failed to load pricing data from {pricing_file}: {e}")
            return {}
    
    def _extract_model_name(self, serialized: Dict[str, Any]) -> str:
        """Extract model name from serialized LLM configuration."""
        # Try different ways to get the model name
        model_name = (
            serialized.get("model") or 
            serialized.get("model_name") or 
            serialized.get("_type", "unknown")
        )
        
        # Handle different model name formats
        if isinstance(model_name, str):
            # Handle cases like "models/claude-3-sonnet-20240229"
            if "/" in model_name:
                model_name = model_name.split("/")[-1]
            return model_name
        
        return "unknown"
    
    def _extract_token_usage(self, response: LLMResult) -> Dict[str, int]:
        """Extract token usage from LLM response."""
        usage_data = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        # Check if response has usage information
        if hasattr(response, 'llm_output') and response.llm_output:
            llm_output = response.llm_output
            
            # Handle different response formats
            if isinstance(llm_output, dict):
                # Anthropic/Claude format
                if "usage" in llm_output:
                    usage = llm_output["usage"]
                    usage_data["input_tokens"] = usage.get("input_tokens", 0)
                    usage_data["output_tokens"] = usage.get("output_tokens", 0)
                
                # OpenAI format
                elif "token_usage" in llm_output:
                    usage = llm_output["token_usage"]
                    usage_data["input_tokens"] = usage.get("prompt_tokens", 0)
                    usage_data["output_tokens"] = usage.get("completion_tokens", 0)
                
                # Alternative OpenAI format
                elif "prompt_tokens" in llm_output:
                    usage_data["input_tokens"] = llm_output.get("prompt_tokens", 0)
                    usage_data["output_tokens"] = llm_output.get("completion_tokens", 0)
        
        # Check response generations for token data (Ollama and others)
        if usage_data["total_tokens"] == 0 and response.generations:
            for generation in response.generations:
                if generation and generation[0].generation_info:
                    gen_info = generation[0].generation_info
                    
                    # Ollama format
                    if "prompt_eval_count" in gen_info:
                        usage_data["input_tokens"] = gen_info.get("prompt_eval_count", 0)
                        usage_data["output_tokens"] = gen_info.get("eval_count", 0)
                        break
                    
                    # Anthropic format (direct in generation_info)
                    elif "input_tokens" in gen_info and "output_tokens" in gen_info:
                        usage_data["input_tokens"] = gen_info.get("input_tokens", 0)
                        usage_data["output_tokens"] = gen_info.get("output_tokens", 0)
                        break
                    
                    # Other formats
                    elif "usage" in gen_info:
                        usage = gen_info["usage"]
                        usage_data["input_tokens"] = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
                        usage_data["output_tokens"] = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
                        break
        
        # Check direct response metadata (newer LangChain format)
        if usage_data["total_tokens"] == 0 and hasattr(response, 'response_metadata'):
            for gen in response.generations:
                if gen and hasattr(gen[0], 'response_metadata'):
                    metadata = gen[0].response_metadata
                    
                    # Ollama format in response metadata
                    if "prompt_eval_count" in metadata:
                        usage_data["input_tokens"] = metadata.get("prompt_eval_count", 0)
                        usage_data["output_tokens"] = metadata.get("eval_count", 0)
                        break
        
        # Calculate total tokens
        usage_data["total_tokens"] = usage_data["input_tokens"] + usage_data["output_tokens"]
        
        # If still no usage data found, try to estimate from response content
        if usage_data["total_tokens"] == 0 and response.generations:
            # Use tiktoken for accurate counting if available
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                total_text = "".join(gen[0].text for gen in response.generations if gen and gen[0].text)
                if total_text:  # Only encode if we have text
                    usage_data["output_tokens"] = len(encoding.encode(total_text))
                    usage_data["total_tokens"] = usage_data["output_tokens"]
                    logger.info(f"📊 Used tiktoken to count {usage_data['output_tokens']} output tokens")
                else:
                    # Fallback if no text
                    usage_data["output_tokens"] = 1
                    usage_data["total_tokens"] = 1
            except (ImportError, TypeError, AttributeError):
                # Fallback to character estimation
                total_chars = sum(len(gen[0].text) for gen in response.generations if gen)
                usage_data["output_tokens"] = max(1, total_chars // 4)
                usage_data["total_tokens"] = usage_data["output_tokens"]
                logger.warning(f"⚠️ No usage data found, estimated {usage_data['output_tokens']} output tokens")
        
        return usage_data
    
    def _calculate_cost(self, model_name: str, usage_data: Dict[str, int]) -> float:
        """Calculate cost based on model pricing and token usage."""
        if model_name not in self.pricing_map:
            logger.warning(f"⚠️ No pricing data for model: {model_name}")
            return 0.0
        
        pricing = self.pricing_map[model_name]
        input_cost_per_million = pricing.get("input_cost_per_million", 0.0)
        output_cost_per_million = pricing.get("output_cost_per_million", 0.0)
        
        # Calculate costs (pricing is per million tokens)
        input_cost = (usage_data["input_tokens"] / 1_000_000) * input_cost_per_million
        output_cost = (usage_data["output_tokens"] / 1_000_000) * output_cost_per_million
        total_cost = input_cost + output_cost
        
        logger.debug(f"💰 Cost calculation for {model_name}: "
                    f"input={usage_data['input_tokens']} tokens (${input_cost:.6f}), "
                    f"output={usage_data['output_tokens']} tokens (${output_cost:.6f}), "
                    f"total=${total_cost:.6f}")
        
        return total_cost
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, **kwargs: Any
    ) -> Any:
        """Record start time of LLM call."""
        self.call_start_times[run_id] = time.time()
        model_name = self._extract_model_name(serialized)
        logger.debug(f"🚀 LLM call started: {model_name} (run_id: {run_id})")
    
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> Any:
        """Calculate cost and add to LangSmith trace metadata."""
        end_time = time.time()
        start_time = self.call_start_times.pop(run_id, end_time)
        duration_seconds = end_time - start_time
        
        try:
            # Extract model name from multiple sources
            model_name = "unknown"
            
            # Try invocation_params first
            if "invocation_params" in kwargs:
                invocation_params = kwargs["invocation_params"]
                model_name = invocation_params.get("model", "unknown")
            
            # Try response.llm_output
            if model_name == "unknown" and hasattr(response, 'llm_output') and response.llm_output:
                model_name = response.llm_output.get("model", "unknown")
            
            # Try generation info (for Anthropic/Claude)
            if model_name == "unknown" and response.generations:
                for gen in response.generations:
                    if gen and gen[0].generation_info:
                        gen_info = gen[0].generation_info
                        model_name = gen_info.get("model", "unknown")
                        if model_name != "unknown":
                            break
            
            # Extract token usage
            usage_data = self._extract_token_usage(response)
            
            # Calculate cost
            cost = self._calculate_cost(model_name, usage_data)
            
            # Create metadata for LangSmith
            trace_metadata = {
                "usage": {
                    "input_tokens": usage_data["input_tokens"],
                    "output_tokens": usage_data["output_tokens"],
                    "total_tokens": usage_data["total_tokens"]
                },
                "cost": cost,
                "cost_breakdown": {
                    "input_cost": (usage_data["input_tokens"] / 1_000_000) * self.pricing_map.get(model_name, {}).get("input_cost_per_million", 0),
                    "output_cost": (usage_data["output_tokens"] / 1_000_000) * self.pricing_map.get(model_name, {}).get("output_cost_per_million", 0)
                },
                "model": model_name,
                "duration_seconds": duration_seconds,
                "pricing_source": "custom"
            }
            
            # Add metadata to the response for LangSmith to pick up
            if not hasattr(response, 'llm_output'):
                response.llm_output = {}
            if response.llm_output is None:
                response.llm_output = {}
            
            # Add our tracking data to the response
            response.llm_output.update({
                "langsmith_cost_tracking": trace_metadata
            })
            
            logger.info(f"💰 LLM call completed: {model_name} | "
                       f"Tokens: {usage_data['input_tokens']}→{usage_data['output_tokens']} | "
                       f"Cost: ${cost:.6f} | "
                       f"Duration: {duration_seconds:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error in cost tracking: {e}")
    
    def on_llm_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> Any:
        """Clean up on LLM error."""
        self.call_start_times.pop(run_id, None)
        logger.error(f"❌ LLM call failed (run_id: {run_id}): {error}")


def create_langsmith_cost_tracker(pricing_file: str = "pricing.json") -> LangSmithCostTracker:
    """
    Factory function to create a LangSmith cost tracker.
    
    Args:
        pricing_file: Path to the pricing configuration file
        
    Returns:
        Configured LangSmithCostTracker instance
    """
    return LangSmithCostTracker(pricing_file)