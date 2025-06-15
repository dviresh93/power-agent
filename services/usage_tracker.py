import json
import logging
import time
from uuid import UUID
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

class LLMUsageTracker(BaseCallbackHandler):
    """Callback handler to track LLM usage, cost, and latency."""

    def __init__(self, usage_log_file: str = "llm_usage.log"):
        super().__init__()
        self.usage_log_file = usage_log_file
        self.call_start_times: Dict[UUID, float] = {}
        self._pricing_map = self._load_pricing_map()

    def _load_pricing_map(self) -> Dict[str, Dict[str, float]]:
        """Load model pricing information from a JSON file."""
        try:
            with open("pricing.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("pricing.json not found or invalid. Cost calculations will be disabled.")
            return {}

    def _log_usage(self, data: Dict[str, Any]):
        """Append usage data to the log file in JSONL format."""
        try:
            with open(self.usage_log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to usage log: {e}")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, **kwargs: Any
    ) -> Any:
        """Record start time of an LLM call."""
        self.call_start_times[run_id] = time.time()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> Any:
        """Calculate and log usage, cost, and latency on LLM end."""
        end_time = time.time()
        start_time = self.call_start_times.pop(run_id, end_time)
        duration_seconds = end_time - start_time

        try:
            # Handle cases where llm_output might be None (e.g., with streaming on some models)
            if not response.llm_output:
                logger.warning(f"No llm_output in response for run_id {run_id}. Attempting to find usage data in generation_info.")
                llm_output = {}
            else:
                llm_output = response.llm_output

            # Extract model name from multiple possible locations
            model_name = llm_output.get("model_name", "unknown_model")
            if model_name == "unknown_model" and response.generations and response.generations[-1]:
                gen_info = response.generations[-1][-1].generation_info
                if gen_info:
                    model_name = gen_info.get("model", "unknown_model")  # For Ollama

            # Extract token usage, checking multiple locations and key names
            token_usage = llm_output.get("token_usage", {})
            input_tokens = token_usage.get("input_tokens", token_usage.get("prompt_tokens", 0))
            output_tokens = token_usage.get("output_tokens", token_usage.get("completion_tokens", 0))

            # If usage is still zero, try generation_info (common with streaming)
            if (input_tokens == 0 and output_tokens == 0) and response.generations and response.generations[-1]:
                 gen_info = response.generations[-1][-1].generation_info
                 if gen_info and "token_usage" in gen_info:
                     token_usage = gen_info.get("token_usage", {})
                     input_tokens = token_usage.get("input_tokens", 0)
                     output_tokens = token_usage.get("output_tokens", 0)
                 # Fallback for Anthropic's specific keys
                 elif gen_info and 'input_token_count' in gen_info:
                     input_tokens = gen_info.get('input_token_count', 0)
                     output_tokens = gen_info.get('output_token_count', 0)
                 # Ollama format in generation_info
                 elif gen_info and 'prompt_eval_count' in gen_info:
                     input_tokens = gen_info.get('prompt_eval_count', 0)
                     output_tokens = gen_info.get('eval_count', 0)


            pricing = self._pricing_map.get(model_name, {})
            input_cost_per_million = pricing.get("input_cost_per_million", 0)
            output_cost_per_million = pricing.get("output_cost_per_million", 0)

            cost = (
                (input_tokens / 1_000_000) * input_cost_per_million
                + (output_tokens / 1_000_000) * output_cost_per_million
            )

            usage_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": str(run_id),
                "model_name": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": round(cost, 6),
                "duration_seconds": round(duration_seconds, 2),
            }
            self._log_usage(usage_data)

        except Exception as e:
            logger.error(f"Error in on_llm_end usage tracking: {e}", exc_info=True) 