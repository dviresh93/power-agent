# Power Agent Logging Issues and Solutions

## What You Were Seeing

The verbose logging output you encountered shows several issues:

```
INFO:new_agent:🔄 Starting bulk validation with weather data...
INFO:new_agent:✅ Retrieved 100 records from vector database
INFO:httpx:HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
INFO:services.langsmith_cost_tracker:📊 Used tiktoken to count 344 output tokens
WARNING:services.langsmith_cost_tracker:⚠️ No pricing data for model: unknown
INFO:services.langsmith_cost_tracker:💰 LLM call completed: unknown | Tokens: 0→344 | Cost: $0.000000
INFO:new_agent:✅ Record 1/10: FALSE POSITIVE (confidence: 0.80)
[...repeats for every record...]
```

## Problems Identified

### 1. **Excessive Verbosity**
- Every step was logged at INFO level
- Each API call generated multiple log entries
- Individual record processing created noise
- HTTP requests from `httpx` were displayed

### 2. **Model Name Detection Issue**
- Cost tracker showed "unknown" instead of actual model
- Model name extraction was incomplete
- No mapping between model versions and pricing keys

### 3. **Repetitive Messages**
- Same messages repeated for every record
- No rate limiting on similar log entries
- Progress updates were too frequent

### 4. **Uncontrolled Log Levels**
- No centralized logging configuration
- Each module had its own `logging.basicConfig()`
- No way to adjust verbosity easily

## Solutions Implemented

### 1. **Centralized Logging Configuration** (`config/logging_config.py`)

```python
# Environment-based configuration
LOG_LEVEL=WARNING        # Default: Only warnings and errors
LOG_LEVEL=INFO          # Show progress and important info  
LOG_LEVEL=DEBUG         # Show everything
VERBOSE_LOGGING=true    # Detailed format with timestamps
```

### 2. **Controlled Logger Hierarchy**

```python
# HTTP requests - suppressed unless DEBUG
logging.getLogger('httpx').setLevel(logging.WARNING)

# Cost tracking - controlled verbosity
cost_logger = logging.getLogger('services.langsmith_cost_tracker')
if verbose_mode:
    cost_logger.setLevel(logging.INFO)
else:
    cost_logger.setLevel(logging.WARNING)  # Only warnings/errors

# Processing agents - reduced noise
processing_loggers = ['new_agent', 'chat_agent', 'report_agent']
for logger_name in processing_loggers:
    logger = logging.getLogger(logger_name)
    if verbose_mode:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
```

### 3. **Smart Progress Logging**

```python
# Before: Noisy per-record logging
logger.info(f"✅ Record {i}/10: FALSE POSITIVE (confidence: 0.80)")
logger.info(f"✅ Record {i+1}/10: FALSE POSITIVE (confidence: 0.80)")
# ... (repeats for every record)

# After: Rate-limited progress updates
progress_logger.log_progress(current, total, "reports validated")
# Output: "🔄 reports validated: 5/10 (50.0% complete)"
```

### 4. **Fixed Model Name Detection**

```python
def _extract_model_name(self, serialized: Dict[str, Any]) -> str:
    model_name = (
        serialized.get("model") or 
        serialized.get("model_name") or
        serialized.get("kwargs", {}).get("model") or    # NEW
        serialized.get("kwargs", {}).get("model_name") or  # NEW
        serialized.get("name") or                       # NEW
        serialized.get("_type", "unknown")
    )
    
    # Map model versions to pricing keys
    model_mappings = {
        "claude-3-sonnet-20240229": "claude-3-sonnet",
        "claude-3-haiku-20240307": "claude-3-haiku",
        "gpt-4-turbo-preview": "gpt-4-turbo",
        # ... more mappings
    }
    
    return model_mappings.get(model_name, model_name)
```

### 5. **Repeated Message Filter**

```python
class ReduceRepeatedMessagesFilter(logging.Filter):
    def filter(self, record):
        key = f"{record.levelname}:{record.name}:{record.getMessage()}"
        
        if key in self.recent_messages:
            self.recent_messages[key] += 1
            
            # After 5 identical messages, only show every 10th
            if self.recent_messages[key] > 5:
                return self.recent_messages[key] % 10 == 0
        else:
            self.recent_messages[key] = 1
        
        return True
```

## Before/After Comparison

### Before (Verbose & Noisy)
```
INFO:new_agent:🔄 Starting bulk validation with weather data...
INFO:httpx:HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
INFO:services.langsmith_cost_tracker:📊 Used tiktoken to count 344 output tokens
WARNING:services.langsmith_cost_tracker:⚠️ No pricing data for model: unknown
INFO:services.langsmith_cost_tracker:💰 LLM call completed: unknown | Tokens: 0→344 | Cost: $0.000000 | Duration: 7.94s
INFO:new_agent:✅ Record 1/10: FALSE POSITIVE (confidence: 0.80)
INFO:httpx:HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
INFO:services.langsmith_cost_tracker:📊 Used tiktoken to count 324 output tokens
WARNING:services.langsmith_cost_tracker:⚠️ No pricing data for model: unknown
INFO:services.langsmith_cost_tracker:💰 LLM call completed: unknown | Tokens: 0→324 | Cost: $0.000000 | Duration: 6.47s
INFO:new_agent:✅ Record 2/10: FALSE POSITIVE (confidence: 0.80)
[...repeats for every record...]
```

### After (Clean & Controlled)
```
INFO:progress:🔄 reports validated: 5/10 (50.0% complete)
INFO:progress:🔄 reports validated: 10/10 (100.0% complete)
INFO:new_agent:✅ Analysis complete: 2 real outages (20.0% accuracy), 8 false positives
```

## How to Control Logging

### Environment Variables

```bash
# Default quiet mode (only warnings and errors)
python simple_ui.py

# Show progress and important info
LOG_LEVEL=INFO python simple_ui.py

# Show everything with detailed format
LOG_LEVEL=INFO VERBOSE_LOGGING=true python simple_ui.py

# Debug mode (everything)
LOG_LEVEL=DEBUG VERBOSE_LOGGING=true python simple_ui.py

# Ultra quiet (only critical errors)
LOG_LEVEL=ERROR python simple_ui.py
```

### Programmatic Control

```python
from config.logging_config import PowerAgentLoggingConfig

config = PowerAgentLoggingConfig()
config.setup_logging()

# Enable specific features
config.enable_progress_logging()      # Show progress bars
config.enable_cost_tracking_details() # Show cost details
config.set_quiet_mode()              # Suppress most output
```

## Demo Script

Run the logging demo to see the different levels:

```bash
# Show default behavior
python simple_logging_demo.py

# Show before/after comparison
python simple_logging_demo.py compare

# Test different levels
LOG_LEVEL=INFO python simple_logging_demo.py
LOG_LEVEL=DEBUG VERBOSE_LOGGING=true python simple_logging_demo.py
```

## Summary

The logging issues you saw were caused by:
1. **No centralized configuration** - each module had its own settings
2. **Excessive verbosity** - everything logged at INFO level
3. **Model detection bugs** - cost tracker couldn't identify the model
4. **No rate limiting** - repetitive messages weren't controlled

The solutions provide:
1. **Environment-based control** - set log levels via environment variables
2. **Smart progress logging** - rate-limited, milestone-based updates
3. **Fixed model detection** - proper model name extraction and mapping
4. **Hierarchical configuration** - different verbosity for different components

Now you can control the logging verbosity based on your needs, from ultra-quiet to detailed debugging, while fixing the model name detection and cost tracking issues. 