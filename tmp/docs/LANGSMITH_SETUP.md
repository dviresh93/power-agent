# LangSmith Setup Guide

This guide will help you set up LangSmith monitoring for your power-agent application.

## Step 1: Get Your LangSmith API Key

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Sign up or log in to your account
3. Navigate to Settings → API Keys
4. Create a new API key and copy it

## Step 2: Configure Environment Variables

Add these variables to your `.env` file:

```bash
# LangSmith Configuration
LANGSMITH_API_KEY=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
LANGCHAIN_PROJECT=power-agent
LANGCHAIN_TRACING_V2=true
```

## Step 3: Test the Integration

Run the test script to verify everything is working:

```bash
python langsmith_test.py
```

This will:
- ✅ Check your configuration
- 🧪 Make sample LLM calls with tracing
- 📊 Display monitoring data
- 📁 Generate a comprehensive report

## Step 4: Understanding the Data

### What LangSmith Tracks

1. **Cost Analysis**
   - Total API costs
   - Cost per model
   - Cost per request
   - Daily/weekly/monthly projections

2. **Performance Metrics**
   - Response latency (average, p95)
   - Throughput (requests per day)
   - Error rates
   - Success rates

3. **Token Usage**
   - Input tokens
   - Output tokens
   - Tokens per request
   - Token efficiency

4. **Model Comparison**
   - Performance by model
   - Cost effectiveness
   - Usage patterns

### Accessing Your Data

**Web Dashboard (Recommended)**
- Visit [https://smith.langchain.com](https://smith.langchain.com)
- Select your "power-agent" project
- View real-time analytics and traces

**Programmatic Access**
```python
from services.llm_service import create_llm_manager

llm_manager = create_llm_manager()
data = llm_manager.get_monitoring_data(days=7)
print(data)
```

## Step 5: Enhanced Reporting

Use the enhanced reporting script to combine DeepEval results with LangSmith data:

```bash
python enhanced_production_reporting.py
```

This provides:
- Combined performance analysis
- Cost efficiency metrics
- Model comparison
- Actionable recommendations

## Step 6: Production Monitoring

### Daily Monitoring
```bash
# Check yesterday's performance
python -c "
from services.llm_service import create_llm_manager
llm = create_llm_manager()
data = llm.get_monitoring_data(1)
print('Yesterday Cost:', data['usage_metrics']['estimated_cost_usd'])
print('Error Rate:', data['usage_metrics']['error_rate_percent'], '%')
"
```

### Weekly Reports
```bash
# Generate weekly report
python enhanced_production_reporting.py production_evaluation_report.json 7
```

### Cost Alerts
Set up monitoring to alert when costs exceed thresholds:
```python
from services.llm_service import create_llm_manager

llm = create_llm_manager()
data = llm.get_monitoring_data(1)
daily_cost = data['usage_metrics']['estimated_cost_usd']

if daily_cost > 5.0:  # Alert if over $5/day
    print(f"⚠️  High daily cost: ${daily_cost:.2f}")
```

## Common Use Cases

### 1. Model Selection
Compare different models to find the best cost/performance ratio:
```python
data = llm_manager.get_monitoring_data(7)
models = data['model_breakdown']
for model, stats in models.items():
    print(f"{model}: ${stats['cost_per_1k_tokens']:.4f} per 1K tokens")
```

### 2. Performance Optimization
Identify slow requests and optimize:
- Check p95 latency
- Analyze failed requests
- Review token usage patterns

### 3. Cost Management
- Set daily/weekly budgets
- Monitor token usage trends
- Compare model costs

### 4. Quality Assurance
- Track error rates
- Monitor response quality
- Identify performance degradation

## Integration with Your Workflow

### Automated Reporting
Add to your CI/CD pipeline:
```bash
# In your deployment script
python enhanced_production_reporting.py
```

### Real-time Monitoring
```python
# In your application
def monitor_llm_call():
    data = llm_manager.get_monitoring_data(1)
    if data['usage_metrics']['error_rate_percent'] > 10:
        send_alert("High LLM error rate detected")
```

## Troubleshooting

### Common Issues

1. **API Key Not Working**
   - Verify key is correct
   - Check for leading/trailing spaces
   - Ensure key has proper permissions

2. **No Data Showing**
   - Wait 5-10 minutes after making calls
   - Check project name matches
   - Verify tracing is enabled

3. **Import Errors**
   - Ensure `langsmith` package is installed
   - Check Python path
   - Verify all dependencies

### Getting Help

- [LangSmith Documentation](https://docs.langchain.com/docs/langsmith)
- [Community Forum](https://community.langchain.com)
- [GitHub Issues](https://github.com/langchain-ai/langsmith-sdk)

## Next Steps

1. Set up your API key
2. Run the test script
3. Make some LLM calls in your application
4. View the data in the web dashboard
5. Set up automated reporting
6. Configure cost alerts

Happy monitoring! 📊