#!/usr/bin/env python3
"""
Enhanced Cost Analysis and Projection System
Builds on existing LangSmith integration to provide detailed cost tracking and projections.
"""

import json
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class CostAnalyzer:
    """Enhanced cost tracking and projection system"""
    
    def __init__(self):
        self.usage_log_file = "enhanced_llm_usage.log"
        self.pricing_file = "pricing.json"
        self.load_pricing()
    
    def load_pricing(self):
        """Load pricing data"""
        try:
            with open(self.pricing_file, 'r') as f:
                self.pricing = json.load(f)
        except FileNotFoundError:
            logger.warning("pricing.json not found, using default pricing")
            self.pricing = {
                "claude-3-sonnet-20240229": {"input_cost_per_million": 3.0, "output_cost_per_million": 15.0},
                "gpt-4-turbo-preview": {"input_cost_per_million": 10.0, "output_cost_per_million": 30.0},
                "llama3": {"input_cost_per_million": 0.0, "output_cost_per_million": 0.0}  # Local model
            }
    
    def log_enhanced_usage(self, operation_data: Dict):
        """Log usage with enhanced context for cost analysis"""
        
        # Calculate cost if not provided
        if 'cost_usd' not in operation_data:
            operation_data['cost_usd'] = self.calculate_cost(
                operation_data.get('model_name', ''),
                operation_data.get('input_tokens', 0),
                operation_data.get('output_tokens', 0)
            )
        
        # Add timestamp if not provided
        if 'timestamp' not in operation_data:
            operation_data['timestamp'] = datetime.now().isoformat()
        
        # Write to enhanced log
        with open(self.usage_log_file, 'a') as f:
            f.write(json.dumps(operation_data) + '\n')
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given model and token usage"""
        if model not in self.pricing:
            return 0.0
        
        pricing = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * pricing.get('input_cost_per_million', 0)
        output_cost = (output_tokens / 1_000_000) * pricing.get('output_cost_per_million', 0)
        
        return input_cost + output_cost
    
    def get_usage_data(self, days: int = 30) -> pd.DataFrame:
        """Get usage data as DataFrame for analysis"""
        if not os.path.exists(self.usage_log_file):
            return pd.DataFrame()
        
        usage_data = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with open(self.usage_log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    log_date = datetime.fromisoformat(data['timestamp'])
                    if log_date >= cutoff_date:
                        usage_data.append(data)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        
        return pd.DataFrame(usage_data)
    
    def calculate_usage_scenarios(self) -> Dict:
        """Calculate cost projections for different usage scenarios"""
        
        # Get recent usage data to calculate averages
        df = self.get_usage_data(days=7)  # Last week
        
        if df.empty:
            return {"error": "No usage data available for projections"}
        
        # Calculate average costs per operation type
        operation_costs = {}
        for op_type in ['validation', 'report', 'chat']:
            op_data = df[df.get('operation_type', '') == op_type]
            if not op_data.empty:
                operation_costs[op_type] = {
                    'avg_cost': op_data['cost_usd'].mean(),
                    'avg_tokens': op_data['total_tokens'].mean(),
                    'avg_latency': op_data['duration_seconds'].mean()
                }
        
        # Define usage scenarios
        scenarios = {
            "light_user": {
                "description": "2-3 reports/day, 30 min chat",
                "daily_operations": {
                    "reports": 2.5,
                    "chat_sessions": 6,  # ~5 min per chat
                    "validations": 50   # Assuming 20 records per report
                }
            },
            "standard_user": {
                "description": "5-10 reports/day, 1-2 hours chat", 
                "daily_operations": {
                    "reports": 7.5,
                    "chat_sessions": 18,  # ~5 min per chat
                    "validations": 150
                }
            },
            "heavy_user": {
                "description": "15+ reports/day, 3+ hours chat",
                "daily_operations": {
                    "reports": 20,
                    "chat_sessions": 36,
                    "validations": 400
                }
            },
            "your_scenario": {
                "description": "10 reports/day, 2 hours chat",
                "daily_operations": {
                    "reports": 10,
                    "chat_sessions": 24,  # ~5 min per chat
                    "validations": 200
                }
            }
        }
        
        # Calculate costs for each scenario
        projections = {}
        
        for scenario_name, scenario in scenarios.items():
            daily_cost = 0
            operations_detail = {}
            
            for op_type, quantity in scenario["daily_operations"].items():
                # Map operation types
                cost_key = {
                    'reports': 'report',
                    'chat_sessions': 'chat', 
                    'validations': 'validation'
                }.get(op_type, op_type)
                
                if cost_key in operation_costs:
                    op_cost = operation_costs[cost_key]['avg_cost'] * quantity
                    daily_cost += op_cost
                    operations_detail[op_type] = {
                        'quantity': quantity,
                        'unit_cost': operation_costs[cost_key]['avg_cost'],
                        'total_cost': op_cost
                    }
                else:
                    # Fallback estimates if no data
                    fallback_costs = {
                        'reports': 0.15,
                        'chat_sessions': 0.05,
                        'validations': 0.02
                    }
                    op_cost = fallback_costs.get(op_type, 0.01) * quantity
                    daily_cost += op_cost
                    operations_detail[op_type] = {
                        'quantity': quantity,
                        'unit_cost': fallback_costs.get(op_type, 0.01),
                        'total_cost': op_cost,
                        'note': 'Estimated (no historical data)'
                    }
            
            projections[scenario_name] = {
                "description": scenario["description"],
                "daily_cost": daily_cost,
                "monthly_cost": daily_cost * 30,
                "yearly_cost": daily_cost * 365,
                "operations": operations_detail
            }
        
        return projections
    
    def compare_models_for_scenario(self, scenario_name: str = "your_scenario") -> Dict:
        """Compare costs across different models for a given scenario"""
        
        # Your specific scenario: 10 reports/day, 2 hours chat
        daily_ops = {
            "reports": 10,
            "chat_sessions": 24,
            "validations": 200
        }
        
        # Estimated token usage per operation (based on your app)
        token_estimates = {
            "reports": {"input": 2000, "output": 1500},      # Report generation
            "chat_sessions": {"input": 500, "output": 300},   # Chat interaction
            "validations": {"input": 300, "output": 150}      # Outage validation
        }
        
        model_comparisons = {}
        
        for model_name, pricing in self.pricing.items():
            total_daily_cost = 0
            model_breakdown = {}
            
            for op_type, quantity in daily_ops.items():
                tokens = token_estimates[op_type]
                unit_cost = self.calculate_cost(model_name, tokens["input"], tokens["output"])
                total_cost = unit_cost * quantity
                total_daily_cost += total_cost
                
                model_breakdown[op_type] = {
                    "quantity": quantity,
                    "unit_cost": unit_cost,
                    "total_cost": total_cost
                }
            
            model_comparisons[model_name] = {
                "daily_cost": total_daily_cost,
                "monthly_cost": total_daily_cost * 30,
                "yearly_cost": total_daily_cost * 365,
                "operations": model_breakdown
            }
        
        return model_comparisons
    
    def generate_cost_report(self) -> str:
        """Generate a comprehensive cost analysis report"""
        
        usage_scenarios = self.calculate_usage_scenarios()
        model_comparison = self.compare_models_for_scenario()
        
        report = []
        report.append("# 💰 Cost Analysis & Projections Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Usage Scenarios
        report.append("## 📊 Usage Scenario Projections\n")
        
        if "error" not in usage_scenarios:
            for scenario_name, data in usage_scenarios.items():
                report.append(f"### {scenario_name.replace('_', ' ').title()}")
                report.append(f"**{data['description']}**")
                report.append(f"- Daily Cost: ${data['daily_cost']:.4f}")
                report.append(f"- Monthly Cost: ${data['monthly_cost']:.2f}")
                report.append(f"- Yearly Cost: ${data['yearly_cost']:.2f}\n")
        else:
            report.append("⚠️ No historical data available. Run some analysis operations first.\n")
        
        # Model Comparison  
        report.append("## 🤖 Model Cost Comparison")
        report.append("Based on your scenario: 10 reports/day + 2 hours chat\n")
        
        for model_name, data in model_comparison.items():
            report.append(f"### {model_name}")
            report.append(f"- Daily: ${data['daily_cost']:.4f}")
            report.append(f"- Monthly: ${data['monthly_cost']:.2f}")
            report.append(f"- Yearly: ${data['yearly_cost']:.2f}\n")
        
        # Recommendations
        cheapest_model = min(model_comparison.keys(), 
                           key=lambda m: model_comparison[m]['monthly_cost'])
        most_expensive = max(model_comparison.keys(), 
                           key=lambda m: model_comparison[m]['monthly_cost'])
        
        savings = (model_comparison[most_expensive]['monthly_cost'] - 
                  model_comparison[cheapest_model]['monthly_cost'])
        
        report.append("## 💡 Recommendations\n")
        report.append(f"- **Most Cost-Effective**: {cheapest_model}")
        report.append(f"- **Most Expensive**: {most_expensive}")
        report.append(f"- **Potential Monthly Savings**: ${savings:.2f}")
        
        return "\n".join(report)

# Integration with existing LLMManager
def enhanced_usage_decorator(cost_analyzer: CostAnalyzer):
    """Decorator to add enhanced usage tracking to LLM calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            operation_type = kwargs.get('operation_type', 'unknown')
            
            try:
                result = func(*args, **kwargs)
                
                # Extract usage data from result if available
                usage_data = {
                    'operation_type': operation_type,
                    'timestamp': start_time.isoformat(),
                    'duration_seconds': (datetime.now() - start_time).total_seconds(),
                    'success': True
                }
                
                # Add token and cost data if available in result
                if hasattr(result, 'usage_metadata'):
                    usage_data.update({
                        'input_tokens': result.usage_metadata.get('input_tokens', 0),
                        'output_tokens': result.usage_metadata.get('output_tokens', 0),
                        'total_tokens': result.usage_metadata.get('total_tokens', 0)
                    })
                
                cost_analyzer.log_enhanced_usage(usage_data)
                return result
                
            except Exception as e:
                # Log failed operations too
                cost_analyzer.log_enhanced_usage({
                    'operation_type': operation_type,
                    'timestamp': start_time.isoformat(),
                    'duration_seconds': (datetime.now() - start_time).total_seconds(),
                    'success': False,
                    'error': str(e)
                })
                raise
        
        return wrapper
    return decorator