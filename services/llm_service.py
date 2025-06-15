"""
LLM Service Module

Provides a clean, reusable interface for LLM operations independent of UI frameworks.
Supports Claude (primary) and OpenAI (fallback) with optional MCP integration.

Features:
- Automatic LLM provider detection and initialization
- Clean error handling and logging
- MCP integration for enhanced tool capabilities
- Framework-agnostic design for reusability across different UIs
"""

import os
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING
from dotenv import load_dotenv

# LLM imports - Claude as default with OpenAI fallback
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama  # NEW: Ollama support

# LangSmith monitoring
from services.langsmith_service import LangSmithMonitor

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# MCP imports for 2025 best practices
if TYPE_CHECKING:
    from langchain_mcp_adapters.client import MultiServerMCPClient

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
    logger.info("MCP adapters available for enhanced tool integration")
except ImportError:
    MultiServerMCPClient = None
    MCP_AVAILABLE = False
    logger.warning("MCP adapters not available - install with: pip install langchain-mcp-adapters")


class LLMManager:
    """
    Enhanced LLM manager following 2025 patterns.
    
    Provides a clean interface for LLM operations with automatic provider detection,
    MCP integration support, and framework-agnostic design.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM manager.
        
        Args:
            model_config: Optional configuration for LLM models.
                         Can override default model names, temperature, etc.
        """
        self.model_config = model_config or {}
        self.langsmith_monitor = LangSmithMonitor()
        self.llm = self._initialize_llm()
        self.mcp_client = self._initialize_mcp() if MCP_AVAILABLE else None
        # Log provider/model info after initialization
        provider_info = self.get_provider_info()
        logger.info(f"LLMManager initialized with provider: {provider_info.get('provider')}, model: {provider_info.get('model')}, streaming: {provider_info.get('streaming')}, mcp: {provider_info.get('mcp_available')}, langsmith: {self.langsmith_monitor.is_enabled()}")
        
    def _initialize_llm(self):
        """Initialize LLM with Claude, OpenAI, or Ollama (Llama) as selected."""
        try:
            # Set up LangSmith environment variables if monitoring is enabled
            if self.langsmith_monitor.is_enabled():
                os.environ["LANGCHAIN_PROJECT"] = self.langsmith_monitor.project_name
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                logger.info(f"LangSmith tracing enabled for project: {self.langsmith_monitor.project_name}")
            
            # Get configuration with defaults from model_config, then .env, then hardcoded
            claude_model = self.model_config.get('claude_model', os.getenv('CLAUDE_MODEL', 'claude-3-sonnet-20240229'))
            openai_model = self.model_config.get('openai_model', os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview'))
            ollama_model = self.model_config.get('ollama_model', os.getenv('OLLAMA_MODEL', 'llama3'))
            temperature = self.model_config.get('temperature', float(os.getenv('LLM_TEMPERATURE', 0.1)))
            streaming = self.model_config.get('streaming', True)

            # Set up callbacks for cost and usage tracking
            callbacks = []
            
            # Add LangSmith cost tracking callback
            try:
                from services.langsmith_cost_tracker import create_langsmith_cost_tracker
                cost_tracker = create_langsmith_cost_tracker()
                callbacks.append(cost_tracker)
                logger.info("💰 LangSmith cost tracking enabled")
            except ImportError as e:
                logger.warning(f"⚠️ LangSmith cost tracking not available: {e}")
            
            # Also add the existing usage tracker for file logging
            try:
                from services.usage_tracker import LLMUsageTracker
                usage_tracker = LLMUsageTracker()
                callbacks.append(usage_tracker)
                logger.info("📊 File-based usage tracking enabled")
            except ImportError as e:
                logger.warning(f"⚠️ Usage tracker not available: {e}")

            # Determine provider: 1. model_config['via'], 2. os.getenv('LLM_PROVIDER')
            provider = self.model_config.get('via', os.getenv('LLM_PROVIDER'))

            # Explicit provider selection from config or .env
            if provider == 'ollama':
                logger.info(f"🤖 Using Llama via Ollama (provider selected): {ollama_model}")
                return ChatOllama(model=ollama_model, temperature=temperature, streaming=streaming, callbacks=callbacks)
            
            if provider == 'openai':
                logger.info(f"🤖 Using OpenAI GPT-4 (provider selected): {openai_model}")
                return ChatOpenAI(model=openai_model, temperature=temperature, streaming=streaming, callbacks=callbacks)

            if provider == 'claude':
                logger.info(f"🤖 Using Anthropic Claude (provider selected): {claude_model}")
                return ChatAnthropic(model=claude_model, temperature=temperature, streaming=streaming, callbacks=callbacks)

            # If no provider is selected, fallback to auto-detect based on API keys
            logger.info("No LLM provider selected. Auto-detecting based on environment...")
            if os.getenv("ANTHROPIC_API_KEY"):
                logger.info("🤖 Using Anthropic Claude (auto-detected from API key)")
                return ChatAnthropic(model=claude_model, temperature=temperature, streaming=streaming, callbacks=callbacks)
            
            if os.getenv("OPENAI_API_KEY"):
                logger.info("🤖 Using OpenAI GPT-4 (auto-detected from API key)")
                return ChatOpenAI(model=openai_model, temperature=temperature, streaming=streaming, callbacks=callbacks)
            
            # Final fallback to Ollama if no keys or provider are set
            try:
                logger.info(f"🤖 Trying Llama via Ollama as final fallback: {ollama_model}")
                return ChatOllama(model=ollama_model, temperature=temperature, streaming=streaming, callbacks=callbacks)
            except Exception as ollama_exc:
                logger.error(f"❌ Ollama fallback failed: {ollama_exc}")

            logger.error("❌ No LLM provider found. Please set LLM_PROVIDER in your .env or an API key.")
            raise ValueError("No LLM provider configured. Check .env or API keys.")
            
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {str(e)}")
            raise
    
    def _initialize_mcp(self) -> Optional[Any]:
        """Initialize MCP client if available and configured."""
        try:
            # MCP configuration can be passed via model_config
            mcp_config = self.model_config.get('mcp_config', {})
            
            if mcp_config:
                logger.info("🔗 Initializing MCP client")
                # Initialize MCP client with provided configuration
                return MultiServerMCPClient(**mcp_config)
            else:
                logger.info("🔗 MCP available but no configuration provided")
                return None
        except Exception as e:
            logger.warning(f"⚠️ MCP initialization failed: {str(e)}")
            return None
    
    def get_llm(self):
        """
        Get the initialized LLM instance.
        
        Returns:
            The LLM instance (ChatAnthropic or ChatOpenAI)
        """
        return self.llm
    
    def get_mcp_client(self) -> Optional[Any]:
        """
        Get the MCP client if available.
        
        Returns:
            MCP client instance or None if not available/configured
        """
        return self.mcp_client
    
    def is_mcp_available(self) -> bool:
        """Check if MCP integration is available and configured."""
        return MCP_AVAILABLE and self.mcp_client is not None
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current LLM provider and configuration.
        
        Returns:
            Dictionary containing provider information
        """
        # Correctly determine provider by checking the actual LLM instance type
        provider = "unknown"
        if isinstance(self.llm, ChatAnthropic):
            provider = "anthropic"
        elif isinstance(self.llm, ChatOpenAI):
            provider = "openai"
        elif isinstance(self.llm, ChatOllama):
            provider = "ollama"

        llm_info = {
            'provider': provider,
            'model': getattr(self.llm, 'model', 'unknown'),
            'temperature': getattr(self.llm, 'temperature', 'unknown'),
            'streaming': getattr(self.llm, 'streaming', False),
            'mcp_available': self.is_mcp_available()
        }
        return llm_info
    
    def create_chain(self, prompt_template):
        """
        Create a LangChain chain with the configured LLM.
        
        Args:
            prompt_template: The prompt template to use
            
        Returns:
            Configured chain ready for execution
        """
        return prompt_template | self.llm
    
    def invoke(self, messages, **kwargs):
        """
        Direct invoke method for the LLM.
        
        Args:
            messages: Messages to send to the LLM
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            LLM response
        """
        return self.llm.invoke(messages, **kwargs)
    
    async def ainvoke(self, messages, **kwargs):
        """
        Async invoke method for the LLM.
        
        Args:
            messages: Messages to send to the LLM
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            LLM response
        """
        return await self.llm.ainvoke(messages, **kwargs)
    
    def stream(self, messages, **kwargs):
        """
        Stream method for the LLM.
        
        Args:
            messages: Messages to send to the LLM
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            Stream of LLM responses
        """
        return self.llm.stream(messages, **kwargs)
    
    async def astream(self, messages, **kwargs):
        """
        Async stream method for the LLM.
        
        Args:
            messages: Messages to send to the LLM
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            Async stream of LLM responses
        """
        return self.llm.astream(messages, **kwargs)
    
    def get_monitoring_data(self, days: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive monitoring data from LangSmith.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary containing usage metrics and model breakdown
        """
        if not self.langsmith_monitor.is_enabled():
            return {"error": "LangSmith monitoring not enabled"}
        
        return {
            "usage_metrics": self.langsmith_monitor.get_usage_metrics(days),
            "model_breakdown": self.langsmith_monitor.get_model_breakdown(days),
            "project_info": self.langsmith_monitor.get_project_info()
        }
    
    def export_monitoring_report(self, days: int = 30) -> str:
        """
        Export comprehensive monitoring report.
        
        Args:
            days: Number of days to include in report
            
        Returns:
            Path to exported report file
        """
        return self.langsmith_monitor.export_analytics_data(days)
    
    def get_langsmith_setup_instructions(self) -> str:
        """
        Get setup instructions for LangSmith.
        
        Returns:
            Setup instructions string
        """
        return self.langsmith_monitor.setup_environment_variables()


def create_llm_manager(model_config: Optional[Dict[str, Any]] = None) -> LLMManager:
    """
    Factory function to create an LLM manager instance.
    
    Args:
        model_config: Optional configuration for LLM models
        
    Returns:
        Configured LLMManager instance
    """
    return LLMManager(model_config)


def get_available_providers() -> Dict[str, bool]:
    """
    Check which LLM providers are available based on environment variables.
    
    Returns:
        Dictionary indicating availability of each provider
    """
    return {
        'anthropic': bool(os.getenv("ANTHROPIC_API_KEY")),
        'openai': bool(os.getenv("OPENAI_API_KEY")),
        'mcp': MCP_AVAILABLE
    }


def validate_environment() -> tuple[bool, list[str]]:
    """
    Validate that the environment is properly configured for LLM operations.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        issues.append("No LLM API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
    
    # Check for any other validation requirements here
    
    return len(issues) == 0, issues