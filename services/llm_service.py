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
        self.llm = self._initialize_llm()
        self.mcp_client = self._initialize_mcp() if MCP_AVAILABLE else None
        
    def _initialize_llm(self):
        """Initialize LLM with Claude as primary choice, OpenAI as fallback."""
        try:
            # Get configuration with defaults
            claude_model = self.model_config.get('claude_model', 'claude-3-sonnet-20240229')
            openai_model = self.model_config.get('openai_model', 'gpt-4-turbo-preview')
            temperature = self.model_config.get('temperature', 0.1)
            streaming = self.model_config.get('streaming', True)
            
            if os.getenv("ANTHROPIC_API_KEY"):
                logger.info("🤖 Using Anthropic Claude (recommended)")
                return ChatAnthropic(
                    model=claude_model,
                    temperature=temperature,
                    streaming=streaming
                )
            elif os.getenv("OPENAI_API_KEY"):
                logger.info("🤖 Using OpenAI GPT-4 (fallback)")
                return ChatOpenAI(
                    model=openai_model, 
                    temperature=temperature,
                    streaming=streaming
                )
            else:
                logger.error("❌ No LLM API keys found")
                raise ValueError(
                    "Please set ANTHROPIC_API_KEY (recommended) or OPENAI_API_KEY in your .env file"
                )
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
        llm_info = {
            'provider': 'anthropic' if os.getenv("ANTHROPIC_API_KEY") else 'openai',
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