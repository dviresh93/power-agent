"""
Centralized Logging Configuration for Power Agent
- Controls log levels and verbosity
- Reduces excessive logging noise
- Configures proper formatting
- Environment-based configuration
"""

import os
import sys
import logging
from typing import Dict, List
from datetime import datetime


class PowerAgentLoggingConfig:
    """Centralized logging configuration for Power Agent system"""
    
    def __init__(self):
        self.log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        # Get log level from environment or default to WARNING to reduce noise
        self.default_level = os.getenv('LOG_LEVEL', 'WARNING').upper()
        self.verbose_mode = os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'
        
    def setup_logging(self):
        """Setup centralized logging configuration"""
        
        # Clear any existing handlers to avoid conflicts
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatter
        if self.verbose_mode:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(levelname)s:%(name)s:%(message)s'
            )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Set log level
        log_level = self.log_levels.get(self.default_level, logging.WARNING)
        
        # Configure root logger
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        
        # Configure specific loggers to reduce noise
        self._configure_specific_loggers()
        
        # Log the configuration
        logger = logging.getLogger(__name__)
        if self.verbose_mode:
            logger.info(f"✅ Logging configured: level={self.default_level}, verbose={self.verbose_mode}")
    
    def _configure_specific_loggers(self):
        """Configure specific loggers to reduce noise"""
        
        # Reduce HTTP request logging unless in debug mode
        if self.default_level != 'DEBUG':
            logging.getLogger('httpx').setLevel(logging.WARNING)
            logging.getLogger('requests').setLevel(logging.WARNING)
            logging.getLogger('urllib3').setLevel(logging.WARNING)
        
        # Control cost tracker verbosity
        cost_logger = logging.getLogger('services.langsmith_cost_tracker')
        if self.verbose_mode or self.default_level == 'DEBUG':
            cost_logger.setLevel(logging.INFO)
        else:
            cost_logger.setLevel(logging.WARNING)  # Only show warnings and errors
        
        # Control processing verbosity
        processing_loggers = [
            'new_agent',
            'chat_agent', 
            'report_agent',
            'agents.hybrid_processing_agent',
            'agents.hybrid_chat_agent',
            'agents.hybrid_report_agent'
        ]
        
        for logger_name in processing_loggers:
            logger = logging.getLogger(logger_name)
            if self.verbose_mode:
                logger.setLevel(logging.INFO)
            else:
                logger.setLevel(logging.WARNING)  # Only important messages
        
        # Vector DB and service loggers
        service_loggers = [
            'services.vector_db_service',
            'services.weather_service',
            'services.geocoding_service',
            'services.llm_service'
        ]
        
        for logger_name in service_loggers:
            logger = logging.getLogger(logger_name)
            if self.verbose_mode:
                logger.setLevel(logging.INFO)
            else:
                logger.setLevel(logging.WARNING)
    
    def enable_progress_logging(self):
        """Enable progress logging for processing operations"""
        progress_loggers = ['new_agent', 'agents.hybrid_processing_agent']
        for logger_name in progress_loggers:
            logging.getLogger(logger_name).setLevel(logging.INFO)
    
    def enable_cost_tracking_details(self):
        """Enable detailed cost tracking logs"""
        logging.getLogger('services.langsmith_cost_tracker').setLevel(logging.INFO)
    
    def set_quiet_mode(self):
        """Set quiet mode - only errors and critical"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.ERROR)
        
        # Still log critical processing updates
        logging.getLogger('new_agent').setLevel(logging.ERROR)


# Custom filter to reduce repetitive messages
class ReduceRepeatedMessagesFilter(logging.Filter):
    """Filter to reduce repeated log messages"""
    
    def __init__(self):
        super().__init__()
        self.recent_messages = {}
        self.repeat_threshold = 5  # After 5 identical messages, start reducing
    
    def filter(self, record):
        # Create a key from the log message
        key = f"{record.levelname}:{record.name}:{record.getMessage()}"
        
        # Track message frequency
        if key in self.recent_messages:
            self.recent_messages[key] += 1
            
            # Reduce frequency of repeated messages
            if self.recent_messages[key] > self.repeat_threshold:
                # Only show every 10th occurrence after threshold
                return self.recent_messages[key] % 10 == 0
        else:
            self.recent_messages[key] = 1
        
        return True


class ProgressLogger:
    """Specialized logger for showing progress without overwhelming output"""
    
    def __init__(self, logger_name: str = "progress"):
        self.logger = logging.getLogger(logger_name)
        self.last_progress_time = 0
        self.min_interval = 2.0  # Minimum seconds between progress updates
    
    def log_progress(self, current: int, total: int, message: str = "Processing"):
        """Log progress with rate limiting"""
        current_time = datetime.now().timestamp()
        
        # Only log progress every min_interval seconds or at milestones
        is_milestone = current % max(1, total // 10) == 0 or current == total
        time_elapsed = current_time - self.last_progress_time
        
        if is_milestone or time_elapsed >= self.min_interval:
            percentage = (current / total) * 100 if total > 0 else 0
            self.logger.info(f"🔄 {message}: {current}/{total} ({percentage:.1f}% complete)")
            self.last_progress_time = current_time


def setup_power_agent_logging():
    """Setup logging for the entire Power Agent system"""
    config = PowerAgentLoggingConfig()
    config.setup_logging()
    
    # Add filter to reduce repeated messages
    filter_obj = ReduceRepeatedMessagesFilter()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(filter_obj)
    
    return config


def get_progress_logger(name: str = "progress") -> ProgressLogger:
    """Get a progress logger instance"""
    return ProgressLogger(name)


# Environment variable configurations:
# LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: WARNING)
# VERBOSE_LOGGING: true/false (default: false)
# Examples:
# export LOG_LEVEL=INFO VERBOSE_LOGGING=true  # Detailed logging
# export LOG_LEVEL=ERROR                      # Quiet mode
# export LOG_LEVEL=WARNING                    # Default (balanced) 