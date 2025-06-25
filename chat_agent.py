"""
Standalone Chat Agent - Reads .pkl files and answers questions about analysis results
Uses MCP calls for additional context when needed
"""

import os
import json
import logging
import joblib
from typing import Dict, Optional
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Import services for MCP calls
from services.llm_service import LLMManager
from services.weather_service import WeatherService
from services.geocoding_service import GeocodingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatAgent:
    """Standalone chat agent that reads .pkl files and provides Q&A about analysis results"""
    
    def __init__(self, pkl_file_path: str = None):
        """Initialize chat agent with .pkl file"""
        self.pkl_file_path = pkl_file_path
        self.results = None
        self.conversation_history = []
        
        # Initialize services for MCP calls
        self.llm_manager = LLMManager()
        self.weather_service = WeatherService()
        self.geocoding_service = GeocodingService()
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Load .pkl file if provided
        if pkl_file_path:
            self.load_results(pkl_file_path)
    
    def _load_prompts(self) -> Dict:
        """Load prompts from prompts.json"""
        try:
            with open('prompts.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("prompts.json file not found!")
            return {}
    
    def load_results(self, pkl_file_path: str) -> bool:
        """Load analysis results from .pkl file"""
        try:
            if not os.path.exists(pkl_file_path):
                logger.error(f"PKL file not found: {pkl_file_path}")
                return False
            
            self.results = joblib.load(pkl_file_path)
            self.pkl_file_path = pkl_file_path
            logger.info(f"✅ Loaded results from {pkl_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load .pkl file: {str(e)}")
            return False
    
    def get_context_summary(self) -> str:
        """Generate a summary of available context for the LLM"""
        if not self.results:
            return "No analysis results loaded"
        
        validation_results = self.results.get('validation_results', {})
        raw_summary = self.results.get('raw_dataset_summary', {})
        filtered_summary = self.results.get('filtered_summary', {})
        
        real_outages = len(validation_results.get('real_outages', []))
        false_positives = len(validation_results.get('false_positives', []))
        total_records = raw_summary.get('total_records', 0)
        
        return f"""
Analysis Context Available:
- Total Records Analyzed: {total_records}
- Real Outages Found: {real_outages}
- False Positives Found: {false_positives}
- Accuracy Rate: {filtered_summary.get('accuracy_rate', 0)*100:.1f}%
- Date Range: {raw_summary.get('date_range', {}).get('start', 'Unknown')} to {raw_summary.get('date_range', {}).get('end', 'Unknown')}
- Geographic Coverage: {raw_summary.get('geographic_bounds', {})}
"""
    
    def answer_question(self, question: str) -> str:
        """Answer questions about the analysis results using LLM + MCP calls"""
        if not self.results:
            return "Please load analysis results first using load_results(pkl_file_path)"
        
        try:
            # Use chatbot_assistant prompt from prompts.json
            if 'chatbot_assistant' in self.prompts:
                chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", self.prompts["chatbot_assistant"]["system"]),
                    ("human", self.prompts["chatbot_assistant"]["human"])
                ])
            else:
                # Fallback prompt
                chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant analyzing power outage data. Answer questions based on the provided analysis context."),
                    ("human", "User question: {user_question}\n\nAnalysis context: {analysis_context}")
                ])
            
            # Prepare context with full analysis results
            analysis_context = {
                "validation_results": self.results.get('validation_results', {}),
                "raw_summary": self.results.get('raw_dataset_summary', {}),
                "filtered_summary": self.results.get('filtered_summary', {}),
                "processing_time": self.results.get('processing_time', {}),
                "context_summary": self.get_context_summary()
            }
            
            # Get LLM response
            chain = chat_prompt | self.llm_manager.get_llm()
            response = chain.invoke({
                "user_question": question,
                "analysis_context": json.dumps(analysis_context, indent=2, default=str)
            })
            
            # Store conversation
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "response": response.content if hasattr(response, 'content') else str(response)
            })
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            error_msg = f"Chat error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_weather_context(self, latitude: float, longitude: float, datetime_str: str) -> Dict:
        """Get additional weather context using MCP calls"""
        try:
            return self.weather_service.get_historical_weather(latitude, longitude, datetime_str)
        except Exception as e:
            logger.error(f"Weather context error: {str(e)}")
            return {"error": str(e)}
    
    def get_location_context(self, latitude: float, longitude: float) -> Dict:
        """Get additional location context using MCP calls"""
        try:
            return self.geocoding_service.reverse_geocode(latitude, longitude)
        except Exception as e:
            logger.error(f"Location context error: {str(e)}")
            return {"error": str(e)}
    
    def get_conversation_history(self) -> list:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

# Helper function for UI integration
def create_chat_agent(pkl_file_path: str) -> ChatAgent:
    """Create a chat agent instance with loaded .pkl file"""
    return ChatAgent(pkl_file_path)

if __name__ == "__main__":
    # Test the chat agent
    import glob
    
    # Find the latest .pkl file
    pkl_files = glob.glob("cache/analysis_results_*.pkl")
    if pkl_files:
        latest_pkl = max(pkl_files, key=os.path.getctime)
        print(f"Testing chat agent with: {latest_pkl}")
        
        agent = ChatAgent(latest_pkl)
        print("Context:", agent.get_context_summary())
        
        # Test question
        response = agent.answer_question("What were the main factors that caused real outages vs false positives?")
        print("Response:", response)
    else:
        print("No .pkl files found in cache/ directory")