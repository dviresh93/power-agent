"""
Hybrid Chat Agent - Uses Vector Database for Context Retrieval
- Queries vector DB semantically for relevant analysis results
- No longer loads entire .pkl files for each conversation
- Implements RAG (Retrieval-Augmented Generation) pattern
- Maintains conversation history and context
"""

import os
import json
import logging
from typing import Dict, Optional, List
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Import services
from services.llm_service import LLMManager
from services.weather_service import WeatherService
from services.geocoding_service import GeocodingService
from services.hybrid_vector_service import HybridOutageVectorDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridChatAgent:
    """
    Hybrid chat agent that uses vector database for context retrieval
    instead of loading entire .pkl files
    """
    
    def __init__(self, vector_db_path: str = "./chroma_db"):
        """Initialize hybrid chat agent with vector database"""
        self.vector_db = HybridOutageVectorDB(vector_db_path)
        self.conversation_history = []
        
        # Initialize services for MCP calls
        self.llm_manager = LLMManager()
        self.weather_service = WeatherService()
        self.geocoding_service = GeocodingService()
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        logger.info("✅ Hybrid Chat Agent initialized with vector database")
    
    def _load_prompts(self) -> Dict:
        """Load prompts from prompts.json"""
        try:
            with open('prompts.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("prompts.json file not found!")
            return {}
    
    def answer_question(self, question: str, context_limit: int = 5) -> str:
        """
        Answer questions using semantic search against vector database
        
        Args:
            question: User's question about the analysis
            context_limit: Maximum number of relevant results to retrieve
            
        Returns:
            str: LLM response based on retrieved context
        """
        try:
            # Get relevant context from vector database using semantic search
            context_data = self.vector_db.query_for_chat(question, n_results=context_limit)
            
            if context_data.get("error"):
                return f"Error retrieving context: {context_data['error']}"
            
            # Get analysis summary for general context
            analysis_summary = self.vector_db.get_analysis_summary()
            
            # Prepare context for LLM
            if context_data.get("context"):
                focused_context = context_data["context"]
                metadata_context = context_data.get("metadatas", [])
            else:
                focused_context = "No specific analysis results found relevant to your question."
                metadata_context = []
            
            # Use enhanced chatbot prompt
            if 'hybrid_chatbot_assistant' in self.prompts:
                system_prompt = self.prompts["hybrid_chatbot_assistant"]["system"]
                human_prompt = self.prompts["hybrid_chatbot_assistant"]["human"]
            elif 'chatbot_assistant' in self.prompts:
                system_prompt = self.prompts["chatbot_assistant"]["system"]
                human_prompt = self.prompts["chatbot_assistant"]["human"]
            else:
                # Fallback prompt for hybrid approach
                system_prompt = """You are an expert power grid operations assistant analyzing outage reports. 
                You have access to analysis results from a vector database that contains classifications of power outages as real or false positives, along with reasoning and weather context.
                
                Provide specific, actionable insights based on the retrieved analysis results. When discussing classifications, reference the confidence scores and reasoning provided. If asked about patterns, analyze the metadata provided.
                
                Be concise but comprehensive. Focus on actionable recommendations for power grid operations."""
                
                human_prompt = """User Question: {user_question}

Analysis Overview:
{analysis_summary}

Relevant Analysis Results:
{focused_context}

Please provide a helpful response based on this analysis data."""
            
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            # Format the context for the LLM
            formatted_summary = json.dumps(analysis_summary, indent=2, default=str)
            
            # Get LLM response
            chain = chat_prompt | self.llm_manager.get_llm()
            response = chain.invoke({
                "user_question": question,
                "analysis_summary": formatted_summary,
                "focused_context": focused_context,
                "results_count": context_data.get("total_results", 0)
            })
            
            # Store conversation with metadata
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "response": response.content if hasattr(response, 'content') else str(response),
                "context_results": context_data.get("total_results", 0),
                "analysis_summary": analysis_summary
            })
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            error_msg = f"Chat error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def ask_follow_up(self, follow_up_question: str, include_history: bool = True) -> str:
        """
        Ask a follow-up question with conversation context
        
        Args:
            follow_up_question: The follow-up question
            include_history: Whether to include recent conversation history
            
        Returns:
            str: LLM response with conversational context
        """
        try:
            # Build conversation context if requested
            conversation_context = ""
            if include_history and self.conversation_history:
                recent_conversations = self.conversation_history[-3:]  # Last 3 conversations
                conversation_context = "\n\nRecent Conversation Context:\n"
                for conv in recent_conversations:
                    conversation_context += f"Q: {conv['question']}\nA: {conv['response'][:200]}...\n\n"
            
            # Combine follow-up with conversation context
            enhanced_question = f"{follow_up_question}{conversation_context}"
            
            return self.answer_question(enhanced_question)
            
        except Exception as e:
            error_msg = f"Follow-up question error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_analysis_insights(self, insight_type: str = "general") -> str:
        """
        Get specific insights about the analysis
        
        Args:
            insight_type: Type of insights requested (general, false_positives, accuracy, etc.)
            
        Returns:
            str: Formatted insights based on analysis data
        """
        try:
            analysis_summary = self.vector_db.get_analysis_summary()
            
            if insight_type == "false_positives":
                false_positives = self.vector_db.query_by_classification("false_positive", limit=10)
                question = "What are the main patterns in false positive classifications and how can we reduce them?"
                
            elif insight_type == "accuracy":
                question = "How accurate is our classification system and what confidence levels do we see?"
                
            elif insight_type == "weather_correlation":
                question = "What weather conditions are most correlated with real outages versus false positives?"
                
            else:  # general
                question = "What are the key insights from this power outage analysis?"
            
            return self.answer_question(question)
            
        except Exception as e:
            error_msg = f"Insights error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def query_specific_results(self, classification: str = None, min_confidence: float = None) -> Dict:
        """
        Query specific analysis results with filters
        
        Args:
            classification: Filter by classification (real_outage, false_positive)
            min_confidence: Filter by minimum confidence score
            
        Returns:
            Dict: Filtered analysis results
        """
        try:
            if classification:
                results = self.vector_db.query_by_classification(classification)
            elif min_confidence:
                results = self.vector_db.query_by_confidence_range(min_confidence)
            else:
                # Get general summary
                return self.vector_db.get_analysis_summary()
            
            return {
                "results": results,
                "count": len(results),
                "filter_applied": f"classification={classification}" if classification else f"min_confidence={min_confidence}"
            }
            
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return {"error": str(e)}
    
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
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history with metadata"""
        return self.conversation_history
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_context_statistics(self) -> Dict:
        """Get statistics about available context in vector database"""
        try:
            analysis_summary = self.vector_db.get_analysis_summary()
            collection_stats = self.vector_db.get_collection_stats()
            
            return {
                "analysis_results": analysis_summary,
                "collection_stats": collection_stats,
                "context_availability": "full" if analysis_summary.get("total", 0) > 0 else "limited"
            }
        except Exception as e:
            logger.error(f"Context statistics error: {str(e)}")
            return {"error": str(e)}

# Helper function for UI integration
def create_hybrid_chat_agent(vector_db_path: str = "./chroma_db") -> HybridChatAgent:
    """Create a hybrid chat agent instance"""
    return HybridChatAgent(vector_db_path)

# Backward compatibility function
def create_chat_agent(pkl_file_path: str = None) -> HybridChatAgent:
    """
    Create chat agent - now uses vector DB instead of .pkl files
    Maintains backward compatibility but ignores pkl_file_path
    """
    logger.info(f"Note: pkl_file_path '{pkl_file_path}' ignored - using vector database instead")
    return HybridChatAgent()

if __name__ == "__main__":
    # Test the hybrid chat agent
    print("Testing Hybrid Chat Agent...")
    
    agent = HybridChatAgent()
    
    # Test context statistics
    stats = agent.get_context_statistics()
    print(f"Context Stats: {stats}")
    
    # Test questions
    test_questions = [
        "What are the main factors that caused real outages vs false positives?",
        "How can we improve our false positive detection?",
        "What weather conditions correlate most with real outages?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        response = agent.answer_question(question)
        print(f"A: {response[:200]}...")
    
    print("\n✅ Hybrid Chat Agent test completed") 