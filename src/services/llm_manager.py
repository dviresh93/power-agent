from typing import Dict, Any
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LLMManager:
    def __init__(self):
        self._llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM with appropriate configuration"""
        self._llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def get_llm(self):
        """Get the initialized LLM instance"""
        return self._llm 