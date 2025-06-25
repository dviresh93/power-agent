# 🔌 Power Outage Analysis System

A streamlined AI-powered system for analyzing power outage reports and identifying false positives using weather correlation.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run simple_ui.py
```

## ✨ Features

- **🗺️ Enhanced Map Visualization** - Large, auto-fitting maps with detailed outage information
- **🤖 AI Classification** - Distinguishes real outages from false positives using weather data
- **💬 Sidebar Chat** - Ask questions about analysis results directly in the sidebar
- **📊 One-Click Reports** - Generate comprehensive analysis reports instantly
- **⚡ Fast Processing** - Streamlined 3-node LangGraph workflow

## 📁 Project Structure

```
power-agent/
├── simple_ui.py          # Main Streamlit application
├── new_agent.py          # Core analysis engine (LangGraph)
├── chat_agent.py         # Chat functionality
├── report_agent.py       # Report generation
├── cost_analyzer.py      # Cost tracking
├── prompts.json          # AI prompts
├── services/             # Service modules
├── config/               # Configuration
├── data/                 # Data files
├── legacy/               # Legacy implementations
└── tmp/                  # Documentation & unused files
```

## 🎯 Core Workflow

1. **Load Data** - CSV with DATETIME, LATITUDE, LONGITUDE, CUSTOMERS
2. **Weather Analysis** - Fetch historical weather for each outage
3. **AI Classification** - Determine real vs false positive outages
4. **Interactive Results** - Enhanced map + chat interface

## 🔧 Configuration

- Set API keys in `.env` file
- Configure LLM providers in `services/llm_service.py`
- Adjust prompts in `prompts.json`

## 📊 Technology Stack

- **Frontend**: Streamlit + Folium maps
- **AI**: LangGraph + LangChain + Claude/GPT
- **Data**: ChromaDB + Pandas + SQLite
- **Weather**: Open-Meteo API
- **Monitoring**: LangSmith + Cost Tracking

---

*Clean, focused, and production-ready power outage analysis.* 