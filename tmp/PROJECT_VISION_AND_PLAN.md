# Power Agent Project Vision & Implementation Plan

## 🎯 **Project Vision**

Create a **simplified, single-click power outage analysis tool** that processes CSV data, correlates with weather conditions, and classifies outages as real or false positives using AI, with results displayed on an interactive map and accessible via chat interface.

## 📋 **Corrected Architecture Understanding**

### **Core Workflow - 3 Node LangGraph (Main Processing Pipeline):**
1. **Data Loading Node (Start Node)**
   - Load CSV file from `/data/raw_data.csv` ONCE
   - Insert data into vector database
   - Basic validation and preprocessing
   - This becomes the entry point of the system

2. **Processing Node** 
   - For each record with (lat, lon) coordinates:
     - Get customer outage count from data
     - Make MCP weather API call for that coordinate/time
     - Analyze if weather conditions (strong winds, storms, snow) could cause outages
     - Classify outages as:
       - **True Positives**: Weather-related outages (red markers)
       - **False Positives**: Non-weather-related outages (light blue markers)
   - **CRITICAL**: Save ALL classified results as `.pkl` file for persistence

3. **Output Node**
   - **ONLY** reads existing `.pkl` file (no processing)
   - Renders basic map and summary to UX
   - Provides final status and completion message

### **Separate Specialized Agents (Outside Main Workflow):**
4. **Chat Agent** (Independent)
   - Reads `.pkl` file as context
   - Uses MCP calls for additional weather/geographic data
   - Answers user questions about analysis results
   - Maintains conversation history

5. **Report Generation Agent** (Independent)  
   - Reads `.pkl` file as input
   - Makes relevant MCP calls for comprehensive data
   - Generates detailed reports with proper reasoning
   - Supports multiple formats (PDF, detailed analysis, etc.)

### **UI Requirements:**
- **Simplified Interface**: Single "Start Processing" button workflow
- **Smart .pkl Detection**: Check for existing `.pkl` files first, avoid reprocessing
- **Model Selection**: Side panel showing available models from:
  - .env file keys (Claude, OpenAI, etc.)
  - Llama models
- **Layout**: 
  - Left 1/4: All buttons and inputs
  - Right 3/4: Summary tab with analysis results
  - Bottom right: Chat interface (calls Chat Agent)

### **Key Workflow Logic:**
1. **UI Startup**: Check if `.pkl` files exist in cache/
2. **If .pkl exists**: Load results instantly, skip processing, enable chat
3. **If no .pkl**: Show "Start Processing" button 
4. **After Processing**: Auto-save `.pkl`, display results, enable chat
5. **Chat Interface**: Calls separate Chat Agent with `.pkl` context
6. **Report Generation**: Calls separate Report Agent with `.pkl` input

## 🏗️ **Revised Implementation Plan**

### **Phase 1: Architecture Correction** 🔄 **IN PROGRESS**
- [x] Reorganize codebase (new vs legacy)
- [x] Move unused files to tmp/ folder
- [x] Create centralized prompt management with prompts.json
- [x] Identify architecture misunderstanding 
- [x] Update vision document with correct architecture
- [ ] **Restructure Main Workflow (new_agent.py)**
  - Simplify to true 3-node pipeline ending with .pkl save
  - Remove complex processing from output node
  - Ensure clean separation of concerns

### **Phase 2: Separate Agent Creation** 📋 **PLANNED**
- [ ] **Chat Agent (chat_agent.py)**
  - Create standalone agent that reads .pkl files
  - Implement MCP call integration for additional context
  - Handle conversation history and context
  - No processing, only Q&A about existing results

- [ ] **Report Generation Agent (report_agent.py)**
  - Create standalone agent that reads .pkl files  
  - Implement MCP calls for comprehensive reporting
  - Generate multiple report formats with reasoning
  - No processing, only report creation from existing results

### **Phase 3: UI Integration** 📋 **PLANNED**
- [ ] **Smart .pkl Workflow in simple_ui.py**
  - Check for existing .pkl files on startup
  - Load results instantly if .pkl exists
  - Show "Start Processing" only if no .pkl
  - Auto-save .pkl after processing completion

- [ ] **Agent Integration**
  - Connect chat interface to Chat Agent
  - Connect report generation to Report Agent  
  - Ensure proper .pkl file passing between agents

### **Phase 4: Core Functionality Testing** 📋 **PLANNED**
- [ ] **End-to-End Testing**
  - Test complete 3-node workflow → .pkl save
  - Test .pkl persistence and loading
  - Test Chat Agent with saved .pkl files
  - Test Report Agent with saved .pkl files
  - Verify weather API integration works

- [ ] **Model Selection & Error Handling**
  - Dynamic model detection from .env
  - Model switching functionality
  - Graceful error handling for all agents

### **Phase 5: Polish & Enhancement** 📋 **FUTURE**
- [ ] **Performance Optimization**
  - Optimize processing speed
  - Add progress indicators
  - Implement better caching

- [ ] **Advanced Features**
  - Batch processing of multiple CSV files
  - Historical analysis comparison
  - Advanced filtering options

## 🎨 **UI Design Specification**

### **Layout Structure:**
```
┌─────────────────┬─────────────────────────────────────────┐
│ Left Sidebar    │ Main Summary Area                       │
│ (1/4 width)     │ (3/4 width)                            │
│                 │                                         │
│ 🤖 Model        │ 📊 Analysis Summary                     │
│ Selection       │ • Real Outages: XX                      │
│                 │ • False Positives: XX                   │
│ 📁 .pkl Status  │ • Accuracy: XX%                         │
│ ✅ 4 saved      │                                         │
│                 │ 🗺️ Interactive Map                      │
│ 🚀 START        │ [Map with red/blue markers]             │
│ PROCESSING      │                                         │
│                 │ 💬 Chat Interface                       │
│ 🔄 Reset        │ "Ask about the analysis..."             │
│                 │                                         │
└─────────────────┴─────────────────────────────────────────┘
```

### **User Journey:**
1. **Welcome State**: User sees explanation and START PROCESSING button
2. **Model Selection**: User can choose AI model from available options
3. **One-Click Process**: Click START PROCESSING → automatic workflow
4. **Results Display**: Summary metrics + map + chat become available
5. **Interactive Analysis**: User can ask questions via chat
6. **Persistence**: Results auto-saved, can be reloaded later

## 🛠️ **Technical Architecture**

### **Corrected File Structure:**
```
power-agent/
├── simple_ui.py              # Main Streamlit interface with .pkl detection
├── new_agent.py              # 3-node LangGraph workflow (ends with .pkl save)
├── chat_agent.py             # Standalone chat agent (reads .pkl + MCP calls)
├── report_agent.py           # Standalone report agent (reads .pkl + MCP calls)
├── prompts.json              # Centralized prompt management
├── services/                 # Backend services (weather, LLM, etc.)
├── config/                   # Configuration management
├── data/                     # CSV input files (raw_data.csv)
├── cache/                    # .pkl files and cached data (persistence layer)
├── legacy_*.py               # Legacy implementations
└── tmp/                      # Archived unused files
```

### **Key Components:**

#### **Main Processing Pipeline (new_agent.py):**
- **3-Node LangGraph**: Data Loading → Processing → Output (.pkl save)
- **Weather Integration**: MCP API calls for historical data
- **AI Classification**: LLM-based outage validation
- **Vector Database**: Efficient data storage and querying
- **Persistence**: Auto-save all results to .pkl files

#### **Specialized Agents:**
- **Chat Agent**: Reads .pkl + MCP calls for Q&A
- **Report Agent**: Reads .pkl + MCP calls for comprehensive reports
- **UI Controller**: Smart .pkl detection and agent orchestration

#### **Data Flow:**
```
CSV → [3-Node Workflow] → .pkl → [Chat Agent]
                             ↳ → [Report Agent]
                             ↳ → [UI Display]
```

## 📊 **Success Criteria**

### **Functional Requirements:**
- [x] Single-click processing workflow
- [x] Real-time progress indicators
- [x] Accurate weather-based classification
- [x] Interactive map with proper color coding
- [x] Working chat interface
- [ ] Reliable .pkl save/load functionality
- [ ] Dynamic model selection

### **UX Requirements:**
- [x] Clean, intuitive interface
- [x] No complex tab navigation
- [x] All controls in left sidebar
- [x] Clear visual feedback
- [ ] Fast response times (<30s for analysis)
- [ ] Graceful error handling

### **Technical Requirements:**
- [x] Centralized prompt management
- [x] Modular, maintainable code
- [x] Proper error logging
- [ ] Memory efficient processing
- [ ] Scalable to larger datasets

## 🚀 **Implementation Steps**

### **Current Implementation Tasks:**
1. **Restructure new_agent.py** → 3-node workflow ending with .pkl save
2. **Create chat_agent.py** → Standalone agent reading .pkl files  
3. **Create report_agent.py** → Standalone agent reading .pkl files
4. **Update simple_ui.py** → Smart .pkl detection, avoid reprocessing
5. **Remove chat/report** → From main workflow (belongs in separate agents)
6. **Test end-to-end** → Complete .pkl-based workflow

## 💡 **Key Innovation Points**

### **What Makes This Special:**
- **Weather Correlation**: AI-powered classification based on meteorological data
- **False Positive Detection**: Reduces noise in outage reporting systems
- **One-Click Simplicity**: Complex analysis made accessible
- **Modular Architecture**: Separate agents for processing, chat, and reporting
- **Persistent Analysis**: .pkl-based persistence avoids reprocessing
- **Smart .pkl Detection**: Instantly load existing results

### **Business Value:**
- **Operational Efficiency**: Quickly identify real vs false outage reports
- **Cost Reduction**: Avoid unnecessary field dispatches for false alarms
- **Data-Driven Insights**: Weather patterns inform infrastructure planning
- **User-Friendly**: Non-technical staff can run complex analyses

---

*This document serves as the master reference for the Power Agent project implementation. Updates should be made as requirements evolve or new features are added.*