"""
Simplified Power Outage Analysis UI
Clean, focused interface with left sidebar controls and main summary area
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# Configure logging before importing other modules
from config.logging_config import setup_power_agent_logging
setup_power_agent_logging()

# Now import the agents (they will use the configured logging)
from new_agent import graph, run_analysis, services, PROMPTS
from chat_agent import create_chat_agent
from report_agent import create_report_agent
import folium
from streamlit_folium import st_folium

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🔌 Power Outage Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== HELPER FUNCTIONS ====================
def detect_available_models():
    """Detect available models based on .env file"""
    available_models = []
    
    # Check for API keys in environment
    if os.getenv('ANTHROPIC_API_KEY'):
        available_models.extend(['Claude-3-Sonnet', 'Claude-3-Haiku'])
    if os.getenv('OPENAI_API_KEY'):
        available_models.extend(['GPT-4', 'GPT-3.5-Turbo'])
    if os.getenv('OLLAMA_BASE_URL') or os.path.exists('/usr/local/bin/ollama'):
        available_models.extend(['Llama-3-8B', 'Llama-3-70B'])
    
    # Default fallback
    if not available_models:
        available_models = ['Claude-3-Sonnet (Default)']
    
    return available_models

def check_pkl_status():
    """Check if processed results (.pkl) files exist and return detailed info"""
    pkl_files = []
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.endswith('.pkl') and 'analysis_results_' in file:
                file_path = os.path.join(cache_dir, file)
                file_stats = os.stat(file_path)
                pkl_files.append({
                    'filename': file,
                    'path': file_path,
                    'size': file_stats.st_size,
                    'modified': datetime.fromtimestamp(file_stats.st_mtime)
                })
    
    # Sort by modification time (newest first)
    pkl_files.sort(key=lambda x: x['modified'], reverse=True)
    return pkl_files

def load_pkl_results(pkl_path):
    """Load results from .pkl file"""
    try:
        import joblib
        results = joblib.load(pkl_path)
        return results
    except Exception as e:
        st.error(f"Error loading {pkl_path}: {str(e)}")
        return None

def save_results_to_pkl(results, filename=None):
    """Save results to .pkl file"""
    try:
        import joblib
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_results_{timestamp}.pkl"
        
        os.makedirs("cache", exist_ok=True)
        filepath = f"cache/{filename}"
        joblib.dump(results, filepath)
        return filename
    except Exception as e:
        st.error(f"Error saving results: {str(e)}")
        return None

def create_enhanced_map(validation_results):
    """Create an enhanced map visualization with auto-fit bounds and better styling"""
    if not validation_results or not validation_results.get('validation_complete'):
        return None
    
    real_outages = validation_results.get('real_outages', [])
    false_positives = validation_results.get('false_positives', [])
    
    if not real_outages and not false_positives:
        return None
    
    # Collect all coordinates and handle both uppercase and lowercase field names
    all_coords = []
    for outage in real_outages + false_positives:
        # Handle both uppercase and lowercase field names for compatibility
        lat = outage.get('LATITUDE') or outage.get('latitude')
        lon = outage.get('LONGITUDE') or outage.get('longitude')
        if lat and lon:
            all_coords.append([float(lat), float(lon)])
    
    if not all_coords:
        return None
    
    # Calculate bounds for auto-fit
    lats = [coord[0] for coord in all_coords]
    lons = [coord[1] for coord in all_coords]
    
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Calculate bounds with padding
    lat_range = max(lats) - min(lats)
    lon_range = max(lons) - min(lons)
    padding = max(lat_range, lon_range) * 0.1  # 10% padding
    
    # Create enhanced map with better styling
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=10,
        tiles='OpenStreetMap',
        prefer_canvas=True
    )
    
    # Add real outages (enhanced red markers)
    for outage in real_outages:
        lat = outage.get('LATITUDE') or outage.get('latitude')
        lon = outage.get('LONGITUDE') or outage.get('longitude')
        customers = outage.get('CUSTOMERS') or outage.get('customers', 'Unknown')
        confidence = outage.get('confidence', 0.8)
        reasoning = outage.get('reasoning', 'Weather analysis supports this classification')
        
        if lat and lon:
            # Popup with enhanced information
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 300px;">
                <h4 style="color: #d32f2f; margin: 0;">🔴 Real Outage</h4>
                <hr style="margin: 5px 0;">
                <p><strong>📍 Location:</strong> {float(lat):.4f}, {float(lon):.4f}</p>
                <p><strong>👥 Customers:</strong> {customers}</p>
                <p><strong>🎯 Confidence:</strong> {confidence*100:.1f}%</p>
                <p><strong>💡 Analysis:</strong><br/>{reasoning[:100]}...</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=max(8, min(15, int(customers) / 10)) if str(customers).isdigit() else 8,
                popup=folium.Popup(popup_html, max_width=320),
                color='darkred',
                weight=2,
                fill=True,
                fillColor='red',
                fillOpacity=0.8
            ).add_to(m)
    
    # Add false positives (enhanced blue markers)
    for outage in false_positives:
        lat = outage.get('LATITUDE') or outage.get('latitude')
        lon = outage.get('LONGITUDE') or outage.get('longitude')
        customers = outage.get('CUSTOMERS') or outage.get('customers', 'Unknown')
        confidence = outage.get('confidence', 0.8)
        reasoning = outage.get('reasoning', 'Weather analysis suggests false positive')
        
        if lat and lon:
            # Popup with enhanced information
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 300px;">
                <h4 style="color: #1976d2; margin: 0;">🔵 False Positive</h4>
                <hr style="margin: 5px 0;">
                <p><strong>📍 Location:</strong> {float(lat):.4f}, {float(lon):.4f}</p>
                <p><strong>👥 Customers:</strong> {customers}</p>
                <p><strong>🎯 Confidence:</strong> {confidence*100:.1f}%</p>
                <p><strong>💡 Analysis:</strong><br/>{reasoning[:100]}...</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=max(6, min(12, int(customers) / 15)) if str(customers).isdigit() else 6,
                popup=folium.Popup(popup_html, max_width=320),
                color='darkblue',
                weight=2,
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.6
            ).add_to(m)
    
    # Auto-fit bounds to show all markers
    if len(all_coords) > 1:
        m.fit_bounds([[min(lats) - padding, min(lons) - padding], 
                      [max(lats) + padding, max(lons) + padding]])
    
    # Add enhanced legend
    legend_html = f"""
    <div style="position: fixed; 
                top: 70px; right: 10px; width: 200px; 
                background-color: white; border: 2px solid grey; 
                border-radius: 10px; padding: 15px; z-index: 1000;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h4 style="margin: 0 0 10px 0; text-align: center;">🗺️ Outage Map</h4>
        <p style="margin: 5px 0;"><span style="color: red; font-size: 16px;">●</span> <strong>Real Outages</strong> ({len(real_outages)})</p>
        <p style="margin: 5px 0;"><span style="color: blue; font-size: 16px;">●</span> <strong>False Positives</strong> ({len(false_positives)})</p>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0; font-size: 12px; color: #666;">
            <strong>Total Points:</strong> {len(all_coords)}<br/>
            <strong>Accuracy:</strong> {len(real_outages)/(len(real_outages)+len(false_positives))*100:.1f}%
        </p>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# ==================== SESSION STATE INITIALIZATION ====================
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'current_pkl_file' not in st.session_state:
    st.session_state.current_pkl_file = None

# ==================== MAIN UI ====================
st.title("🔌 Power Outage Analysis")
st.markdown("*Simple, focused analysis with weather correlation*")

# Create layout: 1/4 sidebar + 3/4 main area
col_sidebar, col_main = st.columns([1, 3])

# ==================== LEFT SIDEBAR (1/4) ====================
with col_sidebar:
    st.markdown("### 🔧 Controls")
    
    # Model Selection
    st.markdown("**🤖 Model Selection**")
    available_models = detect_available_models()
    selected_model = st.selectbox(
        "Choose AI Model:",
        available_models,
        index=0,
        key="model_select"
    )
    st.session_state.selected_model = selected_model
    
    st.markdown("---")
    
    # Smart .pkl Detection and Loading
    st.markdown("**📁 Saved Results**")
    pkl_files = check_pkl_status()
    if pkl_files:
        st.success(f"✅ {len(pkl_files)} saved analysis(es)")
        
        # Show latest file info
        latest_file = pkl_files[0]
        st.info(f"Latest: {latest_file['filename'][:30]}... ({latest_file['modified'].strftime('%m/%d %H:%M')})")
        
        # Auto-load latest if no current results
        if st.session_state.analysis_results is None:
            if st.button("📂 Load Latest Results", type="primary"):
                loaded_results = load_pkl_results(latest_file['path'])
                if loaded_results:
                    st.session_state.analysis_results = loaded_results
                    st.session_state.current_pkl_file = latest_file['path']
                    st.success(f"Loaded: {latest_file['filename']}")
                    st.rerun()
        
        # Manual selection for other files
        if len(pkl_files) > 1:
            with st.expander("📋 Choose Different Analysis"):
                for i, pkl_file in enumerate(pkl_files[1:6]):  # Show up to 5 more files
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"{pkl_file['filename'][:40]}...")
                        st.caption(f"Modified: {pkl_file['modified'].strftime('%Y-%m-%d %H:%M')}")
                    with col2:
                        if st.button("Load", key=f"load_{i}"):
                            loaded_results = load_pkl_results(pkl_file['path'])
                            if loaded_results:
                                st.session_state.analysis_results = loaded_results
                                st.session_state.current_pkl_file = pkl_file['path']
                                st.success(f"Loaded: {pkl_file['filename']}")
                                st.rerun()
    else:
        st.info("📝 No saved analyses - run processing to create one")
    
    st.markdown("---")
    
    # Main Processing Button
    st.markdown("**🚀 Analysis**")
    
    if st.button("START PROCESSING", type="primary", use_container_width=True, disabled=st.session_state.processing):
        st.session_state.processing = True
        with st.spinner("🔄 Processing outage data..."):
            try:
                # Run the new agent analysis
                initial_state = {
                    "dataset_path": "data/raw_data.csv",
                    "llm_provider": selected_model.lower().split('-')[0] if selected_model else "claude",
                    "max_records_to_process": 10  # Limit for demo
                }
                results = run_analysis(initial_state)
                
                st.session_state.analysis_results = results
                
                # Set current pkl file if available
                if results and results.get('pkl_file'):
                    st.session_state.current_pkl_file = results['pkl_file']
                st.session_state.processing = False
                st.success("✅ Analysis complete!")
                st.rerun()
                
            except Exception as e:
                st.session_state.processing = False
                st.error(f"❌ Analysis failed: {str(e)}")
    
    if st.session_state.processing:
        st.info("🔄 Processing in progress...")
    
    # Reset button
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.analysis_results = None
        st.session_state.chat_history = []
        st.rerun()

# ==================== MAIN AREA (3/4) ====================
with col_main:
    if st.session_state.analysis_results is None:
        # Welcome state
        st.markdown("### 👋 Welcome to Power Outage Analysis")
        st.markdown("""
        This tool analyzes power outage reports to identify **real outages** vs **false positives** 
        based on weather conditions and other factors.
        
        **How it works:**
        1. 📊 Loads outage data from CSV files
        2. 🌤️ Gets weather data for each location/time
        3. 🤖 AI classifies each report as real or false positive
        4. 🗺️ Shows results on an interactive map
        5. 💬 Lets you ask questions about the analysis
        
        **Get started:** Click "START PROCESSING" in the sidebar to begin!
        """)
        
        st.info("💡 **Tip:** The analysis will automatically save results so you can return to them later.")
        
    else:
        # Results display with horizontal tabs
        results = st.session_state.analysis_results
        validation_results = results.get('validation_results', {})
        
        if validation_results.get('validation_complete'):
            # Create horizontal tabs for Analysis and Chat
            tab1, tab2 = st.tabs(["📊 Analysis", "💬 Chat"])
            
            with tab1:
                # Summary metrics at the top
                real_outages = validation_results.get('real_outages', [])
                false_positives = validation_results.get('false_positives', [])
                total = len(real_outages) + len(false_positives)
                accuracy = len(real_outages) / total * 100 if total > 0 else 0
                
                st.markdown("### 📊 Analysis Summary")
                
                # Key metrics in columns
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("🔴 Real Outages", len(real_outages))
                with metric_col2:
                    st.metric("🔵 False Positives", len(false_positives))
                with metric_col3:
                    st.metric("📊 Total Reports", total)
                with metric_col4:
                    st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
                
                st.markdown("---")
                
                # Enhanced Map visualization - BIGGER and better!
                st.markdown("### 🗺️ Geographic Distribution")
                map_obj = create_enhanced_map(validation_results)
                if map_obj:
                    # Make the map much larger and more prominent
                    st_folium(map_obj, width=None, height=600, use_container_width=True)
                else:
                    st.info("No geographic data available for mapping")
                
                st.markdown("---")
                
                # Quick insights and report generation - make more prominent
                st.markdown("### 💡 Quick Insights")
                if len(real_outages) > len(false_positives):
                    st.success(f"✅ Good accuracy! Most reports ({len(real_outages)}/{total}) were genuine outages.")
                elif len(false_positives) > len(real_outages):
                    st.warning(f"⚠️ High false positive rate: {len(false_positives)}/{total} reports were false alarms.")
                else:
                    st.info("📊 Balanced results: Equal mix of real outages and false positives.")
                
                # Make report button more prominent
                st.markdown("### 📄 Generate Comprehensive Report")
                if st.button("📊 Generate Detailed Analysis Report", type="primary", use_container_width=True, help="Create comprehensive analysis report with detailed insights"):
                    pkl_file = getattr(st.session_state, 'current_pkl_file', None)
                    if pkl_file:
                        with st.spinner("🔄 Generating comprehensive report..."):
                            try:
                                report_agent = create_report_agent(pkl_file)
                                report = report_agent.generate_comprehensive_report("standard")
                                
                                if report.get("report_file"):
                                    st.success(f"✅ Report generated successfully: {report['report_file']}")
                                    st.balloons()
                                else:
                                    st.error("❌ Report generation failed")
                                    
                            except Exception as e:
                                st.error(f"❌ Error generating report: {str(e)}")
                    else:
                        st.warning("⚠️ No analysis results available for report generation")
            
            with tab2:
                st.markdown("### 💬 Analysis Chat")
                st.markdown("Ask questions about your power outage analysis results:")
                
                # Chat input
                user_question = st.text_input(
                    "Your question:", 
                    placeholder="Why were there so many false positives? What weather patterns caused real outages?",
                    key="tab_chat_input"
                )
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    send_clicked = st.button("Send 💬", type="primary", use_container_width=True)
                
                if send_clicked and user_question:
                    with st.spinner("🤖 Analyzing your question..."):
                        try:
                            pkl_file = getattr(st.session_state, 'current_pkl_file', None)
                            if pkl_file:
                                chat_agent = create_chat_agent(pkl_file)
                                response = chat_agent.answer_question(user_question)
                                
                                st.session_state.chat_history.append({
                                    "question": user_question,
                                    "response": response,
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                })
                                st.rerun()
                            else:
                                st.error("❌ No analysis loaded for chat")
                        except Exception as e:
                            st.error(f"❌ Chat error: {str(e)}")
                
                # Display chat history
                if st.session_state.chat_history:
                    st.markdown("---")
                    st.markdown("### 📝 Chat History")
                    
                    for i, chat in enumerate(reversed(st.session_state.chat_history)):
                        with st.expander(f"💭 {chat['question'][:60]}{'...' if len(chat['question']) > 60 else ''}", expanded=(i==0)):
                            st.markdown(f"**🙋 Question:** {chat['question']}")
                            st.markdown(f"**🤖 Answer:** {chat['response']}")
                            st.caption(f"⏰ {chat['timestamp']}")
                else:
                    st.info("💡 No chat history yet. Ask a question above to get started!")
                    
                    # Show example questions
                    st.markdown("**💡 Example questions you can ask:**")
                    example_questions = [
                        "What weather conditions led to the most real outages?",
                        "Why were there false positive reports?", 
                        "Which areas had the highest outage rates?",
                        "What time patterns do you see in the outage data?",
                        "How reliable is this classification?"
                    ]
                    
                    for q in example_questions:
                        if st.button(f"💭 {q}", key=f"example_{hash(q)}", use_container_width=True):
                            # Set the question and trigger processing
                            st.session_state.example_question = q
                            st.rerun()
                    
                    # Handle example question click
                    if hasattr(st.session_state, 'example_question'):
                        with st.spinner("🤖 Processing example question..."):
                            try:
                                pkl_file = getattr(st.session_state, 'current_pkl_file', None) 
                                if pkl_file:
                                    chat_agent = create_chat_agent(pkl_file)
                                    response = chat_agent.answer_question(st.session_state.example_question)
                                    
                                    st.session_state.chat_history.append({
                                        "question": st.session_state.example_question,
                                        "response": response,
                                        "timestamp": datetime.now().strftime("%H:%M:%S")
                                    })
                                    delattr(st.session_state, 'example_question')
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                                delattr(st.session_state, 'example_question')
        
        else:
            st.warning("⏳ Analysis in progress or incomplete...")

# ==================== REMOVE SIDEBAR CHAT (now in tabs) ====================
with col_sidebar:
    # Remove all chat functionality from sidebar - now it's cleaner
    if st.session_state.analysis_results and st.session_state.analysis_results.get('validation_complete'):
        st.markdown("---")
        st.info("💬 Use the Chat tab above to ask questions about your analysis!")
    else:
        st.markdown("---")
        st.info("💬 Chat will be available after analysis completes")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("*Powered by LangGraph + Weather APIs + AI Classification*")