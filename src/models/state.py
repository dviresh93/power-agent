from typing import TypedDict, Dict, List, Optional

class OutageAnalysisState(TypedDict):
    """2025 LangGraph State Management"""
    dataset_loaded: bool
    validation_complete: bool
    raw_dataset_summary: Dict
    validation_results: Dict
    filtered_summary: Dict  # Summary after false positive filtering
    current_window_analysis: Optional[Dict]
    chat_context: Dict
    errors: List[str] 