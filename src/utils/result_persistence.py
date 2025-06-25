"""
Result Persistence Utilities
Handles saving and loading of validation results in various formats
"""

import pickle
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ResultsPersistence:
    """Handles saving and loading validation results in multiple formats"""
    
    def __init__(self, cache_dir: str = "cache", reports_dir: str = "reports"):
        self.cache_dir = Path(cache_dir)
        self.reports_dir = Path(reports_dir)
        
        # Ensure directories exist
        self.cache_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
    def save_validation_results(self, 
                               validation_results: Dict, 
                               format_type: str = "pickle",
                               filename: str = None,
                               include_metadata: bool = True) -> str:
        """
        Save validation results in specified format
        
        Args:
            validation_results: Results dictionary to save
            format_type: 'pickle', 'json', 'csv', or 'all'
            filename: Custom filename (without extension)
            include_metadata: Whether to include metadata like timestamp
            
        Returns:
            Path to saved file(s) or primary file if multiple formats
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"validation_results_{timestamp}"
        
        # Add metadata if requested
        if include_metadata:
            validation_results = self._add_metadata(validation_results)
        
        files_saved = []
        
        if format_type in ["pickle", "all"]:
            pickle_path = self._save_pickle(validation_results, filename)
            files_saved.append(pickle_path)
            
        if format_type in ["json", "all"]:
            json_path = self._save_json(validation_results, filename)
            files_saved.append(json_path)
            
        if format_type in ["csv", "all"]:
            csv_paths = self._save_csv(validation_results, filename)
            files_saved.extend(csv_paths)
        
        primary_file = files_saved[0] if files_saved else None
        logger.info(f"Saved validation results to {len(files_saved)} file(s). Primary: {primary_file}")
        
        return primary_file
    
    def load_validation_results(self, filepath: str) -> Dict:
        """
        Load validation results from file (auto-detects format)
        
        Args:
            filepath: Path to file to load
            
        Returns:
            Loaded validation results dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        if filepath.suffix == '.pkl':
            return self._load_pickle(filepath)
        elif filepath.suffix == '.json':
            return self._load_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def list_saved_results(self, format_type: str = "all") -> List[Dict]:
        """
        List all saved validation results
        
        Args:
            format_type: 'pickle', 'json', or 'all'
            
        Returns:
            List of file information dictionaries
        """
        files = []
        
        # Check cache directory for pickle and json files
        patterns = []
        if format_type in ["pickle", "all"]:
            patterns.append("validation_results_*.pkl")
        if format_type in ["json", "all"]:
            patterns.append("validation_results_*.json")
        
        for pattern in patterns:
            for file_path in self.cache_dir.glob(pattern):
                files.append({
                    "path": str(file_path),
                    "filename": file_path.name,
                    "format": file_path.suffix[1:],  # Remove the dot
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime),
                    "created": datetime.fromtimestamp(file_path.stat().st_ctime)
                })
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x["modified"], reverse=True)
        return files
    
    def delete_saved_results(self, filepath: str) -> bool:
        """
        Delete saved results file
        
        Args:
            filepath: Path to file to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            Path(filepath).unlink()
            logger.info(f"Deleted results file: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {filepath}: {e}")
            return False
    
    def get_results_summary(self, filepath: str) -> Dict:
        """
        Get summary information about saved results without loading full data
        
        Args:
            filepath: Path to results file
            
        Returns:
            Summary dictionary
        """
        try:
            results = self.load_validation_results(filepath)
            
            real_outages = results.get('real_outages', [])
            false_positives = results.get('false_positives', [])
            total_processed = len(real_outages) + len(false_positives)
            
            summary = {
                "total_processed": total_processed,
                "real_outages": len(real_outages),
                "false_positives": len(false_positives),
                "accuracy_rate": len(real_outages) / total_processed if total_processed > 0 else 0,
                "has_metadata": "metadata" in results,
                "file_info": {
                    "path": filepath,
                    "size": Path(filepath).stat().st_size,
                    "modified": datetime.fromtimestamp(Path(filepath).stat().st_mtime)
                }
            }
            
            # Add metadata info if available
            if "metadata" in results:
                metadata = results["metadata"]
                summary["metadata"] = {
                    "saved_at": metadata.get("saved_at"),
                    "version": metadata.get("version"),
                    "agent_version": metadata.get("agent_version")
                }
            
            return summary
            
        except Exception as e:
            return {"error": f"Failed to get summary: {str(e)}"}
    
    def _add_metadata(self, validation_results: Dict) -> Dict:
        """Add metadata to validation results"""
        results_with_metadata = validation_results.copy()
        results_with_metadata["metadata"] = {
            "saved_at": datetime.now().isoformat(),
            "version": "1.0",
            "agent_version": "langgraph-v1",
            "format_version": "2025.1"
        }
        return results_with_metadata
    
    def _save_pickle(self, data: Dict, filename: str) -> str:
        """Save data as pickle file"""
        filepath = self.cache_dir / f"{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return str(filepath)
    
    def _save_json(self, data: Dict, filename: str) -> str:
        """Save data as JSON file"""
        filepath = self.cache_dir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return str(filepath)
    
    def _save_csv(self, data: Dict, filename: str) -> List[str]:
        """Save data as CSV files (separate files for real outages and false positives)"""
        files_saved = []
        
        real_outages = data.get('real_outages', [])
        false_positives = data.get('false_positives', [])
        
        if real_outages:
            real_df = pd.DataFrame(real_outages)
            real_path = self.cache_dir / f"{filename}_real_outages.csv"
            real_df.to_csv(real_path, index=False)
            files_saved.append(str(real_path))
        
        if false_positives:
            false_df = pd.DataFrame(false_positives)
            false_path = self.cache_dir / f"{filename}_false_positives.csv"
            false_df.to_csv(false_path, index=False)
            files_saved.append(str(false_path))
        
        return files_saved
    
    def _load_pickle(self, filepath: Path) -> Dict:
        """Load data from pickle file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _load_json(self, filepath: Path) -> Dict:
        """Load data from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)

# Global instance for easy import
results_persistence = ResultsPersistence()

def save_results(validation_results: Dict, 
                format_type: str = "pickle", 
                filename: str = None) -> str:
    """Convenience function to save validation results"""
    return results_persistence.save_validation_results(
        validation_results, format_type, filename
    )

def load_results(filepath: str) -> Dict:
    """Convenience function to load validation results"""
    return results_persistence.load_validation_results(filepath)

def list_results(format_type: str = "all") -> List[Dict]:
    """Convenience function to list saved results"""
    return results_persistence.list_saved_results(format_type)

def get_summary(filepath: str) -> Dict:
    """Convenience function to get results summary"""
    return results_persistence.get_results_summary(filepath)