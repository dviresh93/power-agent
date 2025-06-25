"""
Vector Database Service for Outage Data Management
- ChromaDB integration for outage data storage
- Data loading from CSV files
- Summary generation and caching
- Document processing and metadata management
- Collection management functions
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List
import joblib
import chromadb

# Set up logging
logger = logging.getLogger(__name__)


class OutageVectorDB:
    """Vector database for outage data management"""
    
    def __init__(self, db_path: str = "./chroma_db"):
        """
        Initialize the vector database connection.
        
        Args:
            db_path (str): Path to the ChromaDB storage directory
        """
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            
            try:
                self.collection = self.client.get_collection("outages")
                logger.info("✅ Connected to existing outages collection")
            except:
                self.collection = self.client.create_collection("outages")
                logger.info("✅ Created new outages collection")
        except Exception as e:
            logger.error(f"❌ Vector DB initialization failed: {str(e)}")
            raise
    
    def load_outage_data(self, df: pd.DataFrame, force_reload: bool = False) -> Dict:
        """
        Load data and return summary with caching support.
        
        Args:
            df (pd.DataFrame): DataFrame containing outage data with columns:
                              DATETIME, LATITUDE, LONGITUDE, CUSTOMERS
            force_reload (bool): If True, bypass cache and reload all data
            
        Returns:
            Dict: Summary of the loaded outage data
        """
        try:
            # Check for cached data first
            cache_dir = "./cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "outage_data_summary.joblib")
            vector_cache_file = os.path.join(cache_dir, "vector_db_status.json")
            
            # Load from cache if available and not forcing reload
            if not force_reload and os.path.exists(cache_file) and os.path.exists(vector_cache_file):
                try:
                    logger.info("🔄 Loading dataset summary from cache")
                    cached_summary = joblib.load(cache_file)
                    return cached_summary
                except Exception as e:
                    logger.warning(f"⚠️ Cache loading failed: {str(e)}. Proceeding with full load.")
            
            logger.info(f"📊 Loading {len(df)} outage records into vector database")
            
            # Clear existing data
            try:
                self.client.delete_collection("outages")
                self.collection = self.client.create_collection("outages")
            except:
                pass
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in df.iterrows():
                doc_text = f"""Power outage on {row['DATETIME']} at coordinates {row['LATITUDE']}, {row['LONGITUDE']} affecting {row['CUSTOMERS']} customers."""
                
                metadata = {
                    'DATETIME': row['DATETIME'],
                    'LATITUDE': float(row['LATITUDE']),
                    'LONGITUDE': float(row['LONGITUDE']),
                    'CUSTOMERS': int(row['CUSTOMERS']),
                    'date': row['DATETIME'][:10],
                    'hour': int(row['DATETIME'][11:13]) if len(row['DATETIME']) > 11 else 0
                }
                
                documents.append(doc_text.strip())
                metadatas.append(metadata)
                ids.append(f"outage_{idx}")
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Generate raw dataset summary
            summary = self._generate_raw_summary(df)
            
            # Cache the results
            try:
                joblib.dump(summary, cache_file)
                with open(vector_cache_file, 'w') as f:
                    json.dump({"timestamp": datetime.now().isoformat(), "record_count": len(df)}, f)
                logger.info("✅ Data cached successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cache data: {str(e)}")
                
            logger.info("✅ Data loaded successfully")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def _generate_raw_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary of raw dataset before validation.
        
        Args:
            df (pd.DataFrame): DataFrame containing outage data
            
        Returns:
            Dict: Summary statistics including date ranges, customer counts, and geographic coverage
        """
        try:
            df['datetime_parsed'] = pd.to_datetime(df['DATETIME'])
            
            return {
                "total_reports": int(len(df)),
                "date_range": {
                    "start": df['datetime_parsed'].min().strftime('%Y-%m-%d'),
                    "end": df['datetime_parsed'].max().strftime('%Y-%m-%d')
                },
                "raw_customer_claims": {
                    "total_claimed": int(df['CUSTOMERS'].sum()),
                    "avg_per_report": float(df['CUSTOMERS'].mean()),
                    "max_single_report": int(df['CUSTOMERS'].max())
                },
                "geographic_coverage": {
                    "lat_range": f"{float(df['LATITUDE'].min()):.3f} to {float(df['LATITUDE'].max()):.3f}",
                    "lon_range": f"{float(df['LONGITUDE'].min()):.3f} to {float(df['LONGITUDE'].max()):.3f}",
                    "center": [float(df['LATITUDE'].mean()), float(df['LONGITUDE'].mean())]
                }
            }
        except Exception as e:
            logger.error(f"❌ Error generating summary: {str(e)}")
            return {"error": str(e)}
    
    def query_outages(self, query_text: str, n_results: int = 10) -> Dict:
        """
        Query the vector database for similar outages.
        
        Args:
            query_text (str): Natural language query about outages
            n_results (int): Number of results to return
            
        Returns:
            Dict: Query results with documents, metadatas, and distances
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"❌ Error querying outages: {str(e)}")
            return {"error": str(e)}
    
    def get_outages_by_date_range(self, start_date: str, end_date: str) -> Dict:
        """
        Get outages within a specific date range.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            Dict: Outages within the specified date range
        """
        try:
            results = self.collection.get(
                where={
                    "$and": [
                        {"date": {"$gte": start_date}},
                        {"date": {"$lte": end_date}}
                    ]
                }
            )
            return results
        except Exception as e:
            logger.error(f"❌ Error getting outages by date range: {str(e)}")
            return {"error": str(e)}
    
    def query_outages_by_range(self, start_datetime, end_datetime) -> List[Dict]:
        """
        Query outages within a specific datetime range.
        
        Args:
            start_datetime (datetime): Start datetime object
            end_datetime (datetime): End datetime object
            
        Returns:
            List[Dict]: List of outage metadata within the specified datetime range
        """
        try:
            results = self.collection.query(
                query_texts=[f"outages between {start_datetime.strftime('%Y-%m-%d')} and {end_datetime.strftime('%Y-%m-%d')}"],
                n_results=100,
                where={
                    "DATETIME": {
                        "$gte": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                        "$lte": end_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
            )
            
            outages = []
            if results['metadatas']:
                for metadata in results['metadatas'][0]:
                    outages.append(metadata)
            
            return outages
            
        except Exception as e:
            logger.error(f"❌ Error querying outages by range: {str(e)}")
            return []
    
    def get_outages_by_location(self, center_lat: float, center_lon: float, 
                               lat_radius: float = 0.1, lon_radius: float = 0.1) -> Dict:
        """
        Get outages within a geographic area.
        
        Args:
            center_lat (float): Center latitude
            center_lon (float): Center longitude
            lat_radius (float): Latitude search radius (degrees)
            lon_radius (float): Longitude search radius (degrees)
            
        Returns:
            Dict: Outages within the specified geographic area
        """
        try:
            results = self.collection.get(
                where={
                    "$and": [
                        {"LATITUDE": {"$gte": center_lat - lat_radius}},
                        {"LATITUDE": {"$lte": center_lat + lat_radius}},
                        {"LONGITUDE": {"$gte": center_lon - lon_radius}},
                        {"LONGITUDE": {"$lte": center_lon + lon_radius}}
                    ]
                }
            )
            return results
        except Exception as e:
            logger.error(f"❌ Error getting outages by location: {str(e)}")
            return {"error": str(e)}
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the current collection.
        
        Returns:
            Dict: Collection statistics including count and sample data
        """
        try:
            count = self.collection.count()
            sample = self.collection.peek(limit=5) if count > 0 else {"documents": [], "metadatas": []}
            
            return {
                "total_documents": count,
                "sample_documents": sample.get("documents", [])[:3],
                "sample_metadata": sample.get("metadatas", [])[:3]
            }
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """
        Clear all data from the outages collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.delete_collection("outages")
            self.collection = self.client.create_collection("outages")
            logger.info("✅ Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing collection: {str(e)}")
            return False
    
    def delete_cache(self) -> bool:
        """
        Delete cached data files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cache_dir = "./cache"
            cache_file = os.path.join(cache_dir, "outage_data_summary.joblib")
            vector_cache_file = os.path.join(cache_dir, "vector_db_status.json")
            
            if os.path.exists(cache_file):
                os.remove(cache_file)
            if os.path.exists(vector_cache_file):
                os.remove(vector_cache_file)
                
            logger.info("✅ Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {str(e)}")
            return False