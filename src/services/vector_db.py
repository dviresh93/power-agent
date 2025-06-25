import os
import json
import joblib
import pandas as pd
from typing import Dict, List
from datetime import datetime
import chromadb
from chromadb.config import Settings

class OutageVectorDB:
    def __init__(self):
        self.db_path = "./data/vector_db"
        self.cache_dir = "./cache"
        self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB with proper settings"""
        try:
            os.makedirs(self.db_path, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.collection = self.client.get_or_create_collection(
                name="outage_reports",
                metadata={"description": "Power outage reports with weather correlation"}
            )
            
        except Exception as e:
            print(f"Error initializing vector DB: {str(e)}")
            raise

    def load_outage_data(self, df: pd.DataFrame, force_reload: bool = False) -> Dict:
        """Load outage data into vector database with caching"""
        cache_file = os.path.join(self.cache_dir, "outage_data_summary.joblib")
        vector_cache_file = os.path.join(self.cache_dir, "vector_db_status.json")
        
        # Check cache first
        if not force_reload and os.path.exists(cache_file) and os.path.exists(vector_cache_file):
            try:
                with open(vector_cache_file, 'r') as f:
                    cache_metadata = json.load(f)
                if cache_metadata.get('record_count') == len(df):
                    return joblib.load(cache_file)
            except Exception as e:
                print(f"Warning: Cache check failed: {str(e)}")
        
        try:
            # Prepare documents for vector DB
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in df.iterrows():
                doc_text = f"""Power outage on {row['DATETIME']} at coordinates {row['LATITUDE']}, {row['LONGITUDE']} affecting {row['CUSTOMERS']} customers."""
                
                metadata = {
                    'DATETIME': row['DATETIME'],  # PRESERVE ORIGINAL UPPERCASE FIELD NAMES
                    'LATITUDE': float(row['LATITUDE']),
                    'LONGITUDE': float(row['LONGITUDE']),
                    'CUSTOMERS': int(row['CUSTOMERS']),
                    'date': row['DATETIME'][:10],
                    'hour': int(row['DATETIME'][11:13]) if len(row['DATETIME']) > 11 else 0
                }
                
                documents.append(doc_text.strip())
                metadatas.append(metadata)
                ids.append(f"outage_{idx}")
            
            # Add to vector DB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Generate summary
            summary = self._generate_raw_summary(df)
            
            # Cache results
            try:
                joblib.dump(summary, cache_file)
                with open(vector_cache_file, 'w') as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "record_count": len(df)
                    }, f)
            except Exception as e:
                print(f"Warning: Failed to cache data: {str(e)}")
            
            return summary
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _generate_raw_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary of raw dataset"""
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
            print(f"Error generating summary: {str(e)}")
            return {"error": str(e)} 