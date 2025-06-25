"""
Enhanced Vector Database Service for Outage Analysis Results
- Stores both raw data and analysis results with reasoning
- Supports semantic search for chat interactions
- Maintains analysis metadata and confidence scores
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import joblib
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class EnhancedOutageVectorDB:
    """Enhanced vector database that stores analysis results with reasoning"""
    
    def __init__(self, db_path: str = "./chroma_db"):
        """Initialize enhanced vector database"""
        try:
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Raw data collection (existing)
            try:
                self.raw_collection = self.client.get_collection("outages")
                logger.info("✅ Connected to existing raw outages collection")
            except:
                self.raw_collection = self.client.create_collection("outages")
                logger.info("✅ Created new raw outages collection")
            
            # Analysis results collection (new)
            try:
                self.analysis_collection = self.client.get_collection("analysis_results")
                logger.info("✅ Connected to existing analysis results collection")
            except:
                self.analysis_collection = self.client.create_collection("analysis_results")
                logger.info("✅ Created new analysis results collection")
                
        except Exception as e:
            logger.error(f"❌ Enhanced Vector DB initialization failed: {str(e)}")
            raise
    
    def load_outage_data(self, df: pd.DataFrame, force_reload: bool = False) -> Dict:
        """Load raw outage data (existing functionality)"""
        try:
            cache_dir = "./cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "outage_data_summary.joblib")
            vector_cache_file = os.path.join(cache_dir, "vector_db_status.json")
            
            if not force_reload and os.path.exists(cache_file) and os.path.exists(vector_cache_file):
                try:
                    logger.info("🔄 Loading dataset summary from cache")
                    return joblib.load(cache_file)
                except Exception as e:
                    logger.warning(f"⚠️ Cache loading failed: {str(e)}")
            
            logger.info(f"📊 Loading {len(df)} outage records into vector database")
            
            # Clear existing raw data
            try:
                self.client.delete_collection("outages")
                self.raw_collection = self.client.create_collection("outages")
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
                    'hour': int(row['DATETIME'][11:13]) if len(row['DATETIME']) > 11 else 0,
                    'record_id': f"raw_{idx}",
                    'data_type': 'raw_outage'
                }
                
                documents.append(doc_text.strip())
                metadatas.append(metadata)
                ids.append(f"raw_outage_{idx}")
            
            self.raw_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            summary = self._generate_raw_summary(df)
            
            try:
                joblib.dump(summary, cache_file)
                with open(vector_cache_file, 'w') as f:
                    json.dump({"timestamp": datetime.now().isoformat(), "record_count": len(df)}, f)
                logger.info("✅ Raw data cached successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cache data: {str(e)}")
                
            logger.info("✅ Raw data loaded successfully")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error loading raw data: {str(e)}")
            raise
    
    def store_analysis_results(self, validation_results: Dict, batch_id: str = None) -> bool:
        """Store analysis results with reasoning in vector database"""
        try:
            if not batch_id:
                batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            logger.info("🔄 Storing analysis results in vector database...")
            
            real_outages = validation_results.get('real_outages', [])
            false_positives = validation_results.get('false_positives', [])
            
            documents = []
            metadatas = []
            ids = []
            
            # Store real outages with analysis
            for idx, outage in enumerate(real_outages):
                doc_text = f"""REAL OUTAGE: Power outage on {outage.get('datetime', 'unknown')} at {outage.get('latitude', 0)}, {outage.get('longitude', 0)} affecting {outage.get('customers', 0)} customers. 

ANALYSIS: Classified as real outage with {outage.get('confidence', 0.8)*100:.1f}% confidence.

REASONING: {outage.get('reasoning', 'Weather conditions and historical patterns support this classification.')}

WEATHER CONTEXT: {outage.get('weather_conditions', 'Weather data analyzed for correlation.')}"""
                
                metadata = {
                    'datetime': outage.get('datetime', ''),
                    'latitude': float(outage.get('latitude', 0)),
                    'longitude': float(outage.get('longitude', 0)),
                    'customers': int(outage.get('customers', 0)),
                    'classification': 'real_outage',
                    'confidence': float(outage.get('confidence', 0.8)),
                    'reasoning': outage.get('reasoning', ''),
                    'weather_conditions': str(outage.get('weather_conditions', {})),
                    'severity': outage.get('severity', 5),
                    'analysis_batch': batch_id,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'data_type': 'analysis_result'
                }
                
                documents.append(doc_text.strip())
                metadatas.append(metadata)
                ids.append(f"analysis_real_{batch_id}_{idx}")
            
            # Store false positives with analysis
            for idx, outage in enumerate(false_positives):
                doc_text = f"""FALSE POSITIVE: Power outage report on {outage.get('datetime', 'unknown')} at {outage.get('latitude', 0)}, {outage.get('longitude', 0)} claiming {outage.get('customers', 0)} customers affected.

ANALYSIS: Classified as false positive with {outage.get('confidence', 0.8)*100:.1f}% confidence.

REASONING: {outage.get('reasoning', 'Weather conditions and analysis indicate this was likely not a real outage.')}

WEATHER CONTEXT: {outage.get('weather_conditions', 'Weather data suggests conditions were not severe enough for outages.')}"""
                
                metadata = {
                    'datetime': outage.get('datetime', ''),
                    'latitude': float(outage.get('latitude', 0)),
                    'longitude': float(outage.get('longitude', 0)),
                    'customers': int(outage.get('customers', 0)),
                    'classification': 'false_positive',
                    'confidence': float(outage.get('confidence', 0.8)),
                    'reasoning': outage.get('reasoning', ''),
                    'weather_conditions': str(outage.get('weather_conditions', {})),
                    'severity': outage.get('severity', 2),
                    'analysis_batch': batch_id,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'data_type': 'analysis_result'
                }
                
                documents.append(doc_text.strip())
                metadatas.append(metadata)
                ids.append(f"analysis_false_{batch_id}_{idx}")
            
            # Add to analysis collection
            if documents:
                self.analysis_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"✅ Stored {len(documents)} analysis results in vector database")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing analysis results: {str(e)}")
            return False
    
    def query_for_chat(self, user_question: str, n_results: int = 5) -> Dict:
        """Query analysis results for chat context using semantic search"""
        try:
            # Search analysis results for relevant context
            results = self.analysis_collection.query(
                query_texts=[user_question],
                n_results=n_results,
                where={"data_type": "analysis_result"}
            )
            
            if not results or not results.get('documents'):
                return {"documents": [], "metadatas": [], "context": "No relevant analysis results found."}
                
            # Format context for LLM
            context_parts = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                context_parts.append(f"Classification: {metadata['classification']}")
                context_parts.append(f"Confidence: {metadata['confidence']:.2f}")
                context_parts.append(f"Reasoning: {metadata['reasoning']}")
                context_parts.append("---")
            
            return {
                "documents": results['documents'][0],
                "metadatas": results['metadatas'][0],
                "context": "\n".join(context_parts),
                "total_results": len(results['documents'][0])
            }
            
        except Exception as e:
            logger.error(f"❌ Error querying for chat: {str(e)}")
            return {"error": str(e), "documents": [], "metadatas": [], "context": ""}
    
    def query_by_classification(self, classification: str, limit: int = 50) -> List[Dict]:
        """Query results by classification type"""
        try:
            results = self.analysis_collection.get(
                where={
                    "$and": [
                        {"data_type": "analysis_result"},
                        {"classification": classification}
                    ]
                },
                limit=limit
            )
            
            if results and results.get('metadatas'):
                return results['metadatas']
            return []
            
        except Exception as e:
            logger.error(f"❌ Error querying by classification: {str(e)}")
            return []
    
    def query_by_confidence_range(self, min_confidence: float, max_confidence: float = 1.0) -> List[Dict]:
        """Query results by confidence score range"""
        try:
            results = self.analysis_collection.get(
                where={
                    "$and": [
                        {"data_type": "analysis_result"},
                        {"confidence": {"$gte": min_confidence}},
                        {"confidence": {"$lte": max_confidence}}
                    ]
                },
                limit=100
            )
            
            if results and results.get('metadatas'):
                return results['metadatas']
            return []
            
        except Exception as e:
            logger.error(f"❌ Error querying by confidence: {str(e)}")
            return []
    
    def get_analysis_summary(self) -> Dict:
        """Get summary of analysis results stored in vector DB"""
        try:
            # Get all analysis results
            all_results = self.analysis_collection.get(
                where={"data_type": "analysis_result"},
                limit=1000
            )
            
            if not all_results or not all_results.get('metadatas'):
                return {"total": 0, "real_outages": 0, "false_positives": 0}
            
            metadatas = all_results['metadatas']
            real_count = len([m for m in metadatas if m.get('classification') == 'real_outage'])
            false_count = len([m for m in metadatas if m.get('classification') == 'false_positive'])
            
            # Calculate average confidence
            confidences = [m.get('confidence', 0.0) for m in metadatas]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "total": len(metadatas),
                "real_outages": real_count,
                "false_positives": false_count,
                "accuracy_rate": real_count / len(metadatas) if metadatas else 0.0,
                "average_confidence": avg_confidence,
                "latest_analysis": max([m.get('analysis_timestamp', '') for m in metadatas]) if metadatas else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting analysis summary: {str(e)}")
            return {"error": str(e)}
    
    def _generate_raw_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary of raw dataset"""
        try:
            df['datetime_parsed'] = pd.to_datetime(df['DATETIME'])
            
            return {
                "total_records": int(len(df)),
                "date_range": {
                    "start": df['datetime_parsed'].min().strftime('%Y-%m-%d'),
                    "end": df['datetime_parsed'].max().strftime('%Y-%m-%d')
                },
                "customer_stats": {
                    "total_claimed": int(df['CUSTOMERS'].sum()),
                    "avg_per_report": float(df['CUSTOMERS'].mean()),
                    "max_single_report": int(df['CUSTOMERS'].max())
                },
                "geographic_bounds": {
                    "lat_range": f"{float(df['LATITUDE'].min()):.3f} to {float(df['LATITUDE'].max()):.3f}",
                    "lon_range": f"{float(df['LONGITUDE'].min()):.3f} to {float(df['LONGITUDE'].max()):.3f}",
                    "center": [float(df['LATITUDE'].mean()), float(df['LONGITUDE'].mean())]
                }
            }
        except Exception as e:
            logger.error(f"❌ Error generating raw summary: {str(e)}")
            return {"error": str(e)}
    
    def clear_analysis_results(self) -> bool:
        """Clear analysis results collection"""
        try:
            self.client.delete_collection("analysis_results")
            self.analysis_collection = self.client.create_collection("analysis_results")
            logger.info("✅ Analysis results collection cleared")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing analysis results: {str(e)}")
            return False 