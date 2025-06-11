"""
Geocoding Service Module

Provides clean, reusable geocoding functionality with:
- Reverse geocoding using geopy/Nominatim
- Persistent caching using joblib
- Error handling for geocoding timeouts/failures
- Location data formatting
- Cache management functions
- Independence from UI frameworks
"""

import os
import logging
import joblib
from functools import lru_cache
from typing import Dict, Optional

# Reverse geocoding for city/county names
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class GeocodingService:
    """Service for reverse geocoding lat/lon to city/county names"""
    
    def __init__(self, cache_dir: str = "./cache", user_agent: str = "power-outage-agent"):
        """
        Initialize geocoding service with configurable cache directory and user agent
        
        Args:
            cache_dir: Directory to store persistent cache files
            user_agent: User agent string for geopy requests
        """
        self.cache_dir = cache_dir
        self.user_agent = user_agent
        
        if GEOPY_AVAILABLE:
            self.geolocator = Nominatim(user_agent=self.user_agent)
            self._cache = {}  # In-memory cache
            self.cache_file = os.path.join(self.cache_dir, "geocoding_cache.joblib")
            self._load_persistent_cache()
            logger.info("✅ Geocoding service initialized with persistent cache")
        else:
            self.geolocator = None
            logger.warning("⚠️ Geocoding service unavailable - install geopy")
    
    def _load_persistent_cache(self):
        """Load geocoding cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                self._cache = joblib.load(self.cache_file)
                logger.info(f"📍 Loaded {len(self._cache)} cached locations from disk")
            else:
                self._cache = {}
        except Exception as e:
            logger.warning(f"⚠️ Failed to load geocoding cache: {str(e)}")
            self._cache = {}
    
    def _save_persistent_cache(self):
        """Save geocoding cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            joblib.dump(self._cache, self.cache_file)
            logger.debug(f"💾 Saved {len(self._cache)} locations to geocoding cache")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save geocoding cache: {str(e)}")
    
    @lru_cache(maxsize=1000)
    def get_location_name(self, lat: float, lon: float) -> Dict[str, str]:
        """
        Get city, county, state for given coordinates
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            
        Returns:
            Dictionary containing city, county, state, and display_name
        """
        if not self.geolocator:
            return {
                'city': 'Unknown',
                'county': 'Unknown', 
                'state': 'Unknown',
                'display_name': f"Lat: {lat:.4f}, Lon: {lon:.4f}"
            }
        
        # Round coordinates to reduce cache misses for nearby points
        lat_rounded = round(lat, 3)
        lon_rounded = round(lon, 3)
        cache_key = f"{lat_rounded},{lon_rounded}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            location = self.geolocator.reverse(f"{lat}, {lon}", timeout=15, language='en')
            if location and location.raw.get('address'):
                address = location.raw['address']
                
                # Extract location components with better fallbacks
                city = (address.get('city') or 
                       address.get('town') or 
                       address.get('village') or 
                       address.get('hamlet') or
                       address.get('suburb') or
                       address.get('neighbourhood') or
                       address.get('locality') or
                       None)
                
                county = (address.get('county') or 
                         address.get('state_district') or
                         address.get('administrative_area_level_2') or
                         None)
                
                state = (address.get('state') or 
                        address.get('province') or
                        address.get('administrative_area_level_1') or
                        None)
                
                # Build display name more intelligently
                parts = []
                if city:
                    parts.append(city)
                if county and county != city:
                    parts.append(county)
                if state and state != county:
                    parts.append(state)
                
                if parts:
                    display_name = ", ".join(parts)
                else:
                    # Try to get any meaningful location info
                    road = address.get('road') or address.get('street')
                    postcode = address.get('postcode')
                    if road or postcode:
                        display_name = f"{road or ''} {postcode or ''}".strip()
                    else:
                        display_name = f"Near {lat:.3f}, {lon:.3f}"
                
                result = {
                    'city': city or 'Unknown',
                    'county': county or 'Unknown',
                    'state': state or 'Unknown',
                    'display_name': display_name
                }
                
                # Cache the result both in memory and disk
                self._cache[cache_key] = result
                self._save_persistent_cache()
                return result
            else:
                # No address found
                result = {
                    'city': 'Unknown',
                    'county': 'Unknown', 
                    'state': 'Unknown',
                    'display_name': f"Near {lat:.3f}, {lon:.3f}"
                }
                self._cache[cache_key] = result
                self._save_persistent_cache()
                return result
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.warning(f"⚠️ Geocoding failed for {lat}, {lon}: {str(e)}")
            result = {
                'city': 'Lookup Failed',
                'county': 'Lookup Failed', 
                'state': 'Lookup Failed',
                'display_name': f"Lat: {lat:.4f}, Lon: {lon:.4f}"
            }
            return result
        except Exception as e:
            logger.error(f"❌ Geocoding error for {lat}, {lon}: {str(e)}")
            result = {
                'city': 'Error',
                'county': 'Error', 
                'state': 'Error',
                'display_name': f"Lat: {lat:.4f}, Lon: {lon:.4f}"
            }
            return result
    
    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get statistics about the cache
        
        Returns:
            Dictionary with cache statistics
        """
        cache_size_disk = 0
        if os.path.exists(self.cache_file):
            cache_size_disk = os.path.getsize(self.cache_file)
        
        return {
            'in_memory_entries': len(self._cache),
            'cache_file_exists': os.path.exists(self.cache_file),
            'cache_file_size_bytes': cache_size_disk,
            'cache_file_path': self.cache_file,
            'geopy_available': GEOPY_AVAILABLE,
            'lru_cache_info': self.get_location_name.cache_info()._asdict() if hasattr(self.get_location_name, 'cache_info') else {}
        }
    
    def clear_cache(self, clear_disk: bool = True, clear_memory: bool = True, clear_lru: bool = True):
        """
        Clear geocoding cache
        
        Args:
            clear_disk: Whether to delete the persistent cache file
            clear_memory: Whether to clear the in-memory cache
            clear_lru: Whether to clear the LRU cache
        """
        if clear_memory:
            self._cache.clear()
            logger.info("🗑️ Cleared in-memory geocoding cache")
        
        if clear_disk and os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                logger.info("🗑️ Deleted persistent geocoding cache file")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete cache file: {str(e)}")
        
        if clear_lru:
            self.get_location_name.cache_clear()
            logger.info("🗑️ Cleared LRU cache")
    
    def preload_locations(self, coordinates_list: list) -> int:
        """
        Preload multiple locations into cache
        
        Args:
            coordinates_list: List of (lat, lon) tuples
            
        Returns:
            Number of locations successfully cached
        """
        if not self.geolocator:
            logger.warning("⚠️ Cannot preload locations - geocoding service unavailable")
            return 0
        
        cached_count = 0
        total_count = len(coordinates_list)
        
        logger.info(f"🔄 Preloading {total_count} locations...")
        
        for i, (lat, lon) in enumerate(coordinates_list):
            try:
                self.get_location_name(lat, lon)
                cached_count += 1
                
                # Progress logging for large batches
                if total_count > 100 and (i + 1) % 50 == 0:
                    logger.info(f"📍 Preloaded {i + 1}/{total_count} locations...")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to preload location {lat}, {lon}: {str(e)}")
        
        logger.info(f"✅ Preloaded {cached_count}/{total_count} locations")
        return cached_count
    
    def is_available(self) -> bool:
        """
        Check if geocoding service is available
        
        Returns:
            True if service is available, False otherwise
        """
        return GEOPY_AVAILABLE and self.geolocator is not None
    
    def get_raw_location_data(self, lat: float, lon: float, timeout: int = 15) -> Optional[Dict]:
        """
        Get raw location data from geocoding service (not cached)
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate  
            timeout: Request timeout in seconds
            
        Returns:
            Raw location data from geopy, or None if unavailable/failed
        """
        if not self.geolocator:
            return None
        
        try:
            location = self.geolocator.reverse(f"{lat}, {lon}", timeout=timeout, language='en')
            if location:
                return location.raw
            return None
        except Exception as e:
            logger.warning(f"⚠️ Raw geocoding failed for {lat}, {lon}: {str(e)}")
            return None


# Convenience function for quick geocoding without class instantiation
def quick_geocode(lat: float, lon: float, cache_dir: str = "./cache") -> Dict[str, str]:
    """
    Quick geocoding function that creates a temporary service instance
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        cache_dir: Cache directory to use
        
    Returns:
        Dictionary containing city, county, state, and display_name
    """
    service = GeocodingService(cache_dir=cache_dir)
    return service.get_location_name(lat, lon)