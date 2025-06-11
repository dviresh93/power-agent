import os
import json
import sqlite3
from typing import Dict
from functools import lru_cache
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

class GeocodingService:
    def __init__(self):
        self.geocoder = Nominatim(user_agent="power_outage_analyzer")
        self.cache_file = "weather_cache.sqlite"
        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize SQLite cache for geocoding results"""
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS geocoding_cache (
                    lat REAL,
                    lon REAL,
                    display_name TEXT,
                    city TEXT,
                    county TEXT,
                    state TEXT,
                    PRIMARY KEY (lat, lon)
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not initialize geocoding cache: {str(e)}")

    @lru_cache(maxsize=1000)
    def get_location_name(self, lat: float, lon: float) -> Dict[str, str]:
        """Get location name from coordinates with caching"""
        try:
            # Try cache first
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT display_name, city, county, state FROM geocoding_cache WHERE lat = ? AND lon = ?",
                (lat, lon)
            )
            result = cursor.fetchone()
            
            if result:
                return {
                    'display_name': result[0],
                    'city': result[1],
                    'county': result[2],
                    'state': result[3]
                }
            
            # If not in cache, geocode
            location = self.geocoder.reverse(f"{lat}, {lon}", exactly_one=True)
            if location:
                address = location.raw.get('address', {})
                result = {
                    'display_name': location.address,
                    'city': address.get('city', address.get('town', address.get('village', ''))),
                    'county': address.get('county', ''),
                    'state': address.get('state', '')
                }
                
                # Cache the result
                cursor.execute("""
                    INSERT OR REPLACE INTO geocoding_cache 
                    (lat, lon, display_name, city, county, state)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (lat, lon, result['display_name'], result['city'], result['county'], result['state']))
                conn.commit()
                
                return result
            
            return {
                'display_name': f"Near {lat:.3f}, {lon:.3f}",
                'city': '',
                'county': '',
                'state': ''
            }
            
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Geocoding error: {str(e)}")
            return {
                'display_name': f"Near {lat:.3f}, {lon:.3f}",
                'city': '',
                'county': '',
                'state': ''
            }
        except Exception as e:
            print(f"Unexpected error in geocoding: {str(e)}")
            return {
                'display_name': f"Near {lat:.3f}, {lon:.3f}",
                'city': '',
                'county': '',
                'state': ''
            }
        finally:
            if 'conn' in locals():
                conn.close() 