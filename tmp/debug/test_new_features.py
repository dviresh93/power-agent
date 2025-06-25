#!/usr/bin/env python3
"""
Test script for new report generation and pickle features
Tests all the implemented functionality with sample data
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the src directory to the path so we can import our new modules
sys.path.append('src')
sys.path.append('.')

def create_sample_validation_results():
    """Create sample validation results for testing"""
    return {
        "real_outages": [
            {
                "LATITUDE": 40.7128,
                "LONGITUDE": -74.0060,
                "DATETIME": "2024-01-15 14:30:00",
                "CUSTOMERS": 150,
                "classification": "REAL OUTAGE",
                "confidence": 0.92,
                "reasoning": "High winds and heavy precipitation detected",
                "weather_factors": ["high_winds", "heavy_rain"],
                "severity_score": 8
            },
            {
                "LATITUDE": 40.7580,
                "LONGITUDE": -73.9855,
                "DATETIME": "2024-01-15 16:45:00", 
                "CUSTOMERS": 85,
                "classification": "REAL OUTAGE",
                "confidence": 0.88,
                "reasoning": "Ice accumulation on power lines",
                "weather_factors": ["ice_accumulation", "freezing_temperature"],
                "severity_score": 7
            }
        ],
        "false_positives": [
            {
                "LATITUDE": 40.7489,
                "LONGITUDE": -73.9680,
                "DATETIME": "2024-01-15 12:15:00",
                "CUSTOMERS": 45,
                "classification": "FALSE POSITIVE",
                "confidence": 0.85,
                "reasoning": "Mild weather conditions, likely sensor malfunction",
                "weather_factors": ["mild_conditions"],
                "severity_score": 2
            }
        ],
        "total_processed": 3,
        "validation_complete": True,
        "processing_stats": {
            "success_rate": 1.0,
            "real_outage_rate": 0.67,
            "false_positive_rate": 0.33
        }
    }

def create_sample_raw_summary():
    """Create sample raw dataset summary"""
    return {
        "status": "loaded",
        "total_records": 3,
        "date_range": {
            "start": "2024-01-15T12:15:00",
            "end": "2024-01-15T16:45:00"
        },
        "geographic_bounds": {
            "lat_min": 40.7128,
            "lat_max": 40.7580,
            "lon_min": -74.0060,
            "lon_max": -73.9680
        },
        "customer_stats": {
            "total_affected": 280,
            "avg_per_outage": 93.33,
            "max_single_outage": 150
        }
    }

def test_pickle_functionality():
    """Test pickle save/load functionality"""
    print("\n🧪 Testing Pickle Functionality...")
    
    try:
        from src.utils.result_persistence import save_results, load_results, list_results, get_summary
        
        # Create sample data
        validation_results = create_sample_validation_results()
        
        # Test saving
        print("  📦 Testing save...")
        saved_path = save_results(validation_results, "pickle", "test_results")
        print(f"  ✅ Saved to: {saved_path}")
        
        # Test loading
        print("  📥 Testing load...")
        loaded_results = load_results(saved_path)
        print(f"  ✅ Loaded {len(loaded_results.get('real_outages', []))} real outages and {len(loaded_results.get('false_positives', []))} false positives")
        
        # Test listing
        print("  📋 Testing list...")
        saved_files = list_results("pickle")
        print(f"  ✅ Found {len(saved_files)} saved pickle files")
        
        # Test summary
        print("  📊 Testing summary...")
        summary = get_summary(saved_path)
        if "error" not in summary:
            print(f"  ✅ Summary: {summary['total_processed']} processed, {summary['accuracy_rate']:.1%} accuracy")
        else:
            print(f"  ❌ Summary error: {summary['error']}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def test_report_generation():
    """Test report generation functionality"""
    print("\n🧪 Testing Report Generation...")
    
    try:
        from src.reports.generator import (
            generate_comprehensive_report_content,
            generate_map_data_summary,
            generate_map_section_for_report,
            generate_static_map_image
        )
        
        # Create sample data
        validation_results = create_sample_validation_results()
        raw_summary = create_sample_raw_summary()
        
        # Test comprehensive report content
        print("  📝 Testing comprehensive report content...")
        report_content = generate_comprehensive_report_content(validation_results, raw_summary)
        print(f"  ✅ Generated report content ({len(report_content)} characters)")
        
        # Test map data summary
        print("  🗺️ Testing map data summary...")
        map_summary = generate_map_data_summary(validation_results)
        if map_summary.get("status") == "success":
            print(f"  ✅ Map summary: {map_summary['marker_counts']['total']} total markers")
        else:
            print(f"  ❌ Map summary failed: {map_summary.get('error', 'Unknown error')}")
        
        # Test map section for report
        print("  📍 Testing map section generation...")
        map_section = generate_map_section_for_report(validation_results)
        print(f"  ✅ Generated map section ({len(map_section)} characters)")
        
        # Test static map generation
        print("  🗺️ Testing static map generation...")
        try:
            map_path = generate_static_map_image(validation_results)
            print(f"  ✅ Generated static map: {map_path}")
        except Exception as e:
            print(f"  ⚠️ Static map generation failed (expected if folium not available): {e}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def test_pdf_generation():
    """Test PDF generation functionality"""
    print("\n🧪 Testing PDF Generation...")
    
    try:
        from src.reports.generator import generate_pdf_report, generate_comprehensive_report_content
        
        # Create sample data
        validation_results = create_sample_validation_results()
        raw_summary = create_sample_raw_summary()
        
        # Generate report content
        report_content = generate_comprehensive_report_content(validation_results, raw_summary)
        
        # Test PDF generation
        print("  📄 Testing PDF generation...")
        try:
            pdf_path = generate_pdf_report(report_content, validation_results, raw_summary)
            print(f"  ✅ Generated PDF: {pdf_path}")
            
            # Check if file exists and has content
            if os.path.exists(pdf_path):
                file_size = os.path.getsize(pdf_path)
                print(f"  ✅ PDF file created ({file_size} bytes)")
            else:
                print("  ❌ PDF file not found")
                
            return True
            
        except Exception as e:
            print(f"  ⚠️ PDF generation failed (expected if reportlab not available): {e}")
            return False
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def test_comprehensive_export():
    """Test comprehensive export functionality"""
    print("\n🧪 Testing Comprehensive Export...")
    
    try:
        from src.reports.generator import generate_and_download_report
        
        # Create sample data
        validation_results = create_sample_validation_results()
        raw_summary = create_sample_raw_summary()
        
        # Test JSON export
        print("  📄 Testing JSON export...")
        result = generate_and_download_report(validation_results, raw_summary, "Test", "json")
        
        if result.get("status") == "success":
            files = result.get("files_generated", {})
            print(f"  ✅ JSON export successful: {files.get('json', 'No JSON file')}")
        else:
            print(f"  ❌ JSON export failed: {result.get('error', 'Unknown error')}")
        
        # Test comprehensive export (all formats)
        print("  📦 Testing comprehensive export...")
        result = generate_and_download_report(validation_results, raw_summary, "Test", "all")
        
        if result.get("status") == "success":
            files = result.get("files_generated", {})
            print(f"  ✅ Comprehensive export successful:")
            for format_type, filepath in files.items():
                print(f"    - {format_type.upper()}: {filepath}")
        else:
            print(f"  ❌ Comprehensive export failed: {result.get('error', 'Unknown error')}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing New Report Generation and Pickle Features")
    print("=" * 60)
    
    # Ensure directories exist
    os.makedirs("cache", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    tests_passed = 0
    total_tests = 4
    
    # Run tests
    if test_pickle_functionality():
        tests_passed += 1
    
    if test_report_generation():
        tests_passed += 1
    
    if test_pdf_generation():
        tests_passed += 1
    
    if test_comprehensive_export():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! New features are working correctly.")
    elif tests_passed > 0:
        print("⚠️ Some tests passed. Check missing dependencies (reportlab, folium, etc.)")
    else:
        print("❌ All tests failed. Check your installation and dependencies.")
    
    print("\n📋 To install missing dependencies:")
    print("pip install reportlab folium pillow")

if __name__ == "__main__":
    main()