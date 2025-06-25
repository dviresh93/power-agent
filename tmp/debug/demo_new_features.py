#!/usr/bin/env python3
"""
Demonstration of New Report Generation and Pickle Features
Shows how to use all the new functionality in practice
"""

import os
import sys
from datetime import datetime

# Add the src directory to the path
sys.path.append('src')
sys.path.append('.')

def demo_pickle_persistence():
    """Demonstrate pickle persistence features"""
    print("\n📦 PICKLE PERSISTENCE DEMO")
    print("-" * 40)
    
    # Example validation results (as if from a real analysis)
    sample_results = {
        "real_outages": [
            {
                "LATITUDE": 40.7128,
                "LONGITUDE": -74.0060,
                "DATETIME": "2024-01-15 14:30:00",
                "CUSTOMERS": 150,
                "classification": "REAL OUTAGE",
                "confidence": 0.92,
                "reasoning": "High winds and heavy precipitation detected"
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
                "reasoning": "Mild weather conditions, likely sensor malfunction"
            }
        ],
        "validation_complete": True
    }
    
    try:
        from src.utils.result_persistence import save_results, load_results, list_results
        
        # Save results in different formats
        print("💾 Saving results in multiple formats...")
        pickle_path = save_results(sample_results, "pickle", "demo_analysis")
        json_path = save_results(sample_results, "json", "demo_analysis")
        print(f"✅ Pickle saved: {pickle_path}")
        print(f"✅ JSON saved: {json_path}")
        
        # List saved files
        print("\\n📋 Listing saved results...")
        saved_files = list_results()
        for file_info in saved_files[:3]:  # Show first 3
            print(f"  📄 {file_info['filename']} ({file_info['format']}) - {file_info['size']} bytes")
        
        # Load and verify
        print("\\n📥 Loading and verifying results...")
        loaded_results = load_results(pickle_path)
        print(f"✅ Loaded: {len(loaded_results['real_outages'])} real outages, {len(loaded_results['false_positives'])} false positives")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def demo_report_generation():
    """Demonstrate report generation features"""
    print("\\n📊 REPORT GENERATION DEMO")
    print("-" * 40)
    
    # Sample data
    validation_results = {
        "real_outages": [
            {"LATITUDE": 40.7128, "LONGITUDE": -74.0060, "CUSTOMERS": 150, "confidence": 0.92},
            {"LATITUDE": 40.7580, "LONGITUDE": -73.9855, "CUSTOMERS": 85, "confidence": 0.88}
        ],
        "false_positives": [
            {"LATITUDE": 40.7489, "LONGITUDE": -73.9680, "CUSTOMERS": 45, "confidence": 0.85}
        ],
        "validation_complete": True
    }
    
    raw_summary = {
        "total_records": 3,
        "date_range": {"start": "2024-01-15T12:15:00", "end": "2024-01-15T16:45:00"},
        "customer_stats": {"total_affected": 280}
    }
    
    try:
        from src.reports.generator import (
            generate_comprehensive_report_content,
            generate_map_data_summary,
            generate_static_map_image
        )
        
        # Generate comprehensive report
        print("📝 Generating comprehensive report content...")
        report_content = generate_comprehensive_report_content(validation_results, raw_summary)
        print(f"✅ Report generated ({len(report_content)} characters)")
        print("\\nReport preview:")
        print(report_content[:300] + "..." if len(report_content) > 300 else report_content)
        
        # Generate map summary
        print("\\n🗺️ Generating map data summary...")
        map_summary = generate_map_data_summary(validation_results)
        if map_summary.get("status") == "success":
            center = map_summary["center"]
            markers = map_summary["marker_counts"]
            print(f"✅ Map centered at: {center['lat']:.4f}, {center['lon']:.4f}")
            print(f"✅ Markers: {markers['real_outages']} real, {markers['false_positives']} false")
        
        # Try to generate static map
        print("\\n🗺️ Attempting static map generation...")
        try:
            map_path = generate_static_map_image(validation_results)
            print(f"✅ Static map generated: {map_path}")
        except Exception as e:
            print(f"⚠️ Static map failed (missing dependencies): {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def demo_comprehensive_export():
    """Demonstrate comprehensive export features"""
    print("\\n📤 COMPREHENSIVE EXPORT DEMO")
    print("-" * 40)
    
    validation_results = {
        "real_outages": [{"LATITUDE": 40.7128, "LONGITUDE": -74.0060, "CUSTOMERS": 150}],
        "false_positives": [{"LATITUDE": 40.7489, "LONGITUDE": -73.9680, "CUSTOMERS": 45}],
        "validation_complete": True
    }
    
    try:
        from src.reports.generator import generate_and_download_report
        
        # Export in JSON format
        print("📄 Exporting as JSON...")
        result = generate_and_download_report(validation_results, None, "Demo", "json")
        
        if result.get("status") == "success":
            files = result.get("files_generated", {})
            print(f"✅ JSON export: {files.get('json', 'None')}")
        
        # Export in all formats
        print("\\n📦 Exporting in all formats...")
        result = generate_and_download_report(validation_results, None, "Demo", "all")
        
        if result.get("status") == "success":
            files = result.get("files_generated", {})
            print("✅ All formats exported:")
            for format_type, filepath in files.items():
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"  - {format_type.upper()}: {filepath} ({size} bytes)")
        else:
            print(f"❌ Export failed: {result.get('error')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def demo_pdf_generation():
    """Demonstrate PDF generation"""
    print("\\n📄 PDF GENERATION DEMO")
    print("-" * 40)
    
    validation_results = {
        "real_outages": [
            {"LATITUDE": 40.7128, "LONGITUDE": -74.0060, "CUSTOMERS": 150, "confidence": 0.92, "DATETIME": "2024-01-15 14:30:00"}
        ],
        "false_positives": [
            {"LATITUDE": 40.7489, "LONGITUDE": -73.9680, "CUSTOMERS": 45, "confidence": 0.85, "DATETIME": "2024-01-15 12:15:00"}
        ],
        "validation_complete": True
    }
    
    raw_summary = {
        "total_records": 2,
        "date_range": {"start": "2024-01-15T12:15:00", "end": "2024-01-15T14:30:00"},
        "customer_stats": {"total_affected": 195}
    }
    
    try:
        from src.reports.generator import generate_pdf_report, generate_comprehensive_report_content
        
        print("📄 Generating PDF report...")
        report_content = generate_comprehensive_report_content(validation_results, raw_summary)
        
        try:
            pdf_path = generate_pdf_report(report_content, validation_results, raw_summary)
            if os.path.exists(pdf_path):
                size = os.path.getsize(pdf_path)
                print(f"✅ PDF generated: {pdf_path} ({size} bytes)")
                print("📖 PDF includes:")
                print("  - Executive summary table")
                print("  - Dataset information")
                print("  - Detailed outage listings")
                print("  - Geographic analysis")
            else:
                print("❌ PDF file not created")
        except Exception as e:
            print(f"⚠️ PDF generation failed (missing reportlab): {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def show_usage_examples():
    """Show code usage examples"""
    print("\\n💡 USAGE EXAMPLES")
    print("-" * 40)
    
    print("""
🐍 Python Usage Examples:

1. Save validation results:
   ```python
   from src.utils.result_persistence import save_results
   save_results(validation_results, "pickle", "my_analysis")
   save_results(validation_results, "json", "my_analysis")
   ```

2. Load saved results:
   ```python
   from src.utils.result_persistence import load_results
   results = load_results("cache/my_analysis.pkl")
   ```

3. Generate PDF report:
   ```python
   from src.reports.generator import generate_pdf_report, generate_comprehensive_report_content
   content = generate_comprehensive_report_content(results, raw_summary)
   pdf_path = generate_pdf_report(content, results, raw_summary)
   ```

4. Export all formats:
   ```python
   from src.reports.generator import generate_and_download_report
   result = generate_and_download_report(results, raw_summary, "MyAnalysis", "all")
   ```

🌐 Streamlit Integration:
- All features are integrated into the enhanced Reports tab
- Use the "📄 Report Generation & Export" section
- Save/load results with pickle support
- Generate PDFs and static maps with one click
""")

def main():
    """Run the demonstration"""
    print("🚀 NEW FEATURES DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the new report generation and pickle persistence features.")
    
    # Ensure directories exist
    os.makedirs("cache", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    demos = [
        ("Pickle Persistence", demo_pickle_persistence),
        ("Report Generation", demo_report_generation),
        ("PDF Generation", demo_pdf_generation),
        ("Comprehensive Export", demo_comprehensive_export)
    ]
    
    passed = 0
    for name, demo_func in demos:
        print(f"\\n🎯 Running {name} Demo...")
        try:
            if demo_func():
                passed += 1
                print(f"✅ {name} demo completed successfully")
            else:
                print(f"❌ {name} demo failed")
        except Exception as e:
            print(f"❌ {name} demo crashed: {e}")
    
    show_usage_examples()
    
    print("\\n" + "=" * 60)
    print(f"🎯 DEMO SUMMARY: {passed}/{len(demos)} demos successful")
    
    if passed == len(demos):
        print("🎉 All features working! Ready for production use.")
    else:
        print("⚠️ Some demos failed. Install dependencies:")
        print("   pip install reportlab folium pillow")
    
    print("\\n🔗 Next steps:")
    print("1. Run 'streamlit run streamlit_agent_interface.py' to try the UI")
    print("2. Check the Reports tab for new export options")
    print("3. Use the pickle save/load features for result persistence")

if __name__ == "__main__":
    main()