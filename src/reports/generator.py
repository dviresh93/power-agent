from typing import Dict, Optional, Tuple, List
import streamlit as st
import folium
import io
import base64
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import json
import os
from pathlib import Path

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ReportLab not available. Install with: pip install reportlab")

# Image processing imports
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. Install with: pip install Pillow")

def save_results_to_pickle(validation_results: Dict, filename: str = None) -> str:
    """Save validation results to pickle file for later loading"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_results_{timestamp}.pkl"
    
    # Ensure the cache directory exists
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    filepath = cache_dir / filename
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(validation_results, f)
        return str(filepath)
    except Exception as e:
        raise Exception(f"Failed to save results to pickle: {str(e)}")

def load_results_from_pickle(filepath: str) -> Dict:
    """Load validation results from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"Failed to load results from pickle: {str(e)}")

def generate_static_map_image(validation_results: Dict, output_path: str = None) -> str:
    """Generate static map image for reports"""
    try:
        real_outages = validation_results.get('real_outages', [])
        false_positives = validation_results.get('false_positives', [])
        
        if not real_outages and not false_positives:
            raise ValueError("No validation results to map")
        
        # Calculate map center
        all_coords = []
        for outage in real_outages + false_positives:
            if 'LATITUDE' in outage and 'LONGITUDE' in outage:
                all_coords.append([outage['LATITUDE'], outage['LONGITUDE']])
            elif 'latitude' in outage and 'longitude' in outage:
                all_coords.append([outage['latitude'], outage['longitude']])
        
        if not all_coords:
            raise ValueError("No coordinate data available")
        
        center_lat = float(np.mean([coord[0] for coord in all_coords]))
        center_lon = float(np.mean([coord[1] for coord in all_coords]))
        
        # Create Folium map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add real outages (red markers)
        for outage in real_outages:
            lat_key = 'LATITUDE' if 'LATITUDE' in outage else 'latitude'
            lon_key = 'LONGITUDE' if 'LONGITUDE' in outage else 'longitude'
            customers_key = 'CUSTOMERS' if 'CUSTOMERS' in outage else 'customers'
            
            if lat_key in outage and lon_key in outage:
                folium.CircleMarker(
                    location=[outage[lat_key], outage[lon_key]],
                    radius=8,
                    popup=f"Real Outage - {outage.get(customers_key, 'Unknown')} customers",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7
                ).add_to(m)
        
        # Add false positives (blue markers)
        for outage in false_positives:
            lat_key = 'LATITUDE' if 'LATITUDE' in outage else 'latitude'
            lon_key = 'LONGITUDE' if 'LONGITUDE' in outage else 'longitude'
            customers_key = 'CUSTOMERS' if 'CUSTOMERS' in outage else 'customers'
            
            if lat_key in outage and lon_key in outage:
                folium.CircleMarker(
                    location=[outage[lat_key], outage[lon_key]],
                    radius=6,
                    popup=f"False Positive - {outage.get(customers_key, 'Unknown')} customers",
                    color='blue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.5
                ).add_to(m)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; top: 10px; right: 10px; z-index: 1000; 
                    background-color: white; border: 2px solid grey; border-radius: 5px;
                    padding: 10px;">
            <h4>Legend</h4>
            <p><span style="color: red;">●</span> Real Outages ({len(real_outages)})</p>
            <p><span style="color: blue;">●</span> False Positives ({len(false_positives)})</p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map as image if possible, otherwise save as HTML
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"cache/outage_map_{timestamp}.html"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as HTML (static image generation from Folium requires additional dependencies)
        m.save(output_path)
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Map generation failed: {str(e)}")

def generate_pdf_report(report_content: str, validation_results: Dict, raw_summary: Dict, output_path: str = None) -> str:
    """Generate comprehensive PDF report"""
    if not REPORTLAB_AVAILABLE:
        raise Exception("ReportLab not available. Install with: pip install reportlab")
    
    try:
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/power_outage_analysis_{timestamp}.pdf"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        story.append(Paragraph("Power Outage Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        
        real_outages = validation_results.get('real_outages', [])
        false_positives = validation_results.get('false_positives', [])
        total_processed = len(real_outages) + len(false_positives)
        accuracy_rate = len(real_outages) / total_processed if total_processed > 0 else 0
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Reports Analyzed', str(total_processed)],
            ['Real Outages Confirmed', str(len(real_outages))],
            ['False Positives Identified', str(len(false_positives))],
            ['Accuracy Rate', f"{accuracy_rate:.1%}"],
            ['Report Generated', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Dataset Information
        if raw_summary:
            story.append(Paragraph("Dataset Information", styles['Heading2']))
            
            dataset_info = f"""Dataset contains {raw_summary.get('total_records', 'unknown')} records spanning from 
            {raw_summary.get('date_range', {}).get('start', 'unknown')} to 
            {raw_summary.get('date_range', {}).get('end', 'unknown')}. 
            Total customers affected: {raw_summary.get('customer_stats', {}).get('total_affected', 'unknown')}."""
            
            story.append(Paragraph(dataset_info, styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Detailed Analysis
        story.append(Paragraph("Detailed Analysis", styles['Heading2']))
        
        # Convert markdown report content to PDF-friendly format
        report_lines = report_content.split('\n')
        for line in report_lines:
            line = line.strip()
            if line.startswith('# '):
                story.append(Paragraph(line[2:], styles['Heading1']))
            elif line.startswith('## '):
                story.append(Paragraph(line[3:], styles['Heading2']))
            elif line.startswith('- '):
                story.append(Paragraph(f"• {line[2:]}", styles['Normal']))
            elif line and not line.startswith('#'):
                story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 6))
        
        # Real Outages Details
        if real_outages:
            story.append(PageBreak())
            story.append(Paragraph("Real Outages - Detailed List", styles['Heading2']))
            
            real_outage_data = [['Location', 'Date/Time', 'Customers', 'Confidence']]
            for i, outage in enumerate(real_outages[:20]):  # Limit to first 20 for PDF size
                lat = outage.get('LATITUDE', outage.get('latitude', 'N/A'))
                lon = outage.get('LONGITUDE', outage.get('longitude', 'N/A'))
                location = f"{lat}, {lon}" if lat != 'N/A' and lon != 'N/A' else 'N/A'
                
                datetime_val = outage.get('DATETIME', outage.get('datetime', 'N/A'))
                customers = outage.get('CUSTOMERS', outage.get('customers', 'N/A'))
                confidence = f"{outage.get('confidence', 0.8):.2f}"
                
                real_outage_data.append([location, str(datetime_val), str(customers), confidence])
            
            if len(real_outages) > 20:
                real_outage_data.append(['...', f'({len(real_outages) - 20} more entries)', '...', '...'])
            
            real_table = Table(real_outage_data)
            real_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(real_table)
        
        # False Positives Details
        if false_positives:
            story.append(Spacer(1, 20))
            story.append(Paragraph("False Positives - Detailed List", styles['Heading2']))
            
            false_data = [['Location', 'Date/Time', 'Customers', 'Confidence']]
            for i, outage in enumerate(false_positives[:20]):  # Limit to first 20
                lat = outage.get('LATITUDE', outage.get('latitude', 'N/A'))
                lon = outage.get('LONGITUDE', outage.get('longitude', 'N/A'))
                location = f"{lat}, {lon}" if lat != 'N/A' and lon != 'N/A' else 'N/A'
                
                datetime_val = outage.get('DATETIME', outage.get('datetime', 'N/A'))
                customers = outage.get('CUSTOMERS', outage.get('customers', 'N/A'))
                confidence = f"{outage.get('confidence', 0.8):.2f}"
                
                false_data.append([location, str(datetime_val), str(customers), confidence])
            
            if len(false_positives) > 20:
                false_data.append(['...', f'({len(false_positives) - 20} more entries)', '...', '...'])
            
            false_table = Table(false_data)
            false_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(false_table)
        
        # Build PDF
        doc.build(story)
        
        return output_path
        
    except Exception as e:
        raise Exception(f"PDF generation failed: {str(e)}")

def generate_and_download_report(validation_results: Dict, raw_summary: Dict = None, report_mode: str = "Default", format_type: str = "pdf") -> Dict:
    """Generate and prepare report for download"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate report content
        report_content = generate_comprehensive_report_content(validation_results, raw_summary)
        
        files_generated = {}
        
        if format_type in ["pdf", "all"]:
            # Generate PDF report
            pdf_path = generate_pdf_report(report_content, validation_results, raw_summary)
            files_generated["pdf"] = pdf_path
        
        if format_type in ["map", "all"]:
            # Generate static map
            map_path = generate_static_map_image(validation_results)
            files_generated["map"] = map_path
        
        if format_type in ["pickle", "all"]:
            # Save results as pickle
            pickle_path = save_results_to_pickle(validation_results)
            files_generated["pickle"] = pickle_path
        
        if format_type in ["json", "all"]:
            # Save as JSON
            json_path = f"reports/validation_results_{timestamp}.json"
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            files_generated["json"] = json_path
        
        return {
            "status": "success",
            "files_generated": files_generated,
            "report_content": report_content,
            "timestamp": timestamp
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "files_generated": {},
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }

def generate_map_data_summary(validation_results: Dict) -> Dict:
    """Generate comprehensive map data summary"""
    try:
        real_outages = validation_results.get('real_outages', [])
        false_positives = validation_results.get('false_positives', [])
        
        # Geographic analysis
        all_real_coords = []
        all_false_coords = []
        
        for outage in real_outages:
            lat_key = 'LATITUDE' if 'LATITUDE' in outage else 'latitude'
            lon_key = 'LONGITUDE' if 'LONGITUDE' in outage else 'longitude'
            if lat_key in outage and lon_key in outage:
                all_real_coords.append([outage[lat_key], outage[lon_key]])
        
        for outage in false_positives:
            lat_key = 'LATITUDE' if 'LATITUDE' in outage else 'latitude'
            lon_key = 'LONGITUDE' if 'LONGITUDE' in outage else 'longitude'
            if lat_key in outage and lon_key in outage:
                all_false_coords.append([outage[lat_key], outage[lon_key]])
        
        all_coords = all_real_coords + all_false_coords
        
        if not all_coords:
            return {"status": "no_data", "message": "No coordinate data available"}
        
        # Calculate geographic bounds and center
        center_lat = float(np.mean([coord[0] for coord in all_coords]))
        center_lon = float(np.mean([coord[1] for coord in all_coords]))
        
        bounds = {
            "north": float(max([coord[0] for coord in all_coords])),
            "south": float(min([coord[0] for coord in all_coords])),
            "east": float(max([coord[1] for coord in all_coords])),
            "west": float(min([coord[1] for coord in all_coords]))
        }
        
        # Calculate geographic spread
        lat_spread = bounds["north"] - bounds["south"]
        lon_spread = bounds["east"] - bounds["west"]
        
        # Geographic clustering analysis
        real_center = None
        false_center = None
        
        if all_real_coords:
            real_center = {
                "lat": float(np.mean([coord[0] for coord in all_real_coords])),
                "lon": float(np.mean([coord[1] for coord in all_real_coords]))
            }
        
        if all_false_coords:
            false_center = {
                "lat": float(np.mean([coord[0] for coord in all_false_coords])),
                "lon": float(np.mean([coord[1] for coord in all_false_coords]))
            }
        
        return {
            "status": "success",
            "center": {"lat": center_lat, "lon": center_lon},
            "bounds": bounds,
            "geographic_spread": {"lat_spread": lat_spread, "lon_spread": lon_spread},
            "marker_counts": {
                "real_outages": len(real_outages),
                "false_positives": len(false_positives),
                "total": len(real_outages) + len(false_positives)
            },
            "cluster_analysis": {
                "real_outages_center": real_center,
                "false_positives_center": false_center,
                "coordinates_available": {
                    "real_outages": len(all_real_coords),
                    "false_positives": len(all_false_coords)
                }
            }
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

def generate_map_section_for_report(validation_results: Dict) -> str:
    """Generate map section content for reports"""
    try:
        map_summary = generate_map_data_summary(validation_results)
        
        if map_summary.get("status") != "success":
            return "## Geographic Analysis\n\nNo geographic data available for mapping.\n"
        
        real_count = map_summary["marker_counts"]["real_outages"]
        false_count = map_summary["marker_counts"]["false_positives"]
        total_count = map_summary["marker_counts"]["total"]
        
        center = map_summary["center"]
        bounds = map_summary["bounds"]
        spread = map_summary["geographic_spread"]
        
        section = f"""## Geographic Analysis

### Location Summary
- **Total Mapped Locations**: {total_count}
- **Real Outages**: {real_count} (marked in red)
- **False Positives**: {false_count} (marked in blue)
- **Map Center**: {center['lat']:.4f}, {center['lon']:.4f}

### Geographic Distribution
- **Latitude Range**: {bounds['south']:.4f} to {bounds['north']:.4f} (spread: {spread['lat_spread']:.4f}°)
- **Longitude Range**: {bounds['west']:.4f} to {bounds['east']:.4f} (spread: {spread['lon_spread']:.4f}°)

### Clustering Analysis
"""
        
        cluster_info = map_summary.get("cluster_analysis", {})
        
        if cluster_info.get("real_outages_center"):
            real_center = cluster_info["real_outages_center"]
            section += f"- **Real Outages Cluster Center**: {real_center['lat']:.4f}, {real_center['lon']:.4f}\n"
        
        if cluster_info.get("false_positives_center"):
            false_center = cluster_info["false_positives_center"]
            section += f"- **False Positives Cluster Center**: {false_center['lat']:.4f}, {false_center['lon']:.4f}\n"
        
        # Add distance analysis if both centers exist
        if (cluster_info.get("real_outages_center") and cluster_info.get("false_positives_center")):
            real_center = cluster_info["real_outages_center"]
            false_center = cluster_info["false_positives_center"]
            
            # Simple distance calculation (not accounting for Earth's curvature, but good enough for reports)
            lat_diff = abs(real_center['lat'] - false_center['lat'])
            lon_diff = abs(real_center['lon'] - false_center['lon'])
            distance_approx = (lat_diff**2 + lon_diff**2)**0.5
            
            section += f"- **Cluster Separation**: Approximately {distance_approx:.4f}° geographic distance\n"
        
        section += "\n### Map Legend\n- 🔴 **Red Markers**: Confirmed real outages\n- 🔵 **Blue Markers**: Identified false positives\n"
        
        return section
        
    except Exception as e:
        return f"## Geographic Analysis\n\nError generating map analysis: {str(e)}\n"

def generate_comprehensive_report_content(validation_results: Dict, raw_summary: Dict = None) -> str:
    """Generate comprehensive markdown report content"""
    try:
        real_outages = validation_results.get('real_outages', [])
        false_positives = validation_results.get('false_positives', [])
        total_processed = len(real_outages) + len(false_positives)
        
        accuracy_rate = len(real_outages) / total_processed if total_processed > 0 else 0
        
        # Calculate confidence statistics
        real_confidences = [r.get('confidence', 0.8) for r in real_outages if 'confidence' in r]
        false_confidences = [f.get('confidence', 0.8) for f in false_positives if 'confidence' in f]
        
        real_avg_conf = np.mean(real_confidences) if real_confidences else 0.8
        false_avg_conf = np.mean(false_confidences) if false_confidences else 0.8
        
        # Calculate customer impact
        real_customers = sum([r.get('CUSTOMERS', r.get('customers', 0)) for r in real_outages])
        false_customers = sum([f.get('CUSTOMERS', f.get('customers', 0)) for f in false_positives])
        
        report = f"""# Power Outage Analysis Report

## Executive Summary

- **Total reports analyzed**: {total_processed}
- **Real outages confirmed**: {len(real_outages)}
- **False positives identified**: {len(false_positives)}
- **Overall accuracy rate**: {accuracy_rate:.1%}
- **Report generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Classification Performance

### Real Outages
- **Count**: {len(real_outages)}
- **Average Confidence**: {real_avg_conf:.2f}
- **Total Customers Affected**: {real_customers:,}
- **Average Customers per Outage**: {real_customers/len(real_outages):.0f} (if applicable)

### False Positives
- **Count**: {len(false_positives)}
- **Average Confidence**: {false_avg_conf:.2f}
- **Total Falsely Reported Customers**: {false_customers:,}
- **Average Customers per False Report**: {false_customers/len(false_positives):.0f} (if applicable)

## Dataset Information
"""
        
        if raw_summary:
            report += f"""- **Total records in dataset**: {raw_summary.get('total_records', 'Unknown')}
- **Date range**: {raw_summary.get('date_range', {}).get('start', 'Unknown')} to {raw_summary.get('date_range', {}).get('end', 'Unknown')}
- **Geographic coverage**: 
  - Latitude: {raw_summary.get('geographic_bounds', {}).get('lat_min', 'N/A')} to {raw_summary.get('geographic_bounds', {}).get('lat_max', 'N/A')}
  - Longitude: {raw_summary.get('geographic_bounds', {}).get('lon_min', 'N/A')} to {raw_summary.get('geographic_bounds', {}).get('lon_max', 'N/A')}
- **Total customers in dataset**: {raw_summary.get('customer_stats', {}).get('total_affected', 'Unknown'):,}
"""
        else:
            report += "Dataset information not available.\n"
        
        # Add geographic analysis
        report += "\n" + generate_map_section_for_report(validation_results)
        
        report += f"""

## Detailed Analysis

### High-Confidence Classifications
- **Real outages with >80% confidence**: {len([r for r in real_confidences if r > 0.8])}
- **False positives with >80% confidence**: {len([f for f in false_confidences if f > 0.8])}

### Recommendations

#### Operational Improvements
"""
        
        # Add context-specific recommendations
        if accuracy_rate > 0.9:
            report += "- **Excellent Performance**: The system is performing exceptionally well with >90% accuracy.\n"
        elif accuracy_rate > 0.8:
            report += "- **Good Performance**: The system shows good accuracy. Consider minor optimizations.\n"
        else:
            report += "- **Performance Needs Improvement**: Consider reviewing classification criteria and weather thresholds.\n"
        
        if len(false_positives) > len(real_outages):
            report += "- **High False Positive Rate**: Review sensor calibration and weather correlation algorithms.\n"
        
        if false_avg_conf > real_avg_conf:
            report += "- **Confidence Calibration**: False positives show higher confidence than real outages - review classification logic.\n"
        
        report += """- Continue monitoring weather patterns for outage correlation
- Review sensor calibration for areas with high false positive rates  
- Implement additional validation for reports during mild weather conditions
- Consider geographic clustering patterns for predictive maintenance

### Technical Notes
- Classification based on historical weather data correlation
- Confidence scores reflect model certainty in classification decisions
- Geographic analysis helps identify regional patterns and equipment issues

---

*This report was generated by the LLM-Powered Outage Analysis Agent*
"""
        
        return report
        
    except Exception as e:
        return f"# Power Outage Analysis Report\n\nError generating report: {str(e)}\n" 