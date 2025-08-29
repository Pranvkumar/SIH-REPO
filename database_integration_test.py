#!/usr/bin/env python3
"""
Database Integration Test for Ocean Hazard Alert System
Tests MongoDB integration and data persistence
"""

import requests
import json
import time

BACKEND_URL = "https://ocean-hazard-alert.preview.emergentagent.com/api"

def test_database_integration():
    """Test database operations and data persistence"""
    
    print("=" * 60)
    print("DATABASE INTEGRATION TEST")
    print("=" * 60)
    
    session = requests.Session()
    
    # Test 1: Create report and verify persistence
    print("\n1. Testing data persistence...")
    
    test_data = {
        "name": "Database Test Reporter",
        "latitude": -34.9285,
        "longitude": 138.6007,  # Adelaide coordinates
        "address": "Adelaide, SA, Australia",
        "hazard_type": "Flood",
        "description": "Testing database persistence with flood report in Adelaide coastal area."
    }
    
    # Create report
    response = session.post(f"{BACKEND_URL}/reports", data=test_data)
    if response.status_code == 200:
        report = response.json()
        report_id = report["id"]
        print(f"✅ Created report with ID: {report_id}")
        
        # Verify it appears in all reports
        time.sleep(1)  # Small delay for consistency
        all_reports = session.get(f"{BACKEND_URL}/reports").json()
        found_report = next((r for r in all_reports if r["id"] == report_id), None)
        
        if found_report:
            print("✅ Report found in all reports list")
            
            # Verify data integrity
            if (found_report["name"] == test_data["name"] and 
                found_report["hazard_type"] == test_data["hazard_type"] and
                found_report["description"] == test_data["description"]):
                print("✅ Data integrity verified")
            else:
                print("❌ Data integrity check failed")
        else:
            print("❌ Report not found in all reports list")
        
        # Test 2: Verify report appears in priority list
        print("\n2. Testing priority list integration...")
        priority_reports = session.get(f"{BACKEND_URL}/reports/priority").json()
        found_in_priority = any(pr["report"]["id"] == report_id for pr in priority_reports)
        
        if found_in_priority:
            print("✅ Report found in priority list")
        else:
            print("❌ Report not found in priority list")
        
        # Test 3: Verify report appears in heatmap data
        print("\n3. Testing heatmap data integration...")
        heatmap_data = session.get(f"{BACKEND_URL}/reports/heatmap").json()
        heatmap_points = heatmap_data.get("heatmap_data", [])
        found_in_heatmap = any(
            abs(point["lat"] - test_data["latitude"]) < 0.001 and 
            abs(point["lng"] - test_data["longitude"]) < 0.001 
            for point in heatmap_points
        )
        
        if found_in_heatmap:
            print("✅ Report location found in heatmap data")
        else:
            print("❌ Report location not found in heatmap data")
        
        # Test 4: Verify dashboard stats update
        print("\n4. Testing dashboard stats integration...")
        stats_before = session.get(f"{BACKEND_URL}/dashboard/stats").json()
        
        # Create another report to see stats change
        test_data2 = {
            "name": "Second Test Reporter",
            "latitude": -37.8136,
            "longitude": 144.9631,  # Melbourne coordinates
            "address": "Melbourne, VIC, Australia", 
            "hazard_type": "Cyclone",
            "description": "Second test report for stats verification."
        }
        
        response2 = session.post(f"{BACKEND_URL}/reports", data=test_data2)
        if response2.status_code == 200:
            report2 = response2.json()
            report2_id = report2["id"]
            
            time.sleep(1)  # Allow for processing
            stats_after = session.get(f"{BACKEND_URL}/dashboard/stats").json()
            
            if stats_after["total_reports"] > stats_before["total_reports"]:
                print("✅ Dashboard stats updated correctly")
            else:
                print("❌ Dashboard stats not updated")
            
            # Test 5: Test deletion and verify cleanup
            print("\n5. Testing deletion and cleanup...")
            
            # Delete both reports
            delete1 = session.delete(f"{BACKEND_URL}/reports/{report_id}")
            delete2 = session.delete(f"{BACKEND_URL}/reports/{report2_id}")
            
            if delete1.status_code == 200 and delete2.status_code == 200:
                print("✅ Reports deleted successfully")
                
                # Verify they're gone from all endpoints
                time.sleep(1)
                final_reports = session.get(f"{BACKEND_URL}/reports").json()
                final_priority = session.get(f"{BACKEND_URL}/reports/priority").json()
                final_heatmap = session.get(f"{BACKEND_URL}/reports/heatmap").json()
                
                reports_gone = not any(r["id"] in [report_id, report2_id] for r in final_reports)
                priority_gone = not any(pr["report"]["id"] in [report_id, report2_id] for pr in final_priority)
                
                if reports_gone and priority_gone:
                    print("✅ Reports properly removed from all endpoints")
                else:
                    print("❌ Reports still found in some endpoints after deletion")
            else:
                print("❌ Failed to delete reports")
        else:
            print("❌ Failed to create second test report")
    else:
        print(f"❌ Failed to create initial test report: {response.status_code}")
    
    print("\nDatabase integration test completed.")

if __name__ == "__main__":
    test_database_integration()