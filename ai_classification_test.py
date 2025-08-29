#!/usr/bin/env python3
"""
AI Classification Deep Test for Ocean Hazard Alert System
Tests the AI classification functionality with various scenarios
"""

import requests
import json
import time

BACKEND_URL = "https://ocean-hazard-alert.preview.emergentagent.com/api"

def test_ai_classification_scenarios():
    """Test AI classification with different severity scenarios"""
    
    test_scenarios = [
        {
            "name": "Low Severity Scenario",
            "hazard_type": "Other",
            "description": "Minor debris spotted floating near the shore. No immediate danger to vessels.",
            "expected_severity_range": ["Low", "Medium"]
        },
        {
            "name": "Medium Severity Scenario", 
            "hazard_type": "Oil Spill",
            "description": "Small oil leak detected from fishing vessel. Contained area approximately 50 meters.",
            "expected_severity_range": ["Medium", "High"]
        },
        {
            "name": "High Severity Scenario",
            "hazard_type": "Tsunami",
            "description": "URGENT: Massive tsunami waves approaching coastline. Immediate evacuation required. Wave height estimated 15+ meters.",
            "expected_severity_range": ["High"]
        },
        {
            "name": "Cyclone Emergency",
            "hazard_type": "Cyclone", 
            "description": "Category 5 cyclone with winds exceeding 200 km/h. Extreme danger to all maritime activities. Storm surge expected.",
            "expected_severity_range": ["High"]
        }
    ]
    
    print("=" * 60)
    print("AI CLASSIFICATION DEEP TEST")
    print("=" * 60)
    
    session = requests.Session()
    created_ids = []
    
    for scenario in test_scenarios:
        print(f"\nTesting: {scenario['name']}")
        print(f"Description: {scenario['description'][:60]}...")
        
        try:
            # Create report
            test_data = {
                "name": f"AI Test Reporter",
                "latitude": -33.8688,
                "longitude": 151.2093,
                "address": "Test Location",
                "hazard_type": scenario["hazard_type"],
                "description": scenario["description"]
            }
            
            response = session.post(f"{BACKEND_URL}/reports", data=test_data)
            
            if response.status_code == 200:
                report = response.json()
                created_ids.append(report["id"])
                
                severity = report["severity"]
                panic_index = report["panic_index"]
                ai_category = report["ai_category"]
                
                # Check if severity is in expected range
                severity_ok = severity in scenario["expected_severity_range"]
                panic_ok = 0 <= panic_index <= 100
                
                status = "✅ PASS" if (severity_ok and panic_ok) else "❌ FAIL"
                print(f"{status} Severity: {severity}, Panic Index: {panic_index}, AI Category: {ai_category}")
                
                if not severity_ok:
                    print(f"   ⚠️  Expected severity in {scenario['expected_severity_range']}, got {severity}")
                
            else:
                print(f"❌ FAIL Request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ FAIL Exception: {str(e)}")
    
    # Cleanup
    print(f"\nCleaning up {len(created_ids)} test reports...")
    for report_id in created_ids:
        try:
            session.delete(f"{BACKEND_URL}/reports/{report_id}")
        except:
            pass
    
    print("AI Classification test completed.")

if __name__ == "__main__":
    test_ai_classification_scenarios()