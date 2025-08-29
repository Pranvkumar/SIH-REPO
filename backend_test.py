#!/usr/bin/env python3
"""
Ocean Hazard Alert System Backend API Test Suite
Tests all backend endpoints and integrations
"""

import requests
import json
import base64
import os
from datetime import datetime
import time

# Get backend URL from environment
BACKEND_URL = "https://ocean-hazard-alert.preview.emergentagent.com/api"

class OceanHazardAPITester:
    def __init__(self):
        self.base_url = BACKEND_URL
        self.session = requests.Session()
        self.created_report_ids = []
        
    def log_test(self, test_name, success, details=""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   Details: {details}")
        print()
        
    def test_health_endpoint(self):
        """Test GET /api/health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_test("Health Check", True, f"Service: {data.get('service')}")
                    return True
                else:
                    self.log_test("Health Check", False, f"Unexpected response: {data}")
                    return False
            else:
                self.log_test("Health Check", False, f"Status code: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False
    
    def create_sample_media_file(self):
        """Create a sample base64 encoded image for testing"""
        # Simple 1x1 pixel PNG in base64
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    def test_create_hazard_report(self, hazard_type="Cyclone", include_media=False):
        """Test POST /api/reports endpoint"""
        try:
            # Prepare test data
            test_data = {
                "name": f"Emergency Reporter {datetime.now().strftime('%H%M%S')}",
                "latitude": -33.8688,  # Sydney coordinates
                "longitude": 151.2093,
                "address": "Sydney Harbour, NSW, Australia",
                "hazard_type": hazard_type,
                "description": f"Severe {hazard_type.lower()} detected in coastal waters. High winds and dangerous conditions observed. Immediate attention required for maritime safety."
            }
            
            files = {}
            if include_media:
                # Create a mock image file
                media_content = base64.b64decode(self.create_sample_media_file())
                files["media"] = ("test_image.png", media_content, "image/png")
            
            response = self.session.post(f"{self.base_url}/reports", data=test_data, files=files)
            
            if response.status_code == 200:
                report = response.json()
                
                # Validate response structure
                required_fields = ["id", "name", "location", "hazard_type", "description", "severity", "panic_index", "ai_category", "created_at"]
                missing_fields = [field for field in required_fields if field not in report]
                
                if missing_fields:
                    self.log_test(f"Create {hazard_type} Report", False, f"Missing fields: {missing_fields}")
                    return None
                
                # Validate AI classification
                if report["severity"] not in ["Low", "Medium", "High"]:
                    self.log_test(f"Create {hazard_type} Report", False, f"Invalid severity: {report['severity']}")
                    return None
                
                if not isinstance(report["panic_index"], int) or not (0 <= report["panic_index"] <= 100):
                    self.log_test(f"Create {hazard_type} Report", False, f"Invalid panic_index: {report['panic_index']}")
                    return None
                
                # Store report ID for cleanup
                self.created_report_ids.append(report["id"])
                
                media_info = "with media" if include_media else "without media"
                self.log_test(f"Create {hazard_type} Report ({media_info})", True, 
                            f"ID: {report['id']}, Severity: {report['severity']}, Panic: {report['panic_index']}")
                return report
                
            else:
                self.log_test(f"Create {hazard_type} Report", False, f"Status code: {response.status_code}, Response: {response.text}")
                return None
                
        except Exception as e:
            self.log_test(f"Create {hazard_type} Report", False, f"Exception: {str(e)}")
            return None
    
    def test_get_all_reports(self):
        """Test GET /api/reports endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/reports")
            
            if response.status_code == 200:
                reports = response.json()
                
                if isinstance(reports, list):
                    self.log_test("Get All Reports", True, f"Retrieved {len(reports)} reports")
                    return reports
                else:
                    self.log_test("Get All Reports", False, "Response is not a list")
                    return None
            else:
                self.log_test("Get All Reports", False, f"Status code: {response.status_code}")
                return None
                
        except Exception as e:
            self.log_test("Get All Reports", False, f"Exception: {str(e)}")
            return None
    
    def test_get_priority_reports(self):
        """Test GET /api/reports/priority endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/reports/priority")
            
            if response.status_code == 200:
                priority_reports = response.json()
                
                if isinstance(priority_reports, list):
                    # Validate structure
                    if priority_reports:
                        first_report = priority_reports[0]
                        if "report" in first_report and "priority_score" in first_report:
                            # Check if sorted by priority (highest first)
                            scores = [pr["priority_score"] for pr in priority_reports]
                            is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
                            
                            sort_status = "properly sorted" if is_sorted else "NOT properly sorted"
                            self.log_test("Get Priority Reports", True, 
                                        f"Retrieved {len(priority_reports)} priority reports, {sort_status}")
                            return priority_reports
                        else:
                            self.log_test("Get Priority Reports", False, "Invalid priority report structure")
                            return None
                    else:
                        self.log_test("Get Priority Reports", True, "No priority reports (empty list)")
                        return priority_reports
                else:
                    self.log_test("Get Priority Reports", False, "Response is not a list")
                    return None
            else:
                self.log_test("Get Priority Reports", False, f"Status code: {response.status_code}")
                return None
                
        except Exception as e:
            self.log_test("Get Priority Reports", False, f"Exception: {str(e)}")
            return None
    
    def test_get_heatmap_data(self):
        """Test GET /api/reports/heatmap endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/reports/heatmap")
            
            if response.status_code == 200:
                heatmap_data = response.json()
                
                if "heatmap_data" in heatmap_data and isinstance(heatmap_data["heatmap_data"], list):
                    points = heatmap_data["heatmap_data"]
                    
                    # Validate heatmap point structure
                    if points:
                        first_point = points[0]
                        required_fields = ["lat", "lng", "intensity", "hazard_type", "severity"]
                        missing_fields = [field for field in required_fields if field not in first_point]
                        
                        if missing_fields:
                            self.log_test("Get Heatmap Data", False, f"Missing fields in heatmap points: {missing_fields}")
                            return None
                        
                        # Validate intensity range
                        intensities = [point["intensity"] for point in points]
                        valid_intensities = all(0 <= intensity <= 1 for intensity in intensities)
                        
                        if not valid_intensities:
                            self.log_test("Get Heatmap Data", False, "Invalid intensity values (should be 0-1)")
                            return None
                    
                    self.log_test("Get Heatmap Data", True, f"Retrieved {len(points)} heatmap points")
                    return heatmap_data
                else:
                    self.log_test("Get Heatmap Data", False, "Invalid heatmap data structure")
                    return None
            else:
                self.log_test("Get Heatmap Data", False, f"Status code: {response.status_code}")
                return None
                
        except Exception as e:
            self.log_test("Get Heatmap Data", False, f"Exception: {str(e)}")
            return None
    
    def test_get_weather_data(self):
        """Test GET /api/weather endpoint"""
        try:
            # Test with Sydney coordinates
            params = {"lat": -33.8688, "lon": 151.2093}
            response = self.session.get(f"{self.base_url}/weather", params=params)
            
            if response.status_code == 200:
                weather_data = response.json()
                
                required_fields = ["location", "temperature", "humidity", "wind_speed", "wind_direction", "description", "timestamp"]
                missing_fields = [field for field in required_fields if field not in weather_data]
                
                if missing_fields:
                    self.log_test("Get Weather Data", False, f"Missing fields: {missing_fields}")
                    return None
                
                # Validate data types and ranges
                if not isinstance(weather_data["temperature"], (int, float)):
                    self.log_test("Get Weather Data", False, "Invalid temperature type")
                    return None
                
                if not (0 <= weather_data["humidity"] <= 100):
                    self.log_test("Get Weather Data", False, f"Invalid humidity: {weather_data['humidity']}")
                    return None
                
                self.log_test("Get Weather Data", True, 
                            f"Temp: {weather_data['temperature']}°C, Humidity: {weather_data['humidity']}%")
                return weather_data
            else:
                self.log_test("Get Weather Data", False, f"Status code: {response.status_code}")
                return None
                
        except Exception as e:
            self.log_test("Get Weather Data", False, f"Exception: {str(e)}")
            return None
    
    def test_get_dashboard_stats(self):
        """Test GET /api/dashboard/stats endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/dashboard/stats")
            
            if response.status_code == 200:
                stats = response.json()
                
                required_fields = ["total_reports", "severity_breakdown", "hazard_types", "average_panic_index", "active_alerts"]
                missing_fields = [field for field in required_fields if field not in stats]
                
                if missing_fields:
                    self.log_test("Get Dashboard Stats", False, f"Missing fields: {missing_fields}")
                    return None
                
                # Validate severity breakdown structure
                severity_breakdown = stats["severity_breakdown"]
                if not all(key in severity_breakdown for key in ["high", "medium", "low"]):
                    self.log_test("Get Dashboard Stats", False, "Invalid severity_breakdown structure")
                    return None
                
                # Validate data consistency
                total_from_breakdown = sum(severity_breakdown.values())
                if stats["total_reports"] != total_from_breakdown:
                    self.log_test("Get Dashboard Stats", False, 
                                f"Inconsistent totals: {stats['total_reports']} vs {total_from_breakdown}")
                    return None
                
                self.log_test("Get Dashboard Stats", True, 
                            f"Total: {stats['total_reports']}, Active Alerts: {stats['active_alerts']}")
                return stats
            else:
                self.log_test("Get Dashboard Stats", False, f"Status code: {response.status_code}")
                return None
                
        except Exception as e:
            self.log_test("Get Dashboard Stats", False, f"Exception: {str(e)}")
            return None
    
    def test_delete_report(self, report_id):
        """Test DELETE /api/reports/{id} endpoint"""
        try:
            response = self.session.delete(f"{self.base_url}/reports/{report_id}")
            
            if response.status_code == 200:
                result = response.json()
                if "message" in result:
                    self.log_test("Delete Report", True, f"Deleted report {report_id}")
                    return True
                else:
                    self.log_test("Delete Report", False, "No message in response")
                    return False
            elif response.status_code == 404:
                self.log_test("Delete Report", False, f"Report {report_id} not found")
                return False
            else:
                self.log_test("Delete Report", False, f"Status code: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Delete Report", False, f"Exception: {str(e)}")
            return False
    
    def run_comprehensive_test_suite(self):
        """Run all tests in sequence"""
        print("=" * 60)
        print("OCEAN HAZARD ALERT SYSTEM - BACKEND API TEST SUITE")
        print("=" * 60)
        print(f"Testing backend at: {self.base_url}")
        print()
        
        test_results = {}
        
        # 1. Health Check
        test_results["health"] = self.test_health_endpoint()
        
        # 2. Create different types of hazard reports
        hazard_types = ["Cyclone", "Oil Spill", "Flood", "Tsunami", "Other"]
        created_reports = []
        
        for i, hazard_type in enumerate(hazard_types):
            include_media = (i == 0)  # Include media for first report only
            report = self.test_create_hazard_report(hazard_type, include_media)
            if report:
                created_reports.append(report)
            test_results[f"create_{hazard_type.lower().replace(' ', '_')}"] = report is not None
        
        # Small delay to ensure data consistency
        time.sleep(1)
        
        # 3. Test retrieval endpoints
        test_results["get_all_reports"] = self.test_get_all_reports() is not None
        test_results["get_priority_reports"] = self.test_get_priority_reports() is not None
        test_results["get_heatmap_data"] = self.test_get_heatmap_data() is not None
        
        # 4. Test weather endpoint
        test_results["get_weather"] = self.test_get_weather_data() is not None
        
        # 5. Test dashboard stats
        test_results["get_dashboard_stats"] = self.test_get_dashboard_stats() is not None
        
        # 6. Test delete functionality (clean up created reports)
        delete_success = 0
        for report_id in self.created_report_ids:
            if self.test_delete_report(report_id):
                delete_success += 1
        
        test_results["delete_reports"] = delete_success == len(self.created_report_ids)
        
        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in test_results.values() if result)
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
        
        print()
        print(f"Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Backend API is working correctly.")
        else:
            print(f"⚠️  {total - passed} test(s) failed. Please check the issues above.")
        
        return test_results

def main():
    """Main test execution"""
    tester = OceanHazardAPITester()
    results = tester.run_comprehensive_test_suite()
    
    # Return exit code based on results
    all_passed = all(results.values())
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())