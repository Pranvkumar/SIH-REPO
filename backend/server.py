from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import json
import uuid
import asyncio
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import httpx
import base64
from emergentintegrations.llm.chat import LlmChat, UserMessage

# Load environment variables
load_dotenv()

app = FastAPI(title="Ocean Hazard Alert API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URL = os.environ.get('MONGO_URL')
client = AsyncIOMotorClient(MONGO_URL)
db = client.ocean_hazards

# OpenWeatherMap API (free tier)
OPENWEATHER_API_KEY = "demo_key"  # Using demo for now
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# LLM Integration
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

# Pydantic models
class Location(BaseModel):
    latitude: float
    longitude: float
    address: Optional[str] = None

class HazardReport(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    location: Location
    hazard_type: str  # Cyclone, Oil Spill, Flood, Tsunami, Other
    description: str
    media_base64: Optional[str] = None
    media_type: Optional[str] = None
    severity: Optional[str] = None  # Low, Medium, High
    panic_index: Optional[int] = None  # 0-100
    ai_category: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "pending"  # pending, reviewed, resolved

class WeatherData(BaseModel):
    location: str
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: int
    description: str
    timestamp: str

class PriorityReport(BaseModel):
    report: HazardReport
    priority_score: float

# Helper functions
def prepare_for_mongo(data):
    """Convert data for MongoDB storage"""
    if isinstance(data, dict):
        # Handle datetime serialization
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
    return data

async def classify_hazard_with_ai(description: str, hazard_type: str) -> Dict[str, Any]:
    """Use LLM to classify hazard severity and generate panic index"""
    try:
        system_message = """You are an expert ocean hazard analyst. Analyze the given hazard report and provide:
1. Severity level: Low, Medium, or High
2. Panic index: Score from 0-100 (0=no panic, 100=extreme panic)
3. AI category: Refined category based on description

Respond in JSON format:
{
    "severity": "Low|Medium|High",
    "panic_index": 0-100,
    "ai_category": "refined_category",
    "reasoning": "brief explanation"
}"""

        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"hazard-{uuid.uuid4()}",
            system_message=system_message
        ).with_model("openai", "gpt-4o-mini")

        user_message = UserMessage(
            text=f"Hazard Type: {hazard_type}\nDescription: {description}\n\nPlease analyze this ocean hazard report."
        )

        response = await chat.send_message(user_message)
        
        # Try to parse JSON response
        try:
            result = json.loads(response)
            return {
                "severity": result.get("severity", "Medium"),
                "panic_index": result.get("panic_index", 50),
                "ai_category": result.get("ai_category", hazard_type),
                "reasoning": result.get("reasoning", "AI analysis completed")
            }
        except json.JSONDecodeError:
            # Fallback if response is not JSON
            return {
                "severity": "Medium",
                "panic_index": 50,
                "ai_category": hazard_type,
                "reasoning": "AI analysis completed"
            }
    except Exception as e:
        print(f"AI classification error: {e}")
        return {
            "severity": "Medium",
            "panic_index": 50,
            "ai_category": hazard_type,
            "reasoning": "Default classification applied"
        }

async def get_weather_data(lat: float, lon: float) -> Optional[WeatherData]:
    """Fetch weather data from OpenWeatherMap API"""
    try:
        # Using mock data for demo since we don't have a real API key
        # In production, you would use: f"{OPENWEATHER_BASE_URL}/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        
        # Mock weather data based on location
        mock_data = {
            "location": f"{lat:.2f}, {lon:.2f}",
            "temperature": 28.5,
            "humidity": 75,
            "wind_speed": 15.2,
            "wind_direction": 180,
            "description": "Partly cloudy with moderate winds",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return WeatherData(**mock_data)
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

# API Routes

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Ocean Hazard Alert API"}

@app.post("/api/reports", response_model=HazardReport)
async def create_report(
    name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    address: str = Form(""),
    hazard_type: str = Form(...),
    description: str = Form(...),
    media: Optional[UploadFile] = File(None)
):
    """Create a new hazard report"""
    
    # Handle media upload
    media_base64 = None
    media_type = None
    if media:
        content = await media.read()
        media_base64 = base64.b64encode(content).decode('utf-8')
        media_type = media.content_type
    
    # Create location
    location = Location(
        latitude=latitude,
        longitude=longitude,
        address=address
    )
    
    # Create initial report
    report = HazardReport(
        name=name,
        location=location,
        hazard_type=hazard_type,
        description=description,
        media_base64=media_base64,
        media_type=media_type
    )
    
    # AI Classification
    ai_result = await classify_hazard_with_ai(description, hazard_type)
    report.severity = ai_result["severity"]
    report.panic_index = ai_result["panic_index"]
    report.ai_category = ai_result["ai_category"]
    
    # Save to database
    report_dict = report.dict()
    report_dict = prepare_for_mongo(report_dict)
    await db.reports.insert_one(report_dict)
    
    return report

@app.get("/api/reports", response_model=List[HazardReport])
async def get_reports():
    """Get all hazard reports"""
    reports = await db.reports.find().sort("created_at", -1).to_list(length=None)
    return [HazardReport(**report) for report in reports]

@app.get("/api/reports/priority", response_model=List[PriorityReport])
async def get_priority_reports():
    """Get reports sorted by priority (severity + panic index)"""
    reports = await db.reports.find().to_list(length=None)
    
    priority_reports = []
    for report_dict in reports:
        report = HazardReport(**report_dict)
        
        # Calculate priority score
        severity_weights = {"Low": 1, "Medium": 2, "High": 3}
        severity_score = severity_weights.get(report.severity, 2)
        panic_score = (report.panic_index or 50) / 100
        priority_score = (severity_score * 0.6) + (panic_score * 0.4)
        
        priority_reports.append(PriorityReport(
            report=report,
            priority_score=priority_score
        ))
    
    # Sort by priority score (highest first)
    priority_reports.sort(key=lambda x: x.priority_score, reverse=True)
    
    return priority_reports[:10]  # Top 10 priority reports

@app.get("/api/reports/heatmap")
async def get_heatmap_data():
    """Get heatmap data for map visualization"""
    reports = await db.reports.find().to_list(length=None)
    
    heatmap_points = []
    for report_dict in reports:
        report = HazardReport(**report_dict)
        
        # Calculate intensity based on severity and panic index
        severity_weights = {"Low": 0.3, "Medium": 0.6, "High": 1.0}
        intensity = severity_weights.get(report.severity, 0.5)
        panic_boost = (report.panic_index or 50) / 200  # 0-0.5 boost
        final_intensity = min(intensity + panic_boost, 1.0)
        
        heatmap_points.append({
            "lat": report.location.latitude,
            "lng": report.location.longitude,
            "intensity": final_intensity,
            "hazard_type": report.hazard_type,
            "severity": report.severity
        })
    
    return {"heatmap_data": heatmap_points}

@app.get("/api/weather")
async def get_weather(lat: float, lon: float):
    """Get weather data for specified coordinates"""
    weather_data = await get_weather_data(lat, lon)
    if weather_data:
        return weather_data.dict()
    else:
        raise HTTPException(status_code=500, detail="Weather data unavailable")

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    total_reports = await db.reports.count_documents({})
    
    # Count by severity
    high_severity = await db.reports.count_documents({"severity": "High"})
    medium_severity = await db.reports.count_documents({"severity": "Medium"})
    low_severity = await db.reports.count_documents({"severity": "Low"})
    
    # Count by type
    hazard_types = {}
    reports = await db.reports.find().to_list(length=None)
    for report in reports:
        hazard_type = report.get("hazard_type", "Other")
        hazard_types[hazard_type] = hazard_types.get(hazard_type, 0) + 1
    
    # Calculate average panic index
    total_panic = sum(report.get("panic_index", 50) for report in reports)
    avg_panic = total_panic / total_reports if total_reports > 0 else 0
    
    return {
        "total_reports": total_reports,
        "severity_breakdown": {
            "high": high_severity,
            "medium": medium_severity,
            "low": low_severity
        },
        "hazard_types": hazard_types,
        "average_panic_index": round(avg_panic, 1),
        "active_alerts": high_severity  # High severity reports as active alerts
    }

@app.delete("/api/reports/{report_id}")
async def delete_report(report_id: str):
    """Delete a hazard report (admin function)"""
    result = await db.reports.delete_one({"id": report_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"message": "Report deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)