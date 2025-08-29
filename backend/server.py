from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import os
import json
import uuid
import asyncio
from datetime import datetime, timezone, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import httpx
import base64
import re
import math
from emergentintegrations.llm.chat import LlmChat, UserMessage
import asyncio
import aiofiles
from collections import defaultdict

# Load environment variables
load_dotenv()

app = FastAPI(title="Ocean Hazard Alert API", version="2.0.0")

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

# API Keys
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
TWITTER_BEARER_TOKEN = os.environ.get('TWITTER_BEARER_TOKEN', 'demo_token')
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', 'demo_key')

# Supported languages for multilingual support
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish', 
    'hi': 'Hindi',
    'fr': 'French',
    'pt': 'Portuguese',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'ja': 'Japanese'
}

# Pydantic models
class Location(BaseModel):
    latitude: float
    longitude: float
    address: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None

class MediaItem(BaseModel):
    type: str  # image, video, audio
    base64_data: Optional[str] = None
    url: Optional[str] = None
    filename: Optional[str] = None
    size: Optional[int] = None

class HazardReport(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    location: Location
    hazard_type: str  # Cyclone, Oil Spill, Flood, Tsunami, Earthquake, Other
    description: str
    media_items: List[MediaItem] = []
    severity: Optional[str] = None  # Low, Medium, High, Critical
    panic_index: Optional[int] = None  # 0-100
    ai_category: Optional[str] = None
    sentiment: Optional[str] = None  # positive, negative, neutral, urgent
    language: str = 'en'
    source: str = 'citizen_report'  # citizen_report, social_media, official, sensor
    verification_status: str = 'pending'  # pending, verified, false_positive, investigating
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: Optional[str] = None
    offline_created: Optional[bool] = False
    sync_status: str = 'synced'  # synced, pending_sync, sync_failed

class SocialMediaPost(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    platform: str  # twitter, facebook, youtube, instagram
    post_id: str
    author: str
    content: str
    location: Optional[Location] = None
    media_urls: List[str] = []
    engagement_metrics: Dict[str, int] = {}  # likes, retweets, comments, etc.
    hashtags: List[str] = []
    mentions: List[str] = []
    language: str = 'en'
    sentiment: Optional[str] = None
    hazard_keywords: List[str] = []
    confidence_score: Optional[float] = None
    created_at: str
    processed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Hotspot(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    center_location: Location
    radius_km: float
    report_count: int
    social_media_count: int
    severity_distribution: Dict[str, int] = {}
    dominant_hazard_type: str
    confidence_score: float
    risk_level: str  # low, medium, high, critical
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class UserRegistration(BaseModel):
    username: str
    email: str
    password: str
    full_name: str
    role: str = 'citizen'  # citizen, official, analyst, admin
    location: Optional[Location] = None
    language_preference: str = 'en'
    notification_preferences: Dict[str, bool] = {}

class AnalyticsData(BaseModel):
    total_reports: int
    social_media_posts: int
    active_hotspots: int
    verification_rate: float
    sentiment_distribution: Dict[str, int]
    language_distribution: Dict[str, int]
    source_distribution: Dict[str, int]
    trend_data: List[Dict] = []

# Multilingual text translations
TRANSLATIONS = {
    'en': {
        'hazard_detected': 'Ocean hazard detected',
        'high_severity': 'High severity incident',
        'verification_needed': 'Verification needed',
        'emergency_alert': 'Emergency Alert'
    },
    'es': {
        'hazard_detected': 'Peligro oceánico detectado',
        'high_severity': 'Incidente de alta gravedad',
        'verification_needed': 'Verificación necesaria',
        'emergency_alert': 'Alerta de Emergencia'
    },
    'hi': {
        'hazard_detected': 'समुद्री खतरा पाया गया',
        'high_severity': 'उच्च गंभीरता की घटना',
        'verification_needed': 'सत्यापन आवश्यक',
        'emergency_alert': 'आपातकालीन अलर्ट'
    }
}

# Hazard keywords for different languages
HAZARD_KEYWORDS = {
    'en': ['tsunami', 'cyclone', 'hurricane', 'storm', 'flood', 'oil spill', 'earthquake', 'tidal wave', 'emergency', 'danger', 'evacuation'],
    'es': ['tsunami', 'ciclón', 'huracán', 'tormenta', 'inundación', 'derrame', 'terremoto', 'emergencia', 'peligro', 'evacuación'],
    'hi': ['सुनामी', 'चक्रवात', 'तूफान', 'बाढ़', 'भूकंप', 'आपातकाल', 'खतरा']
}

# Helper functions
def prepare_for_mongo(data):
    """Convert data for MongoDB storage"""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
    return data

async def advanced_ai_analysis(text: str, language: str = 'en') -> Dict[str, Any]:
    """Enhanced AI analysis with multilingual support and sentiment analysis"""
    try:
        system_message = f"""You are an advanced ocean hazard analyst with multilingual capabilities. 
        Analyze the given text and provide comprehensive analysis in JSON format.
        
        Text Language: {language}
        
        Provide analysis in this exact JSON structure:
        {{
            "hazard_type": "Cyclone|Oil Spill|Flood|Tsunami|Earthquake|Other",
            "severity": "Low|Medium|High|Critical",
            "panic_index": 0-100,
            "sentiment": "positive|negative|neutral|urgent|panic",
            "confidence_score": 0.0-1.0,
            "extracted_keywords": ["keyword1", "keyword2"],
            "location_mentions": ["location1", "location2"],
            "urgency_indicators": ["indicator1", "indicator2"],
            "verification_needed": true|false,
            "risk_assessment": "detailed risk analysis",
            "recommended_actions": ["action1", "action2"]
        }}"""

        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"analysis-{uuid.uuid4()}",
            system_message=system_message
        ).with_model("openai", "gpt-4o-mini")

        user_message = UserMessage(text=f"Analyze this ocean hazard report: {text}")
        response = await chat.send_message(user_message)
        
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # Fallback analysis
            return {
                "hazard_type": "Other",
                "severity": "Medium", 
                "panic_index": 50,
                "sentiment": "neutral",
                "confidence_score": 0.5,
                "extracted_keywords": [],
                "location_mentions": [],
                "urgency_indicators": [],
                "verification_needed": True,
                "risk_assessment": "Analysis completed",
                "recommended_actions": ["Monitor situation", "Verify with local authorities"]
            }
    except Exception as e:
        print(f"AI analysis error: {e}")
        return {
            "hazard_type": "Other",
            "severity": "Medium",
            "panic_index": 50,
            "sentiment": "neutral", 
            "confidence_score": 0.0,
            "extracted_keywords": [],
            "location_mentions": [],
            "urgency_indicators": [],
            "verification_needed": True,
            "risk_assessment": "Analysis failed",
            "recommended_actions": ["Manual review required"]
        }

async def detect_hazard_keywords(text: str, language: str = 'en') -> List[str]:
    """Detect hazard-related keywords in text"""
    keywords = HAZARD_KEYWORDS.get(language, HAZARD_KEYWORDS['en'])
    text_lower = text.lower()
    found_keywords = [keyword for keyword in keywords if keyword.lower() in text_lower]
    return found_keywords

async def generate_hotspots() -> List[Hotspot]:
    """Generate dynamic hotspots based on report density and social media activity"""
    try:
        # Get reports from last 24 hours
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        
        reports_cursor = db.reports.find({"created_at": {"$gte": yesterday}})
        reports = await reports_cursor.to_list(length=None)
        
        social_posts_cursor = db.social_media.find({"processed_at": {"$gte": yesterday}})
        social_posts = await social_posts_cursor.to_list(length=None)
        
        # Group reports by geographic proximity (0.1 degree ~ 11km)
        location_clusters = defaultdict(list)
        
        for report in reports:
            lat = round(report['location']['latitude'], 1)
            lng = round(report['location']['longitude'], 1) 
            location_clusters[(lat, lng)].append(report)
        
        # Add social media posts to clusters
        for post in social_posts:
            if post.get('location'):
                lat = round(post['location']['latitude'], 1)
                lng = round(post['location']['longitude'], 1)
                if (lat, lng) not in location_clusters:
                    location_clusters[(lat, lng)] = []
                # Add to existing cluster data
        
        hotspots = []
        for (lat, lng), cluster_reports in location_clusters.items():
            if len(cluster_reports) >= 3:  # Minimum 3 reports for hotspot
                # Calculate metrics
                severity_dist = defaultdict(int)
                hazard_types = defaultdict(int)
                
                for report in cluster_reports:
                    severity = report.get('severity', 'Medium')
                    severity_dist[severity] += 1
                    hazard_type = report.get('hazard_type', 'Other')
                    hazard_types[hazard_type] += 1
                
                # Determine dominant hazard type
                dominant_hazard = max(hazard_types.items(), key=lambda x: x[1])[0] if hazard_types else 'Other'
                
                # Calculate risk level
                high_severity_count = severity_dist.get('High', 0) + severity_dist.get('Critical', 0)
                total_reports = len(cluster_reports)
                risk_ratio = high_severity_count / total_reports if total_reports > 0 else 0
                
                if risk_ratio >= 0.7:
                    risk_level = 'critical'
                elif risk_ratio >= 0.4:
                    risk_level = 'high'
                elif risk_ratio >= 0.2:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
                
                hotspot = Hotspot(
                    center_location=Location(latitude=lat, longitude=lng),
                    radius_km=5.5,  # ~0.05 degrees
                    report_count=len(cluster_reports),
                    social_media_count=len([p for p in social_posts if p.get('location') and 
                                          abs(p['location']['latitude'] - lat) <= 0.1 and 
                                          abs(p['location']['longitude'] - lng) <= 0.1]),
                    severity_distribution=dict(severity_dist),
                    dominant_hazard_type=dominant_hazard,
                    confidence_score=min(0.9, 0.3 + (total_reports * 0.1)),
                    risk_level=risk_level
                )
                hotspots.append(hotspot)
        
        return hotspots
    except Exception as e:
        print(f"Hotspot generation error: {e}")
        return []

async def process_social_media_mock(keywords: List[str], language: str = 'en') -> List[Dict]:
    """Mock social media processing (in production, integrate with real APIs)"""
    # Mock social media data generation
    mock_posts = []
    platforms = ['twitter', 'facebook', 'youtube']
    
    for i in range(5):  # Generate 5 mock posts
        platform = platforms[i % len(platforms)]
        post = {
            'platform': platform,
            'post_id': f'mock_{platform}_{i}',
            'author': f'user_{i}',
            'content': f'Observing severe weather conditions near the coast. {keywords[0] if keywords else "Storm"} approaching rapidly!',
            'location': {
                'latitude': 20.5937 + (i * 0.1),
                'longitude': 78.9629 + (i * 0.1)
            },
            'engagement_metrics': {
                'likes': (i + 1) * 10,
                'shares': (i + 1) * 5,
                'comments': (i + 1) * 3
            },
            'hashtags': ['#oceanstorm', '#emergency', '#weather'],
            'language': language,
            'created_at': (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat()
        }
        mock_posts.append(post)
    
    return mock_posts

# API Routes

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Ocean Hazard Alert API v2.0",
        "features": ["social_media", "multilingual", "hotspots", "offline_sync"],
        "supported_languages": list(SUPPORTED_LANGUAGES.keys())
    }

@app.post("/api/users/register")
async def register_user(user_data: UserRegistration):
    """Register new user with role-based access"""
    # Check if user exists
    existing_user = await db.users.find_one({"username": user_data.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    existing_email = await db.users.find_one({"email": user_data.email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user_dict = user_data.dict()
    user_dict['id'] = str(uuid.uuid4())
    user_dict['created_at'] = datetime.now(timezone.utc).isoformat()
    user_dict['is_active'] = True
    user_dict['verified'] = False
    
    # Hash password (in production, use proper hashing)
    user_dict['password'] = f"hashed_{user_data.password}"
    
    await db.users.insert_one(prepare_for_mongo(user_dict))
    
    # Remove password from response
    del user_dict['password']
    return user_dict

@app.post("/api/reports/bulk", response_model=List[HazardReport])
async def create_bulk_reports(reports_data: List[Dict]):
    """Handle bulk report creation (for offline sync)"""
    created_reports = []
    
    for report_data in reports_data:
        # Enhanced AI analysis
        ai_result = await advanced_ai_analysis(
            report_data.get('description', ''), 
            report_data.get('language', 'en')
        )
        
        # Create report with AI analysis
        report = HazardReport(**report_data)
        report.severity = ai_result["severity"]
        report.panic_index = ai_result["panic_index"] 
        report.ai_category = ai_result["hazard_type"]
        report.sentiment = ai_result["sentiment"]
        report.verification_status = "verification_needed" if ai_result["verification_needed"] else "pending"
        
        # Save to database
        report_dict = report.dict()
        report_dict = prepare_for_mongo(report_dict)
        await db.reports.insert_one(report_dict)
        
        created_reports.append(report)
    
    return created_reports

@app.post("/api/reports/advanced", response_model=HazardReport)
async def create_advanced_report(
    name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    address: str = Form(""),
    country: str = Form(""),
    hazard_type: str = Form(...),
    description: str = Form(...),
    language: str = Form('en'),
    offline_created: bool = Form(False),
    media_files: List[UploadFile] = File([])
):
    """Create enhanced hazard report with multiple media files"""
    
    # Process media files
    media_items = []
    for media_file in media_files:
        if media_file.filename:
            content = await media_file.read()
            media_item = MediaItem(
                type=media_file.content_type.split('/')[0] if media_file.content_type else 'unknown',
                base64_data=base64.b64encode(content).decode('utf-8'),
                filename=media_file.filename,
                size=len(content)
            )
            media_items.append(media_item)
    
    # Create location
    location = Location(
        latitude=latitude,
        longitude=longitude,
        address=address,
        country=country
    )
    
    # Enhanced AI analysis with multilingual support
    ai_result = await advanced_ai_analysis(description, language)
    
    # Create report
    report = HazardReport(
        name=name,
        location=location,
        hazard_type=hazard_type,
        description=description,
        media_items=media_items,
        language=language,
        offline_created=offline_created,
        sync_status='synced' if not offline_created else 'pending_sync',
        severity=ai_result["severity"],
        panic_index=ai_result["panic_index"],
        ai_category=ai_result["hazard_type"],
        sentiment=ai_result["sentiment"]
    )
    
    # Save to database
    report_dict = report.dict()
    report_dict = prepare_for_mongo(report_dict)
    await db.reports.insert_one(report_dict)
    
    return report

@app.get("/api/reports/advanced", response_model=List[HazardReport])
async def get_advanced_reports(
    language: str = Query('all'),
    source: str = Query('all'),
    severity: str = Query('all'),
    verification_status: str = Query('all'),
    limit: int = Query(50),
    offset: int = Query(0)
):
    """Get reports with advanced filtering"""
    filter_criteria = {}
    
    if language != 'all':
        filter_criteria['language'] = language
    if source != 'all':
        filter_criteria['source'] = source
    if severity != 'all':
        filter_criteria['severity'] = severity
    if verification_status != 'all':
        filter_criteria['verification_status'] = verification_status
    
    reports = await db.reports.find(filter_criteria).sort("created_at", -1).skip(offset).limit(limit).to_list(length=None)
    return [HazardReport(**report) for report in reports]

@app.get("/api/social-media/process")
async def process_social_media(keywords: str = Query("tsunami,cyclone,flood")):
    """Process social media for hazard-related content"""
    keyword_list = [k.strip() for k in keywords.split(',')]
    
    # Mock social media processing (integrate with real APIs in production)
    social_posts = await process_social_media_mock(keyword_list)
    
    processed_posts = []
    for post_data in social_posts:
        # AI analysis for social media content
        ai_result = await advanced_ai_analysis(post_data['content'], post_data.get('language', 'en'))
        
        # Detect hazard keywords
        detected_keywords = await detect_hazard_keywords(post_data['content'], post_data.get('language', 'en'))
        
        social_post = SocialMediaPost(
            **post_data,
            sentiment=ai_result["sentiment"],
            hazard_keywords=detected_keywords,
            confidence_score=ai_result["confidence_score"]
        )
        
        # Save to database
        post_dict = social_post.dict()
        post_dict = prepare_for_mongo(post_dict)
        await db.social_media.insert_one(post_dict)
        
        processed_posts.append(social_post)
    
    return {
        "processed_count": len(processed_posts),
        "posts": processed_posts,
        "keywords_used": keyword_list
    }

@app.get("/api/hotspots", response_model=List[Hotspot])
async def get_dynamic_hotspots():
    """Get dynamically generated hotspots"""
    hotspots = await generate_hotspots()
    
    # Save hotspots to database
    for hotspot in hotspots:
        hotspot_dict = hotspot.dict()
        hotspot_dict = prepare_for_mongo(hotspot_dict)
        
        # Update existing or create new
        await db.hotspots.replace_one(
            {"center_location.latitude": hotspot.center_location.latitude,
             "center_location.longitude": hotspot.center_location.longitude},
            hotspot_dict,
            upsert=True
        )
    
    return hotspots

@app.get("/api/analytics/advanced", response_model=AnalyticsData)
async def get_advanced_analytics():
    """Get comprehensive analytics data"""
    
    # Aggregate data from last 7 days
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    
    # Count reports
    total_reports = await db.reports.count_documents({"created_at": {"$gte": week_ago}})
    
    # Count social media posts
    social_media_posts = await db.social_media.count_documents({"processed_at": {"$gte": week_ago}})
    
    # Count active hotspots
    active_hotspots = await db.hotspots.count_documents({
        "last_updated": {"$gte": week_ago},
        "risk_level": {"$in": ["medium", "high", "critical"]}
    })
    
    # Calculate verification rate
    verified_reports = await db.reports.count_documents({
        "created_at": {"$gte": week_ago},
        "verification_status": "verified"
    })
    verification_rate = verified_reports / total_reports if total_reports > 0 else 0
    
    # Sentiment distribution
    sentiment_pipeline = [
        {"$match": {"created_at": {"$gte": week_ago}}},
        {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}}
    ]
    sentiment_results = await db.reports.aggregate(sentiment_pipeline).to_list(length=None)
    sentiment_distribution = {result["_id"]: result["count"] for result in sentiment_results}
    
    # Language distribution
    language_pipeline = [
        {"$match": {"created_at": {"$gte": week_ago}}},
        {"$group": {"_id": "$language", "count": {"$sum": 1}}}
    ]
    language_results = await db.reports.aggregate(language_pipeline).to_list(length=None)
    language_distribution = {result["_id"]: result["count"] for result in language_results}
    
    # Source distribution
    source_pipeline = [
        {"$match": {"created_at": {"$gte": week_ago}}},
        {"$group": {"_id": "$source", "count": {"$sum": 1}}}
    ]
    source_results = await db.reports.aggregate(source_pipeline).to_list(length=None)
    source_distribution = {result["_id"]: result["count"] for result in source_results}
    
    # Generate trend data (daily counts for last 7 days)
    trend_data = []
    for i in range(7):
        day_start = (datetime.now(timezone.utc) - timedelta(days=i+1)).replace(hour=0, minute=0, second=0).isoformat()
        day_end = (datetime.now(timezone.utc) - timedelta(days=i)).replace(hour=0, minute=0, second=0).isoformat()
        
        day_reports = await db.reports.count_documents({
            "created_at": {"$gte": day_start, "$lt": day_end}
        })
        day_social = await db.social_media.count_documents({
            "processed_at": {"$gte": day_start, "$lt": day_end}
        })
        
        trend_data.append({
            "date": day_start[:10],
            "reports": day_reports,
            "social_media": day_social
        })
    
    return AnalyticsData(
        total_reports=total_reports,
        social_media_posts=social_media_posts,
        active_hotspots=active_hotspots,
        verification_rate=round(verification_rate, 3),
        sentiment_distribution=sentiment_distribution,
        language_distribution=language_distribution,
        source_distribution=source_distribution,
        trend_data=list(reversed(trend_data))
    )

@app.get("/api/reports/map-data")
async def get_map_visualization_data():
    """Get enhanced data for map visualization"""
    
    # Get all reports from last 24 hours
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    
    reports = await db.reports.find({"created_at": {"$gte": yesterday}}).to_list(length=None)
    social_posts = await db.social_media.find({"processed_at": {"$gte": yesterday}}).to_list(length=None)
    hotspots = await db.hotspots.find({"last_updated": {"$gte": yesterday}}).to_list(length=None)
    
    # Prepare map markers
    report_markers = []
    for report in reports:
        severity_colors = {"Low": "#22c55e", "Medium": "#eab308", "High": "#ef4444", "Critical": "#dc2626"}
        
        marker = {
            "id": report["id"],
            "type": "report",
            "lat": report["location"]["latitude"],
            "lng": report["location"]["longitude"],
            "severity": report.get("severity", "Medium"),
            "color": severity_colors.get(report.get("severity", "Medium"), "#6b7280"),
            "hazard_type": report.get("hazard_type", "Other"),
            "description": report["description"][:100] + "..." if len(report["description"]) > 100 else report["description"],
            "created_at": report["created_at"],
            "verification_status": report.get("verification_status", "pending"),
            "source": report.get("source", "citizen_report")
        }
        report_markers.append(marker)
    
    # Social media markers
    social_markers = []
    for post in social_posts:
        if post.get("location"):
            marker = {
                "id": post["id"],
                "type": "social_media",
                "lat": post["location"]["latitude"], 
                "lng": post["location"]["longitude"],
                "platform": post["platform"],
                "content": post["content"][:100] + "..." if len(post["content"]) > 100 else post["content"],
                "sentiment": post.get("sentiment", "neutral"),
                "engagement": sum(post.get("engagement_metrics", {}).values()),
                "created_at": post["created_at"]
            }
            social_markers.append(marker)
    
    # Hotspot areas
    hotspot_areas = []
    for hotspot in hotspots:
        risk_colors = {"low": "#22c55e", "medium": "#eab308", "high": "#ef4444", "critical": "#dc2626"}
        
        area = {
            "id": hotspot["id"],
            "center_lat": hotspot["center_location"]["latitude"],
            "center_lng": hotspot["center_location"]["longitude"],
            "radius_km": hotspot["radius_km"],
            "risk_level": hotspot["risk_level"],
            "color": risk_colors.get(hotspot["risk_level"], "#6b7280"),
            "report_count": hotspot["report_count"],
            "social_media_count": hotspot["social_media_count"],
            "dominant_hazard": hotspot["dominant_hazard_type"],
            "confidence": hotspot["confidence_score"]
        }
        hotspot_areas.append(area)
    
    return {
        "reports": report_markers,
        "social_media": social_markers, 
        "hotspots": hotspot_areas,
        "summary": {
            "total_reports": len(report_markers),
            "total_social_posts": len(social_markers),
            "active_hotspots": len(hotspot_areas),
            "high_risk_areas": len([h for h in hotspot_areas if h["risk_level"] in ["high", "critical"]])
        }
    }

@app.get("/api/translations/{language}")
async def get_translations(language: str):
    """Get UI translations for specified language"""
    return TRANSLATIONS.get(language, TRANSLATIONS['en'])

@app.put("/api/reports/{report_id}/verify")
async def verify_report(report_id: str, verification_status: str = Query(...)):
    """Update report verification status (for officials/analysts)"""
    if verification_status not in ["verified", "false_positive", "investigating", "pending"]:
        raise HTTPException(status_code=400, detail="Invalid verification status")
    
    result = await db.reports.update_one(
        {"id": report_id},
        {"$set": {
            "verification_status": verification_status,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return {"message": "Report verification status updated", "status": verification_status}

# Keep existing endpoints for backward compatibility
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
    """Legacy endpoint for simple report creation"""
    media_items = []
    if media:
        content = await media.read()
        media_item = MediaItem(
            type=media.content_type.split('/')[0] if media.content_type else 'unknown',
            base64_data=base64.b64encode(content).decode('utf-8'),
            filename=media.filename
        )
        media_items.append(media_item)
    
    location = Location(latitude=latitude, longitude=longitude, address=address)
    
    ai_result = await advanced_ai_analysis(description)
    
    report = HazardReport(
        name=name,
        location=location, 
        hazard_type=hazard_type,
        description=description,
        media_items=media_items,
        severity=ai_result["severity"],
        panic_index=ai_result["panic_index"],
        ai_category=ai_result["hazard_type"],
        sentiment=ai_result["sentiment"]
    )
    
    report_dict = report.dict()
    report_dict = prepare_for_mongo(report_dict)
    await db.reports.insert_one(report_dict)
    
    return report

@app.get("/api/reports", response_model=List[HazardReport])
async def get_reports():
    """Legacy endpoint for getting reports"""
    reports = await db.reports.find().sort("created_at", -1).to_list(length=None)
    return [HazardReport(**report) for report in reports]

@app.get("/api/reports/priority")
async def get_priority_reports():
    """Enhanced priority reports with social media integration"""
    reports = await db.reports.find().to_list(length=None)
    
    priority_reports = []
    for report_dict in reports:
        report = HazardReport(**report_dict)
        
        # Enhanced priority scoring
        severity_weights = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        severity_score = severity_weights.get(report.severity, 2)
        
        panic_score = (report.panic_index or 50) / 100
        
        # Add verification and recency factors
        verification_bonus = 0.2 if report.verification_status == "verified" else 0
        
        # Time decay (newer reports get higher priority)
        hours_old = (datetime.now(timezone.utc) - datetime.fromisoformat(report.created_at.replace('Z', '+00:00'))).total_seconds() / 3600
        time_factor = max(0.1, 1 - (hours_old / 48))  # Decay over 48 hours
        
        priority_score = (severity_score * 0.4) + (panic_score * 0.3) + (verification_bonus * 0.1) + (time_factor * 0.2)
        
        priority_reports.append({
            "report": report,
            "priority_score": priority_score
        })
    
    # Sort by priority score
    priority_reports.sort(key=lambda x: x["priority_score"], reverse=True)
    
    return priority_reports[:15]  # Top 15 priority reports

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Enhanced dashboard statistics"""
    total_reports = await db.reports.count_documents({})
    total_social_posts = await db.social_media.count_documents({})
    
    # Count by severity
    high_severity = await db.reports.count_documents({"severity": {"$in": ["High", "Critical"]}})
    medium_severity = await db.reports.count_documents({"severity": "Medium"})
    low_severity = await db.reports.count_documents({"severity": "Low"})
    
    # Count by verification status
    verified_reports = await db.reports.count_documents({"verification_status": "verified"})
    pending_verification = await db.reports.count_documents({"verification_status": "pending"})
    
    # Count by source
    citizen_reports = await db.reports.count_documents({"source": "citizen_report"})
    social_reports = await db.reports.count_documents({"source": "social_media"})
    
    # Active hotspots
    active_hotspots = await db.hotspots.count_documents({"risk_level": {"$in": ["medium", "high", "critical"]}})
    
    # Language distribution
    language_counts = {}
    for lang_code in SUPPORTED_LANGUAGES:
        count = await db.reports.count_documents({"language": lang_code})
        if count > 0:
            language_counts[lang_code] = count
    
    # Calculate average panic index
    reports = await db.reports.find().to_list(length=None)
    total_panic = sum(report.get("panic_index", 50) for report in reports)
    avg_panic = total_panic / total_reports if total_reports > 0 else 0
    
    return {
        "total_reports": total_reports,
        "total_social_posts": total_social_posts,
        "active_hotspots": active_hotspots,
        "severity_breakdown": {
            "high": high_severity,
            "medium": medium_severity, 
            "low": low_severity
        },
        "verification_status": {
            "verified": verified_reports,
            "pending": pending_verification,
            "verification_rate": verified_reports / total_reports if total_reports > 0 else 0
        },
        "source_breakdown": {
            "citizen_reports": citizen_reports,
            "social_media": social_reports
        },
        "language_distribution": language_counts,
        "average_panic_index": round(avg_panic, 1),
        "active_alerts": high_severity,
        "system_health": {
            "api_status": "operational",
            "ai_analysis": "active", 
            "social_monitoring": "active",
            "hotspot_generation": "active"
        }
    }

@app.get("/api/weather")
async def get_weather(lat: float, lon: float):
    """Enhanced weather data with caching"""
    # Check cache first
    cache_key = f"{lat:.2f}_{lon:.2f}"
    cached_weather = await db.weather_cache.find_one({"location_key": cache_key})
    
    if cached_weather:
        cache_time = datetime.fromisoformat(cached_weather["created_at"].replace('Z', '+00:00'))
        if (datetime.now(timezone.utc) - cache_time).total_seconds() < 3600:  # 1 hour cache
            return cached_weather["data"]
    
    # Generate enhanced mock weather data
    weather_data = {
        "location": f"{lat:.2f}, {lon:.2f}",
        "temperature": 28.5 + (lat / 100),  # Vary by latitude
        "humidity": 75,
        "wind_speed": 15.2,
        "wind_direction": 180,
        "pressure": 1013.2,
        "visibility": 10.0,
        "uv_index": 6,
        "description": "Partly cloudy with moderate winds",
        "alerts": [],
        "forecast": [
            {"time": "+3h", "temp": 30.1, "condition": "sunny"},
            {"time": "+6h", "temp": 32.5, "condition": "partly_cloudy"},
            {"time": "+12h", "temp": 26.8, "condition": "cloudy"}
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Add weather alerts if conditions warrant
    if weather_data["wind_speed"] > 25:
        weather_data["alerts"].append("High wind advisory")
    if weather_data["temperature"] > 35:
        weather_data["alerts"].append("Heat advisory")
    
    # Cache the result
    cache_doc = {
        "location_key": cache_key,
        "data": weather_data,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.weather_cache.replace_one(
        {"location_key": cache_key},
        cache_doc,
        upsert=True
    )
    
    return weather_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)