// MongoDB initialization script for Ocean Hazard Alert System

// Create database
db = db.getSiblingDB('ocean_hazards');

// Create collections with proper indexing
db.createCollection('reports');
db.createCollection('users');
db.createCollection('weather_cache');

// Create indexes for better performance
db.reports.createIndex({ "location.latitude": 1, "location.longitude": 1 });
db.reports.createIndex({ "created_at": -1 });
db.reports.createIndex({ "severity": 1 });
db.reports.createIndex({ "hazard_type": 1 });
db.reports.createIndex({ "panic_index": -1 });

// Create geospatial index for location queries
db.reports.createIndex({ 
  "location": "2dsphere" 
});

// Create text index for search functionality
db.reports.createIndex({
  "description": "text",
  "hazard_type": "text",
  "name": "text"
});

// Weather cache TTL index (expire after 1 hour)
db.weather_cache.createIndex(
  { "created_at": 1 }, 
  { expireAfterSeconds: 3600 }
);

// Insert sample data for testing
db.reports.insertMany([
  {
    id: "sample-1",
    name: "Sample Reporter",
    location: {
      latitude: -33.8688,
      longitude: 151.2093,
      address: "Sydney Harbour, NSW, Australia"
    },
    hazard_type: "Cyclone",
    description: "Severe weather conditions observed in coastal waters. High winds and dangerous waves reported.",
    severity: "High",
    panic_index: 85,
    ai_category: "Cyclone",
    created_at: new Date().toISOString(),
    status: "pending"
  },
  {
    id: "sample-2", 
    name: "Coast Guard",
    location: {
      latitude: 40.7128,
      longitude: -74.0060,
      address: "New York Harbor, NY, USA"
    },
    hazard_type: "Oil Spill",
    description: "Minor oil leak detected from vessel. Containment measures in progress.",
    severity: "Medium",
    panic_index: 45,
    ai_category: "Oil Spill",
    created_at: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
    status: "reviewed"
  }
]);

print('Ocean Hazard Alert System database initialized successfully!');
print('Collections created: reports, users, weather_cache');
print('Indexes created for optimal performance');
print('Sample data inserted for testing');