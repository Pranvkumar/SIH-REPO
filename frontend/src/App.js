import React, { useState, useEffect, useCallback } from 'react';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

// Map Component (Simple Leaflet alternative using basic markers)
const Map = ({ reports, onLocationSelect, selectedLocation, weatherData }) => {
  const [mapCenter, setMapCenter] = useState({ lat: 20.5937, lng: 78.9629 }); // Center of India
  
  // Mock map component - in production would use Leaflet or Google Maps
  return (
    <div className="relative w-full h-96 bg-blue-100 border-2 border-blue-300 rounded-lg overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-blue-200 to-blue-400"></div>
      
      {/* Weather overlay */}
      {weatherData && (
        <div className="absolute top-4 left-4 bg-white bg-opacity-90 p-3 rounded-lg shadow-lg">
          <h4 className="font-semibold text-sm">Weather Conditions</h4>
          <p className="text-xs">Temp: {weatherData.temperature}°C</p>
          <p className="text-xs">Wind: {weatherData.wind_speed} km/h</p>
          <p className="text-xs">{weatherData.description}</p>
        </div>
      )}
      
      {/* Selected location marker */}
      {selectedLocation && (
        <div 
          className="absolute w-4 h-4 bg-red-500 rounded-full border-2 border-white transform -translate-x-2 -translate-y-2"
          style={{
            left: `${((selectedLocation.lng + 180) / 360) * 100}%`,
            top: `${((90 - selectedLocation.lat) / 180) * 100}%`
          }}
        />
      )}
      
      {/* Hazard markers */}
      {reports.map((report, index) => {
        const severityColors = {
          'High': 'bg-red-600',
          'Medium': 'bg-yellow-500',
          'Low': 'bg-green-500'
        };
        
        return (
          <div
            key={report.id}
            className={`absolute w-3 h-3 ${severityColors[report.severity] || 'bg-gray-500'} rounded-full border border-white transform -translate-x-1 -translate-y-1 animate-pulse`}
            style={{
              left: `${((report.location.longitude + 180) / 360) * 100}%`,
              top: `${((90 - report.location.latitude) / 180) * 100}%`
            }}
            title={`${report.hazard_type} - ${report.severity}`}
          />
        );
      })}
      
      {/* Click handler overlay */}
      <div 
        className="absolute inset-0 cursor-crosshair"
        onClick={(e) => {
          if (onLocationSelect) {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;
            const lng = (x * 360) - 180;
            const lat = 90 - (y * 180);
            onLocationSelect({ lat, lng });
          }
        }}
      />
      
      <div className="absolute bottom-4 right-4 bg-white bg-opacity-90 p-2 rounded text-xs">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-2 h-2 bg-red-600 rounded-full"></div>
          <span>High Severity</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
          <span>Medium Severity</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span>Low Severity</span>
        </div>
      </div>
    </div>
  );
};

// Home Page Component
const HomePage = ({ onNavigate }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-blue-800 to-cyan-700 text-white">
      <div className="container mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
            Ocean Hazard Alert Platform
          </h1>
          <p className="text-xl text-blue-200 max-w-3xl mx-auto">
            Advanced early warning system for ocean hazards with AI-powered severity classification and real-time monitoring
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-8 border border-white border-opacity-20">
            <div className="text-center mb-6">
              <div className="w-16 h-16 bg-gradient-to-r from-red-500 to-orange-500 rounded-full mx-auto mb-4 flex items-center justify-center">
                <svg className="w-8 h-8" fill="white" viewBox="0 0 24 24">
                  <path d="M12 2L13.09 8.26L22 9L13.09 9.74L12 16L10.91 9.74L2 9L10.91 8.26L12 2Z"/>
                </svg>
              </div>
              <h2 className="text-2xl font-bold mb-2">Report Ocean Hazard</h2>
              <p className="text-blue-200 mb-6">Submit hazard reports with AI-powered severity analysis</p>
            </div>
            <button
              onClick={() => onNavigate('report')}
              className="w-full bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-700 hover:to-orange-700 text-white font-bold py-4 px-6 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg"
            >
              Report Hazard
            </button>
          </div>
          
          <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-8 border border-white border-opacity-20">
            <div className="text-center mb-6">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full mx-auto mb-4 flex items-center justify-center">
                <svg className="w-8 h-8" fill="white" viewBox="0 0 24 24">
                  <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
                </svg>
              </div>
              <h2 className="text-2xl font-bold mb-2">View Dashboard</h2>
              <p className="text-blue-200 mb-6">Monitor hazards on interactive maps with real-time data</p>
            </div>
            <button
              onClick={() => onNavigate('dashboard')}
              className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-bold py-4 px-6 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg"
            >
              View Dashboard
            </button>
          </div>
        </div>
        
        <div className="mt-12 text-center">
          <button
            onClick={() => onNavigate('admin')}
            className="bg-white bg-opacity-20 hover:bg-opacity-30 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-300"
          >
            Admin Panel
          </button>
        </div>
      </div>
    </div>
  );
};

// Report Hazard Form Component
const ReportPage = ({ onNavigate }) => {
  const [formData, setFormData] = useState({
    name: '',
    latitude: '',
    longitude: '',
    address: '',
    hazard_type: 'Cyclone',
    description: ''
  });
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [media, setMedia] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);

  // Auto-detect geolocation
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setFormData(prev => ({
            ...prev,
            latitude: latitude.toFixed(6),
            longitude: longitude.toFixed(6)
          }));
          setSelectedLocation({ lat: latitude, lng: longitude });
        },
        (error) => {
          console.log('Geolocation error:', error);
        }
      );
    }
  }, []);

  const handleLocationSelect = (location) => {
    setSelectedLocation(location);
    setFormData(prev => ({
      ...prev,
      latitude: location.lat.toFixed(6),
      longitude: location.lng.toFixed(6)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      const submitFormData = new FormData();
      submitFormData.append('name', formData.name);
      submitFormData.append('latitude', parseFloat(formData.latitude));
      submitFormData.append('longitude', parseFloat(formData.longitude));
      submitFormData.append('address', formData.address);
      submitFormData.append('hazard_type', formData.hazard_type);
      submitFormData.append('description', formData.description);
      
      if (media) {
        submitFormData.append('media', media);
      }
      
      const response = await fetch(`${BACKEND_URL}/api/reports`, {
        method: 'POST',
        body: submitFormData
      });
      
      if (response.ok) {
        setSuccess(true);
        setTimeout(() => {
          onNavigate('dashboard');
        }, 2000);
      } else {
        throw new Error('Failed to submit report');
      }
    } catch (error) {
      console.error('Error submitting report:', error);
      alert('Failed to submit report. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-900 to-blue-900 flex items-center justify-center">
        <div className="bg-white rounded-2xl p-8 text-center max-w-md">
          <div className="w-16 h-16 bg-green-500 rounded-full mx-auto mb-4 flex items-center justify-center">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Report Submitted Successfully!</h2>
          <p className="text-gray-600 mb-4">AI is analyzing your report for severity classification.</p>
          <p className="text-sm text-gray-500">Redirecting to dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-900 to-orange-800 py-8">
      <div className="container mx-auto px-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">Report Ocean Hazard</h1>
            <p className="text-orange-200">Help protect coastal communities by reporting hazards</p>
          </div>
          
          <div className="bg-white rounded-2xl shadow-2xl p-8">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Your Name</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
                    required
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Hazard Type</label>
                  <select
                    value={formData.hazard_type}
                    onChange={(e) => setFormData(prev => ({ ...prev, hazard_type: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
                  >
                    <option value="Cyclone">Cyclone</option>
                    <option value="Oil Spill">Oil Spill</option>
                    <option value="Flood">Flood</option>
                    <option value="Tsunami">Tsunami</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>
              
              <div className="grid md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Latitude</label>
                  <input
                    type="number"
                    step="0.000001"
                    value={formData.latitude}
                    onChange={(e) => setFormData(prev => ({ ...prev, latitude: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
                    required
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Longitude</label>
                  <input
                    type="number"
                    step="0.000001"
                    value={formData.longitude}
                    onChange={(e) => setFormData(prev => ({ ...prev, longitude: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
                    required
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Address (Optional)</label>
                  <input
                    type="text"
                    value={formData.address}
                    onChange={(e) => setFormData(prev => ({ ...prev, address: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Location (Click on map to select)
                </label>
                <Map 
                  reports={[]} 
                  onLocationSelect={handleLocationSelect}
                  selectedLocation={selectedLocation}
                />
              </div>
              
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Description</label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                  rows={4}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
                  placeholder="Describe the hazard in detail..."
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Media Upload (Optional)</label>
                <input
                  type="file"
                  accept="image/*,video/*"
                  onChange={(e) => setMedia(e.target.files[0])}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
                />
              </div>
              
              <div className="flex gap-4">
                <button
                  type="button"
                  onClick={() => onNavigate('home')}
                  className="flex-1 bg-gray-300 hover:bg-gray-400 text-gray-700 font-bold py-3 px-6 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="flex-1 bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-700 hover:to-orange-700 text-white font-bold py-3 px-6 rounded-lg transition-all duration-300 disabled:opacity-50"
                >
                  {isSubmitting ? 'Submitting...' : 'Submit Report'}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

// Dashboard Component
const Dashboard = ({ onNavigate }) => {
  const [reports, setReports] = useState([]);
  const [priorityReports, setPriorityReports] = useState([]);
  const [stats, setStats] = useState(null);
  const [weatherData, setWeatherData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const [reportsRes, priorityRes, statsRes] = await Promise.all([
        fetch(`${BACKEND_URL}/api/reports`),
        fetch(`${BACKEND_URL}/api/reports/priority`),
        fetch(`${BACKEND_URL}/api/dashboard/stats`)
      ]);
      
      const reportsData = await reportsRes.json();
      const priorityData = await priorityRes.json();
      const statsData = await statsRes.json();
      
      setReports(reportsData);
      setPriorityReports(priorityData);
      setStats(statsData);
      
      // Fetch weather for first report location
      if (reportsData.length > 0) {
        const firstReport = reportsData[0];
        const weatherRes = await fetch(
          `${BACKEND_URL}/api/weather?lat=${firstReport.location.latitude}&lon=${firstReport.location.longitude}`
        );
        if (weatherRes.ok) {
          const weatherData = await weatherRes.json();
          setWeatherData(weatherData);
        }
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [fetchData]);

  const getPanicLevelColor = (panicIndex) => {
    if (panicIndex >= 70) return 'text-red-600 bg-red-100';
    if (panicIndex >= 40) return 'text-yellow-600 bg-yellow-100';
    return 'text-green-600 bg-green-100';
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'High': return 'bg-red-500 text-white';
      case 'Medium': return 'bg-yellow-500 text-white';
      case 'Low': return 'bg-green-500 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-900 to-cyan-800 flex items-center justify-center">
        <div className="text-white text-xl">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 to-cyan-800">
      <div className="container mx-auto px-6 py-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-4xl font-bold text-white">Ocean Hazard Dashboard</h1>
          <button
            onClick={() => onNavigate('home')}
            className="bg-white bg-opacity-20 hover:bg-opacity-30 text-white font-semibold py-2 px-4 rounded-lg"
          >
            Home
          </button>
        </div>
        
        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-xl p-6 border border-white border-opacity-20">
              <h3 className="text-lg font-semibold text-white mb-2">Total Reports</h3>
              <p className="text-3xl font-bold text-cyan-300">{stats.total_reports}</p>
            </div>
            
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-xl p-6 border border-white border-opacity-20">
              <h3 className="text-lg font-semibold text-white mb-2">Active Alerts</h3>
              <p className="text-3xl font-bold text-red-300">{stats.active_alerts}</p>
            </div>
            
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-xl p-6 border border-white border-opacity-20">
              <h3 className="text-lg font-semibold text-white mb-2">Avg Panic Index</h3>
              <p className={`text-3xl font-bold ${getPanicLevelColor(stats.average_panic_index)}`}>
                {stats.average_panic_index}
              </p>
            </div>
            
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-xl p-6 border border-white border-opacity-20">
              <h3 className="text-lg font-semibold text-white mb-2">High Severity</h3>
              <p className="text-3xl font-bold text-red-300">{stats.severity_breakdown.high}</p>
            </div>
          </div>
        )}
        
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Map */}
          <div className="lg:col-span-2">
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-xl p-6 border border-white border-opacity-20">
              <h2 className="text-xl font-bold text-white mb-4">Hazard Map</h2>
              <Map reports={reports} weatherData={weatherData} />
            </div>
          </div>
          
          {/* Priority Reports Sidebar */}
          <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-xl p-6 border border-white border-opacity-20">
            <h2 className="text-xl font-bold text-white mb-4">Priority Reports</h2>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {priorityReports.map(({ report, priority_score }) => (
                <div key={report.id} className="bg-white bg-opacity-20 rounded-lg p-4">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-semibold text-white text-sm">{report.hazard_type}</h3>
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${getSeverityColor(report.severity)}`}>
                      {report.severity}
                    </span>
                  </div>
                  <p className="text-blue-200 text-xs mb-2">{report.description.substring(0, 100)}...</p>
                  <div className="flex justify-between text-xs">
                    <span className="text-blue-300">Panic: {report.panic_index}</span>
                    <span className="text-blue-300">Priority: {priority_score.toFixed(1)}</span>
                  </div>
                  <div className="text-xs text-blue-400 mt-1">
                    By: {report.name} | {new Date(report.created_at).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Admin Panel Component
const AdminPanel = ({ onNavigate }) => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/reports`);
      const data = await response.json();
      setReports(data);
    } catch (error) {
      console.error('Error fetching reports:', error);
    } finally {
      setLoading(false);
    }
  };

  const deleteReport = async (reportId) => {
    if (window.confirm('Are you sure you want to delete this report?')) {
      try {
        const response = await fetch(`${BACKEND_URL}/api/reports/${reportId}`, {
          method: 'DELETE'
        });
        if (response.ok) {
          fetchReports();
        }
      } catch (error) {
        console.error('Error deleting report:', error);
      }
    }
  };

  const filteredReports = reports.filter(report => {
    if (filter === 'all') return true;
    return report.hazard_type.toLowerCase() === filter.toLowerCase();
  });

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'High': return 'bg-red-100 text-red-800';
      case 'Medium': return 'bg-yellow-100 text-yellow-800';
      case 'Low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-xl">Loading admin panel...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="container mx-auto px-6 py-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">Admin Panel</h1>
          <button
            onClick={() => onNavigate('home')}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg"
          >
            Home
          </button>
        </div>
        
        {/* Filters */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex flex-wrap gap-4">
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="all">All Hazard Types</option>
              <option value="cyclone">Cyclone</option>
              <option value="oil spill">Oil Spill</option>
              <option value="flood">Flood</option>
              <option value="tsunami">Tsunami</option>
              <option value="other">Other</option>
            </select>
          </div>
        </div>
        
        {/* Reports Table */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Reporter
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Hazard Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Severity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Panic Index
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Location
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredReports.map((report) => (
                  <tr key={report.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {report.name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {report.hazard_type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getSeverityColor(report.severity)}`}>
                        {report.severity}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {report.panic_index}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {report.location.latitude.toFixed(4)}, {report.location.longitude.toFixed(4)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {new Date(report.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <button
                        onClick={() => deleteReport(report.id)}
                        className="text-red-600 hover:text-red-900"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const [currentPage, setCurrentPage] = useState('home');

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage onNavigate={setCurrentPage} />;
      case 'report':
        return <ReportPage onNavigate={setCurrentPage} />;
      case 'dashboard':
        return <Dashboard onNavigate={setCurrentPage} />;
      case 'admin':
        return <AdminPanel onNavigate={setCurrentPage} />;
      default:
        return <HomePage onNavigate={setCurrentPage} />;
    }
  };

  return (
    <div className="App">
      {renderCurrentPage()}
    </div>
  );
}

export default App;