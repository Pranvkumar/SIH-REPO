# 🌊 Ocean Hazard Alert System

A comprehensive full-stack web application for ocean hazard reporting, monitoring, and management with AI-powered severity classification.

![Ocean Hazard Alert System](https://img.shields.io/badge/Tech-Stack-blue?style=flat-square) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black) ![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white) ![AI](https://img.shields.io/badge/AI-Powered-orange?style=flat-square)

## ✨ Features

### 🚨 **Core Functionality**
- **Dynamic Login System** with role-based access (Citizen, Admin, Authority)
- **AI-Powered Hazard Classification** using GPT-4o-mini via Emergent LLM
- **Interactive Maps** with real-time hazard visualization
- **Weather Integration** with overlay data
- **Real-time Dashboard** with statistics and priority reports
- **Admin Panel** for authorities with report management

### 🎯 **Key Capabilities**
- Report ocean hazards with geolocation and media upload
- Automatic severity classification (Low/Medium/High)
- Panic index calculation (0-100 scale)
- Priority-based report ranking
- Interactive heatmap visualization
- Weather condition overlays
- Role-based permissions and access control

### 🔐 **Authentication System**
- **Citizen Access**: Report hazards and view public dashboard
- **Admin Access**: Full system access with administrative privileges  
- **Authority Access**: Monitor hazards and manage emergency responses
- Secure login with demo credentials for testing

## 🛠️ Tech Stack

### **Backend**
- **FastAPI** - High-performance Python web framework
- **MongoDB** - NoSQL database with Motor async driver
- **Pydantic** - Data validation and serialization
- **Emergent LLM Integration** - AI-powered text classification
- **CORS** enabled for cross-origin requests

### **Frontend** 
- **React.js** - Component-based UI library
- **TailwindCSS** - Utility-first CSS framework
- **Interactive Maps** - Custom map component with markers
- **Responsive Design** - Mobile-first approach
- **Real-time Updates** - Auto-refresh dashboard data

### **AI/ML Integration**
- **GPT-4o-mini** via Emergent LLM Key
- **Severity Classification** - Low/Medium/High categorization
- **Panic Index Scoring** - 0-100 risk assessment
- **Hazard Categorization** - Refined classification

## 🚀 Getting Started

### **Prerequisites**
- Python 3.8+
- Node.js 16+
- MongoDB (local or cloud instance)
- Git

### **Installation & Setup**

#### 1. **Clone the Repository**
```bash
git clone <your-repository-url>
cd ocean-hazard-alert-system
```

#### 2. **Backend Setup**
```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file with your configurations:
# MONGO_URL=mongodb://localhost:27017
# EMERGENT_LLM_KEY=your-emergent-llm-key
```

#### 3. **Frontend Setup**
```bash
# Navigate to frontend directory
cd ../frontend

# Install Node.js dependencies
yarn install
# or npm install

# Set up environment variables
cp .env.example .env
# Edit .env file:
# REACT_APP_BACKEND_URL=http://localhost:8001
```

#### 4. **Database Setup**
```bash
# Start MongoDB (if running locally)
mongod

# The application will automatically create required collections
```

### **Running the Application**

#### **Development Mode**

**Backend (Terminal 1):**
```bash
cd backend
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

**Frontend (Terminal 2):**
```bash
cd frontend
yarn start
# or npm start
```

#### **Production Mode**
```bash
# Backend
cd backend
uvicorn server:app --host 0.0.0.0 --port 8001

# Frontend (build and serve)
cd frontend
yarn build
# Serve the build folder with your preferred web server
```

### **Access the Application**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs

## 🔑 Demo Credentials

The application includes built-in demo accounts for testing:

| Role | Username | Password | Access Level |
|------|----------|----------|-------------|
| **Citizen** | `citizen` | `ocean123` | Report hazards, view dashboard |
| **Admin** | `admin` | `admin123` | Full system access |
| **Authority** | `authority` | `auth456` | Monitor and manage responses |

## 📡 API Endpoints

### **Authentication**
- `POST /api/auth/login` - User authentication
- `POST /api/auth/logout` - User logout

### **Hazard Reports**
- `GET /api/health` - Health check
- `POST /api/reports` - Create new hazard report
- `GET /api/reports` - Fetch all reports
- `GET /api/reports/priority` - Get priority-sorted reports
- `DELETE /api/reports/{id}` - Delete report (admin only)

### **Dashboard Data**
- `GET /api/dashboard/stats` - Dashboard statistics
- `GET /api/reports/heatmap` - Heatmap data
- `GET /api/weather` - Weather information

## 🎨 Screenshots & Demo

### **Login Page**
Beautiful gradient design with role-based authentication and demo access buttons.

### **Homepage**
Ocean-themed interface with clear navigation to report hazards or view dashboard.

### **Report Form**
Comprehensive form with geolocation, interactive map, and media upload capabilities.

### **Dashboard**
Real-time statistics, interactive maps with hazard markers, and priority reports sidebar.

### **Admin Panel**
Professional table interface for authorities to manage reports with filtering and deletion.

## 🔧 Configuration

### **Environment Variables**

**Backend (.env)**
```env
MONGO_URL=mongodb://localhost:27017
EMERGENT_LLM_KEY=your-emergent-llm-key
OPENWEATHER_API_KEY=your-openweather-key (optional)
```

**Frontend (.env)**
```env
REACT_APP_BACKEND_URL=http://localhost:8001
```

### **MongoDB Collections**
- `reports` - Hazard reports with AI classification
- `users` - User authentication data (if implemented)
- `weather_cache` - Cached weather data

## 🧪 Testing

### **Backend Testing**
```bash
cd backend
pytest test_api.py -v
```

### **Frontend Testing**
```bash
cd frontend
yarn test
# or npm test
```

### **API Testing**
Use the included test files or access the interactive API documentation at `/docs`.

## 📱 Mobile Responsiveness

The application is fully responsive and works seamlessly on:
- **Desktop** (1920px+)
- **Tablet** (768px - 1919px)
- **Mobile** (320px - 767px)

## 🔒 Security Features

- **Input validation** with Pydantic models
- **CORS protection** configured
- **Role-based access control**
- **SQL injection prevention** (NoSQL MongoDB)
- **XSS protection** through React's built-in sanitization

## 🚀 Deployment Options

### **Cloud Deployment**
- **Heroku**: Easy deployment with buildpacks
- **AWS**: EC2 + RDS + S3 for scalable infrastructure
- **Google Cloud**: App Engine + Cloud Storage
- **Azure**: App Service + Cosmos DB

### **Docker Deployment**
```dockerfile
# Example Dockerfile for backend
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📞 Support

For support and questions:
- **Issues**: Create a GitHub issue
- **Documentation**: Check the `/docs` API documentation
- **Community**: Join our discussions

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

## 🎯 Roadmap

### **Phase 1** ✅ Complete
- Basic hazard reporting
- AI classification
- Interactive dashboard
- Authentication system

### **Phase 2** 🚧 In Progress
- Social media integration
- Advanced weather APIs
- Mobile app (React Native)
- Real-time notifications

### **Phase 3** 📋 Planned
- Machine learning improvements
- Predictive analytics
- Multi-language support
- API rate limiting

---

## 🌊 **Ocean Hazard Alert System - Protecting Coastal Communities with AI** 🌊

**Built with ❤️ for maritime safety and emergency response**