# 🚀 Ocean Hazard Alert System - Deployment Guide

Complete deployment instructions for the Ocean Hazard Alert System on various platforms.

## 🎯 Quick Start (Recommended)

### **1. Automatic Setup**
```bash
# Clone the repository
git clone <your-repo-url>
cd ocean-hazard-alert-system

# Run the setup script
chmod +x setup.sh
./setup.sh
```

### **2. Manual Configuration**
```bash
# Configure backend environment
cd backend
cp ../.env.example .env
# Edit .env with your values:
# MONGO_URL=mongodb://localhost:27017
# EMERGENT_LLM_KEY=your-key-here

# Configure frontend environment  
cd ../frontend
cp .env.example .env
# Edit .env:
# REACT_APP_BACKEND_URL=http://localhost:8001
```

### **3. Start Services**
```bash
# Terminal 1 - Backend
./start_backend.sh

# Terminal 2 - Frontend  
./start_frontend.sh
```

## 🐳 Docker Deployment

### **Development with Docker Compose**
```bash
# Create environment file
echo "EMERGENT_LLM_KEY=your-key-here" > .env

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### **Production Docker**
```bash
# Build images
docker build -t ocean-hazard-backend ./backend
docker build -t ocean-hazard-frontend ./frontend

# Run with external database
docker run -d -p 8001:8001 \
  -e MONGO_URL=mongodb://your-mongodb:27017 \
  -e EMERGENT_LLM_KEY=your-key \
  ocean-hazard-backend

docker run -d -p 3000:3000 \
  -e REACT_APP_BACKEND_URL=http://your-backend:8001 \
  ocean-hazard-frontend
```

## ☁️ Cloud Platform Deployment

### **Heroku Deployment**

#### Backend (FastAPI)
```bash
# Install Heroku CLI
# Create Heroku app
heroku create ocean-hazard-api

# Add MongoDB addon
heroku addons:create mongolab:sandbox

# Set environment variables
heroku config:set EMERGENT_LLM_KEY=your-key

# Create Procfile
echo "web: uvicorn server:app --host 0.0.0.0 --port \$PORT" > backend/Procfile

# Deploy
git subtree push --prefix=backend heroku main
```

#### Frontend (React)
```bash
# Create frontend app
heroku create ocean-hazard-frontend

# Set buildpack
heroku buildpacks:set https://github.com/mars/create-react-app-buildpack.git

# Set environment
heroku config:set REACT_APP_BACKEND_URL=https://ocean-hazard-api.herokuapp.com

# Deploy
git subtree push --prefix=frontend heroku main
```

### **AWS Deployment**

#### Using AWS App Runner
```yaml
# apprunner.yaml (backend)
version: 1.0
runtime: python3
build:
  commands:
    build:
      - pip install -r requirements.txt
run:
  runtime-version: 3.9
  command: uvicorn server:app --host 0.0.0.0 --port 8000
  network:
    port: 8000
    env: PORT
env:
  - name: MONGO_URL
    value: "mongodb+srv://user:pass@cluster.mongodb.net/"
  - name: EMERGENT_LLM_KEY
    value: "your-key"
```

#### Using EC2 + RDS
```bash
# Launch EC2 instance
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start

# Clone and run
git clone <your-repo>
cd ocean-hazard-alert-system
docker-compose up -d
```

### **Google Cloud Platform**

#### Cloud Run Deployment
```bash
# Build and push images
gcloud builds submit --tag gcr.io/PROJECT-ID/ocean-hazard-backend ./backend
gcloud builds submit --tag gcr.io/PROJECT-ID/ocean-hazard-frontend ./frontend

# Deploy backend
gcloud run deploy ocean-hazard-api \
  --image gcr.io/PROJECT-ID/ocean-hazard-backend \
  --platform managed \
  --set-env-vars MONGO_URL=mongodb://...,EMERGENT_LLM_KEY=your-key

# Deploy frontend  
gcloud run deploy ocean-hazard-frontend \
  --image gcr.io/PROJECT-ID/ocean-hazard-frontend \
  --platform managed \
  --set-env-vars REACT_APP_BACKEND_URL=https://your-backend-url
```

### **Vercel (Frontend) + Railway (Backend)**

#### Vercel Frontend
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd frontend
vercel

# Set environment variable in Vercel dashboard:
# REACT_APP_BACKEND_URL=https://your-railway-backend.up.railway.app
```

#### Railway Backend
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway add mongodb
railway deploy

# Set environment variables in Railway dashboard
```

## 🗄️ Database Setup Options

### **Local MongoDB**
```bash
# Install MongoDB
# Ubuntu/Debian
sudo apt-get install mongodb

# macOS
brew install mongodb/brew/mongodb-community

# Start service
sudo systemctl start mongod  # Linux
brew services start mongodb-community  # macOS
```

### **MongoDB Atlas (Cloud)**
```bash
# 1. Create account at https://cloud.mongodb.com/
# 2. Create cluster
# 3. Get connection string
# 4. Update MONGO_URL in environment:
# MONGO_URL=mongodb+srv://username:password@cluster.mongodb.net/ocean_hazards
```

### **Docker MongoDB**
```bash
# Run MongoDB container
docker run -d \
  --name ocean-mongodb \
  -p 27017:27017 \
  -v mongodb_data:/data/db \
  mongo:5.0
```

## 🔧 Environment Configuration

### **Required Environment Variables**

#### Backend (.env)
```env
# Database
MONGO_URL=mongodb://localhost:27017

# AI Integration
EMERGENT_LLM_KEY=sk-emergent-your-key-here

# Weather API (Optional)
OPENWEATHER_API_KEY=your-openweather-key

# Security (Production)
SECRET_KEY=your-secret-key-for-jwt
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
```

#### Frontend (.env)
```env
# Backend API URL
REACT_APP_BACKEND_URL=http://localhost:8001

# Optional: Analytics
REACT_APP_GA_TRACKING_ID=your-google-analytics-id
```

## 🔒 Production Security Checklist

### **Backend Security**
- [ ] Set strong SECRET_KEY
- [ ] Configure ALLOWED_HOSTS
- [ ] Enable HTTPS/SSL
- [ ] Set up rate limiting
- [ ] Configure CORS properly
- [ ] Use environment variables for secrets
- [ ] Enable MongoDB authentication
- [ ] Set up monitoring and logging

### **Frontend Security**
- [ ] Build with production settings
- [ ] Configure CSP headers
- [ ] Enable HTTPS
- [ ] Implement proper authentication
- [ ] Sanitize user inputs
- [ ] Use HTTPS for API calls

### **Infrastructure Security**
- [ ] Set up firewall rules
- [ ] Configure SSL certificates
- [ ] Enable automatic security updates
- [ ] Set up monitoring and alerts
- [ ] Configure backup strategies
- [ ] Implement access controls

## 📊 Monitoring & Logging

### **Application Monitoring**
```python
# Add to backend/server.py
import logging
from fastapi import Request
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
    return response
```

### **Health Checks**
```bash
# Backend health
curl http://localhost:8001/api/health

# Frontend health (after build)
curl http://localhost:3000

# Database health
mongosh --eval "db.adminCommand('ping')"
```

## 🚨 Troubleshooting

### **Common Issues**

#### MongoDB Connection Failed
```bash
# Check MongoDB status
sudo systemctl status mongod

# Check connection
mongosh --eval "db.runCommand('ping')"

# Reset MongoDB
sudo systemctl restart mongod
```

#### Frontend Can't Connect to Backend
```bash
# Check backend is running
curl http://localhost:8001/api/health

# Check environment variables
echo $REACT_APP_BACKEND_URL

# Check CORS configuration in backend
```

#### Dependencies Installation Failed
```bash
# Clear caches
pip cache purge
yarn cache clean

# Reinstall
pip install -r requirements.txt
yarn install --frozen-lockfile
```

## 📈 Performance Optimization

### **Backend Optimization**
- Enable FastAPI caching
- Use database indexes
- Implement connection pooling
- Add Redis for caching
- Use async operations

### **Frontend Optimization**
- Enable code splitting
- Optimize images
- Use lazy loading
- Implement service workers
- Enable gzip compression

### **Database Optimization**
- Create proper indexes
- Use aggregation pipelines
- Implement data archiving
- Monitor query performance
- Set up read replicas

## 🔄 CI/CD Pipeline

The included GitHub Actions workflow (`.github/workflows/deploy.yml`) provides:

- **Automated Testing**: Runs on every push/PR
- **Build Verification**: Ensures both backend and frontend build successfully
- **Deployment**: Deploys to production on main branch

### **Customizing Deployment**
1. Update the workflow file with your deployment commands
2. Add secrets in GitHub repository settings
3. Configure your target deployment platform

---

## 📞 Support & Maintenance

### **Regular Maintenance Tasks**
- [ ] Update dependencies monthly
- [ ] Monitor disk usage and logs
- [ ] Backup database regularly
- [ ] Check security updates
- [ ] Review performance metrics

### **Getting Help**
- Check logs: `docker-compose logs -f`
- Health endpoints: `/api/health`
- Monitor resources: `docker stats`

**Successfully deployed Ocean Hazard Alert System! 🌊**