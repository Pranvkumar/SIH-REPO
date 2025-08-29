#!/bin/bash

# Ocean Hazard Alert System - Quick Setup Script
# This script sets up the development environment for GitHub users

echo "🌊 Ocean Hazard Alert System - Setup Script"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_status "Python found: $PYTHON_VERSION"
    else
        print_error "Python 3 is required but not installed. Please install Python 3.8+."
        exit 1
    fi
}

# Check if Node.js is installed
check_node() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Node.js found: $NODE_VERSION"
    else
        print_error "Node.js is required but not installed. Please install Node.js 16+."
        exit 1
    fi
}

# Check if yarn is installed
check_yarn() {
    if command -v yarn &> /dev/null; then
        YARN_VERSION=$(yarn --version)
        print_status "Yarn found: $YARN_VERSION"
    else
        print_warning "Yarn not found. Installing yarn..."
        npm install -g yarn
        if [ $? -eq 0 ]; then
            print_status "Yarn installed successfully"
        else
            print_error "Failed to install yarn. Please install manually."
            exit 1
        fi
    fi
}

# Setup backend environment
setup_backend() {
    print_info "Setting up backend environment..."
    
    cd backend
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install Python dependencies
    print_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        print_status "Backend dependencies installed"
    else
        print_error "Failed to install backend dependencies"
        exit 1
    fi
    
    # Setup environment file
    if [ ! -f ".env" ]; then
        print_info "Creating backend .env file..."
        cp ../.env.example .env
        print_warning "Please edit backend/.env file with your configurations"
        print_info "Required: MONGO_URL, EMERGENT_LLM_KEY"
    fi
    
    cd ..
}

# Setup frontend environment
setup_frontend() {
    print_info "Setting up frontend environment..."
    
    cd frontend
    
    # Install Node.js dependencies
    print_info "Installing Node.js dependencies..."
    yarn install
    
    if [ $? -eq 0 ]; then
        print_status "Frontend dependencies installed"
    else
        print_error "Failed to install frontend dependencies"
        exit 1
    fi
    
    # Setup environment file
    if [ ! -f ".env" ]; then
        print_info "Creating frontend .env file..."
        cp .env.example .env
        print_warning "Please edit frontend/.env file with your backend URL"
        print_info "Default: REACT_APP_BACKEND_URL=http://localhost:8001"
    fi
    
    cd ..
}

# Check MongoDB connection
check_mongodb() {
    print_info "Checking MongoDB connection..."
    
    # Try to connect to local MongoDB
    if command -v mongosh &> /dev/null; then
        mongosh --eval "db.runCommand('ping')" --quiet > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            print_status "MongoDB is running and accessible"
        else
            print_warning "MongoDB connection failed. Please ensure MongoDB is running."
            print_info "Start MongoDB: sudo systemctl start mongod (Linux) or brew services start mongodb/brew/mongodb-community (macOS)"
        fi
    else
        print_warning "mongosh not found. Please install MongoDB tools."
        print_info "Installation guide: https://docs.mongodb.com/manual/installation/"
    fi
}

# Create startup scripts
create_startup_scripts() {
    print_info "Creating startup scripts..."
    
    # Backend startup script
    cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Ocean Hazard Alert Backend..."
cd backend
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
EOF
    
    # Frontend startup script
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Ocean Hazard Alert Frontend..."
cd frontend
yarn start
EOF
    
    # Make scripts executable
    chmod +x start_backend.sh start_frontend.sh
    
    print_status "Startup scripts created (start_backend.sh, start_frontend.sh)"
}

# Main setup process
main() {
    echo ""
    print_info "Starting Ocean Hazard Alert System setup..."
    echo ""
    
    # System checks
    print_info "Checking system requirements..."
    check_python
    check_node
    check_yarn
    echo ""
    
    # Setup environments
    setup_backend
    echo ""
    setup_frontend
    echo ""
    
    # Check services
    check_mongodb
    echo ""
    
    # Create utilities
    create_startup_scripts
    echo ""
    
    # Final instructions
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Configure environment files:"
    echo "   - backend/.env (MongoDB URL, LLM API key)"
    echo "   - frontend/.env (Backend URL)"
    echo ""
    echo "2. Start the services:"
    echo "   Backend:  ./start_backend.sh"
    echo "   Frontend: ./start_frontend.sh"
    echo ""
    echo "3. Access the application:"
    echo "   Frontend: http://localhost:3000"
    echo "   Backend:  http://localhost:8001"
    echo "   API Docs: http://localhost:8001/docs"
    echo ""
    echo "🔐 Demo Login Credentials:"
    echo "   Citizen:   citizen / ocean123"
    echo "   Admin:     admin / admin123"
    echo "   Authority: authority / auth456"
    echo ""
    print_status "Happy coding! 🌊"
}

# Run main function
main