#!/bin/bash

# Video Advertisement Placement Service Installation Script

echo "🚀 Installing Video Advertisement Placement Service..."

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [ "$major_version" -ge 3 ] && [ "$minor_version" -ge 8 ]; then
    echo "✅ Python $python_version is supported"
else
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create required directories
echo "📁 Creating required directories..."
mkdir -p logs uploads/advertisements

echo "✅ Directories created"

# Set executable permissions for run script
chmod +x run_server.py

echo "🎉 Installation complete!"
echo ""
echo "To start the service:"
echo "  python run_server.py"
echo ""
echo "Then visit:"
echo "  📖 API Documentation: http://localhost:8000/docs"
echo "  🏥 Health Check: http://localhost:8000/api/v1/health"
echo "  ℹ️  Service Info: http://localhost:8000/api/v1/info"
echo ""
echo "Happy coding! 🚀" 