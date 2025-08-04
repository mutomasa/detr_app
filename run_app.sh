#!/bin/bash

# DETR Streamlit Application Launcher
# This script sets up and runs the DETR visualization application

echo "🎯 DETR Visualization Application Launcher"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "✅ uv found: $(uv --version)"
    USE_UV=true
else
    echo "⚠️  uv not found, using pip"
    USE_UV=false
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."

if [ "$USE_UV" = true ]; then
    echo "Using uv to install dependencies..."
    uv sync
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies with uv"
        exit 1
    fi
else
    echo "Using pip to install dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies with pip"
        exit 1
    fi
fi

echo "✅ Dependencies installed successfully"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found in current directory"
    exit 1
fi

echo ""
echo "🎯 Starting DETR Visualization Application..."
echo "📍 URL: http://localhost:8504"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Run the application
if [ "$USE_UV" = true ]; then
    uv run streamlit run app.py --server.port 8504 --server.address 0.0.0.0
else
    streamlit run app.py --server.port 8504 --server.address 0.0.0.0
fi 