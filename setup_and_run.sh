#!/bin/bash

# SmartVision Setup and Run Script
# Creates venv, installs dependencies, and runs the backend

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"

echo "======================================================================"
echo "SmartVision Vehicle Registration - Setup & Run"
echo "======================================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at: $VENV_DIR"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install/update requirements
echo ""
echo "Installing dependencies..."
pip install -r "$REQUIREMENTS" -q
echo "Dependencies installed"

# Create database directory
echo ""
echo "Setting up database directory..."
mkdir -p "$PROJECT_DIR/database"
echo "Database directory ready"

# Show database info
DB_PATH="$PROJECT_DIR/database/vehicle_registration.db"
echo ""
echo "======================================================================"
echo "Database Information"
echo "======================================================================"
echo "Database location: $DB_PATH"

if [ -f "$DB_PATH" ]; then
    DB_SIZE=$(du -h "$DB_PATH" | cut -f1)
    echo "Status: Database exists ($DB_SIZE)"
else
    echo "Status: Will be created on first run"
fi

# Database info
echo ""
echo "Using: SQLite (Local database)"

# Show what to do next
echo ""
echo "======================================================================"
echo "Starting Backend Server"
echo "======================================================================"
echo ""
echo "API will be available at:"
echo "   http://localhost:5000"
echo "   http://127.0.0.1:5000"
echo ""

# Get local IP for network access
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -n "$LOCAL_IP" ]; then
    echo "Access from phone (same WiFi):"
    echo "   http://$LOCAL_IP:5000"
    echo ""
fi

echo "Press Ctrl+C to stop the server"
echo ""
echo "======================================================================"
echo ""

# Run the backend
cd "$PROJECT_DIR/app"
python app.py

