#!/bin/bash

# Setup script for Waste Classification project
# This script creates a virtual environment and installs all dependencies

echo "Setting up Waste Classification environment..."

# Detect activation script path (Windows vs Unix)
if [ -f "venv/Scripts/activate" ]; then
    VENV_ACTIVATE="venv/Scripts/activate"
else
    VENV_ACTIVATE="venv/bin/activate"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    # Create virtual environment
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    # Detect activation script path again after creation
    if [ -f "venv/Scripts/activate" ]; then
        VENV_ACTIVATE="venv/Scripts/activate"
    else
        VENV_ACTIVATE="venv/bin/activate"
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source "$VENV_ACTIVATE"
    
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    
    echo ""
    echo "Setup complete!"
    echo ""
else
    echo "Virtual environment already exists. Activating..."
    source "$VENV_ACTIVATE"
    echo "Virtual environment activated!"
    echo ""
fi

echo "To deactivate when done, run:"
echo "  deactivate"
