#!/bin/bash

# Complete system launch - API + GUI
echo "================================================"
echo " Waste Classification - Complete System Launch"
echo "================================================"
echo ""

cd "$(dirname "$0")/src"

# Start API server in background
echo "Starting API server..."
python api.py > api.log 2>&1 &
API_PID=$!

echo "API server started (PID: $API_PID)"
echo "Waiting for API to initialize..."
sleep 5

# Start GUI
echo ""
echo "Starting GUI..."
python gui.py

# Cleanup - kill API server when GUI closes
echo ""
echo "Shutting down API server..."
kill $API_PID 2>/dev/null

echo "System shutdown complete."
