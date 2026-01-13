#!/bin/bash

# Launch Waste Classification GUI
echo "Starting Waste Classification GUI..."

cd "$(dirname "$0")/src"
python gui.py
