#!/bin/bash

# Update and install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev build-essential

# Install Python dependencies
pip3 install -r requirements.txt

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Confirm setup
echo "Setup complete. You can now run the app with: streamlit run dashboard.py"
