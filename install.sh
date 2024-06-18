#!/bin/bash

# Create a Python virtual environment
python -m venv venv
# Activate the virtual environment
source venv/bin/activate

# Install other dependencies from requirements.txt
pip install -r requirements.txt
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118

python xtts_demo.py

