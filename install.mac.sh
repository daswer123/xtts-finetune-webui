#!/bin/sh

python -m venv venv
source ./venv/bin/activate

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
pip install -r requirements.txt

python xtts_demo.py