#!/bin/bash
cd ~/Desktop/Varek/varek
source varek_env/bin/activate
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
