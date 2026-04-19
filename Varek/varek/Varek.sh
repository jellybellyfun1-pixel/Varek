#!/bin/bash
# Move to the exact project folder
cd ~/Desktop/Varek/varek

# Activate the virtual environment
source varek_env/bin/activate

# Launch the server using the environment's python
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
