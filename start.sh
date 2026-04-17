#!/bin/bash

# Start the edge simulator in the background
python edge_simulator.py &

# Start the FastAPI backend in the foreground
uvicorn backend:app --host 0.0.0.0 --port 8080
