#!/bin/bash

# Start Backend
echo "Starting Backend..."
uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start Frontend
echo "Starting Frontend..."
streamlit run frontend/app.py

# Cleanup
kill $BACKEND_PID
