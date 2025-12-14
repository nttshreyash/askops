#!/bin/bash
set -e

echo "Starting backend (uvicorn) on 127.0.0.1:8000..."
python -m uvicorn backend:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Waiting for backend to be ready..."
sleep 3

echo "Starting nginx..."
nginx -g 'daemon off;' &
NGINX_PID=$!

# Function to handle shutdown
shutdown() {
    echo "Shutting down..."
    kill $NGINX_PID 2>/dev/null || true
    kill $BACKEND_PID 2>/dev/null || true
    exit 0
}

trap shutdown SIGTERM SIGINT

# Wait for either process to exit
wait -n $BACKEND_PID $NGINX_PID

# If we get here, one of the processes died, so kill the other
kill $NGINX_PID 2>/dev/null || true
kill $BACKEND_PID 2>/dev/null || true

echo "One of the processes exited. Shutting down..."
exit 1
