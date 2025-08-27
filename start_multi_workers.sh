#!/bin/bash
# Shell script to start the RAG PDF API with multiple workers
# This handles Poetry activation and environment variable loading automatically

# Default number of workers
WORKERS=4

# Check if a parameter was provided
if [ ! -z "$1" ]; then
    WORKERS=$1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Please install Poetry first."
    exit 1
fi
# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    set -o allexport
    source .env
    set +o allexport
fi

# Activate Poetry environment and run the multi-worker script
echo "Activating Poetry environment and loading environment variables..."
poetry run python -c "
import os
import sys
import subprocess

# Start the API with multiple workers
workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
print(f'Starting RAG PDF API with {workers} workers')
subprocess.run(['python', 'multi_worker_start.py', '--workers', str(workers)])
" $WORKERS
