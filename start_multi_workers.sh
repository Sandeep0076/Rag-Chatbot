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

# Activate Poetry environment and run the multi-worker script
echo "Activating Poetry environment and loading environment variables..."
poetry run python -c "
import os
import sys
import subprocess

# Load environment variables from env_variables.txt if it exists
env_file = 'env_variables.txt'
if os.path.exists(env_file):
    print(f'Loading environment variables from {env_file}')
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            os.environ[key] = value
else:
    print(f'Warning: {env_file} not found. Make sure environment variables are set.')

# Start the API with multiple workers
workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
print(f'Starting RAG PDF API with {workers} workers')
subprocess.run(['python', 'multi_worker_start.py', '--workers', str(workers)])
" $WORKERS
