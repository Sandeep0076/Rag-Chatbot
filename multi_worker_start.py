#!/usr/bin/env python3
"""
Multi-worker startup script for the RAG PDF API.
This script starts the API with multiple worker processes to enable true concurrency.
"""
import argparse
import logging
import warnings

import uvicorn

# Suppress warnings from google-cloud-aiplatform and vertexai
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="google.cloud.aiplatform"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="vertexai._model_garden._model_garden_models"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start the RAG PDF API with multiple workers"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes (default: 4)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(
        f"Starting RAG PDF API with {args.workers} workers on port {args.port}"
    )

    # Start Uvicorn server with multiple workers
    # We import app directly from the module instead of using string reference
    # This is just an alternative approach that makes the relationship clearer
    uvicorn.run(
        "rtl_rag_chatbot_api.app:app",
        host="0.0.0.0",
        port=args.port,
        workers=args.workers,
        log_level="info",
        timeout_keep_alive=60,  # Keep same timeout as in app.py
    )
