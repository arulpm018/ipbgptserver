#!/bin/bash

# Download embedding model and vector store from GCS
gsutil -m cp -r gs://[BUCKET_NAME]/model/* /app/models/embedding/
gsutil -m cp -r gs://[BUCKET_NAME]/vector_store/* /app/vector_store/

# Start the FastAPI application
exec uvicorn main:app --host 0.0.0.0 --port 8000