#!/bin/bash


sudo mkdir -p model
sudo mkdir -p vector_store
sudo gsutil -m cp -r gs://ipbgptbucket/model/* model/
sudo gsutil -m cp -r gs://ipbgptbucket/vector_store/* vector_store/

sudo git clone https://github.com/arulpm018/ipbgptserver.git
pip install -r requirements.txt

# Start the FastAPI application
exec uvicorn main:app --host 0.0.0.0 --port 8000
