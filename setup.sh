#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y \
    python3-pip \
    python3.9 \
    git \
    nvidia-driver-525

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-docker2
systemctl restart docker

# Create app directory
mkdir -p /app
cd /app

# Install Google Cloud SDK
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-442.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-cli-442.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh --quiet

# Clone your application (replace with your repo)
git clone https://github.com/arulpm018/ipbgptserver.git
cd ipbgptserver

# Download models and vector store
mkdir -p model
mkdir -p vector_store
gsutil -m cp -r gs://ipbgptbucket/model/* model/
gsutil -m cp -r gs://ipbgptbucket/vector_store/* vector_store/

# Build and run Docker container
docker build -t rag-app .
docker run -d --gpus all -p 80:8000 rag-app
