#!/bin/bash

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install prerequisites
echo "Installing prerequisites..."
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    python3-pip \
    python3.9

# Add Docker's official GPG key
echo "Adding Docker GPG key..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up Docker repository
echo "Setting up Docker repository..."
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
echo "Installing Docker..."
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Add current user to docker group
echo "Adding user to docker group..."
sudo usermod -aG docker $USER

# Install NVIDIA drivers
echo "Installing NVIDIA drivers..."
sudo apt-get install -y nvidia-driver-525

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker daemon
echo "Restarting Docker daemon..."
sudo systemctl restart docker

# Verify installations
echo "Verifying installations..."
docker --version
nvidia-smi

# Create application directories
echo "Creating application directories..."
sudo mkdir -p /app
sudo chown -R $USER:$USER /app
cd /app

# Install Google Cloud SDK
echo "Installing Google Cloud SDK..."
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-442.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-cli-442.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh --quiet

# Create directories for models and vector store
echo "Creating model and vector store directories..."
mkdir -p models/embedding
mkdir -p vector_store

echo "Installation complete!"