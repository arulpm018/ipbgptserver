FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and vector store
RUN mkdir -p /app/models/embedding
RUN mkdir -p /app/vector_store

# Download model and vector store from GCS at startup
COPY startup.sh .
RUN chmod +x startup.sh

EXPOSE 8000

# Use startup script as entrypoint
ENTRYPOINT ["./startup.sh"]
