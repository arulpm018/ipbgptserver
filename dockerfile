# Use an official Python base image with CUDA support for GPU
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install Python and other essential dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set up a symbolic link for Python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Copy requirements.txt file to the container
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Install additional dependencies for GPU and huggingface transformers
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy the application code into the container
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Expose port 8000 to the outside world
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
