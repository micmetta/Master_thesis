# Use a base image with CUDA support for development
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files
COPY job.sh /app/job.sh
COPY script/inference_8B.py /app/script/inference_8B.py
COPY script/inference_70B.py /app/script/inference_70B.py
COPY script/utils /app/script/utils
COPY lib/bitsandbytes /app/lib/bitsandbytes
COPY lib/bitsandbytes /app/lib/bitsandbytes
COPY requirements.txt /app/requirements.txt

# Install git and other necessary system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake make

# Set CUDA_HOME for the CUDA toolkit
ENV CUDA_HOME=/usr/local/cuda

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Manually install Bitsandbytes from the provided directory
RUN cd /app/lib/bitsandbytes && \
    pip install -r requirements-dev.txt && \
    rm -rf CMakeCache.txt CMakeFiles && \
    cmake -DCOMPUTE_BACKEND=cuda -S . && \
    make && \
    pip install . && \
    python setup.py install --user

# Make job.sh file executable
RUN chmod +x /app/job.sh

# Set the default command for the container
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]