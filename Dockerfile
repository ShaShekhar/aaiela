# Define an argument for the CUDA version
ARG CUDA_VERSION=11.8

# Base image based on the CUDA version
FROM nvidia/cuda:${CUDA_VERSION}.0-cudnn8-runtime-ubuntu20.04

# Environment Setup
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Install system dependencies and Conda
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    wget bzip2 libgl1 \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    nano tmux \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    conda clean -a -y

# Create and activate environment
RUN conda create -n aaiela python=3.9 -y
SHELL ["conda", "run", "-n", "aaiela", "/bin/bash", "-c"]

# Install PyTorch, torchvision, torchaudio with the correct CUDA version
RUN conda install pytorch torchvision torchaudio pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia -y

# Install your project's Python dependencies (except ctranslate2)
RUN pip install --no-cache-dir \
    google-generativeai \
    tokenizers \
    transformers \
    openai \
    ftfy \
    einops \
    pillow \
    omegaconf \
    opencv-python \
    flask flask_cors python-dotenv

# Install ctranslate2 based on the CUDA version
RUN if [ "$CUDA_VERSION" = "11.8.0" ]; then \
    pip install --no-cache-dir ctranslate2==3.24.0; \
    elif [ "$CUDA_VERSION" = "12.1.0" ]; then \
    pip install --no-cache-dir ctranslate2; \
    else \
    echo "Invalid CUDA version specified. Please use 11.8 or 12.1."; \
    exit 1; \
    fi

# Copy the entire contents of the project into the Docker image
COPY . /app

# Install detectron2
WORKDIR /app/models
RUN python -m pip install -e detectron2
WORKDIR /app

# Initialize conda for the bash shell
RUN conda init bash
# Expose the port for Flask (adjust if needed)
EXPOSE 5000