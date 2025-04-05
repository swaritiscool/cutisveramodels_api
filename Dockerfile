FROM python:3.10-slim

# Avoid Python bytecode and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=100

# Force pip to prefer prebuilt binaries (especially for blis)
ENV PIP_INSTALL_OPTIONS="--prefer-binary"

# Install system build tools (just in case)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install $PIP_INSTALL_OPTIONS -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port 8000
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
