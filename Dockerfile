# Use official Python image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (optional: add more if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching layers)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app

# Default command: run tests
CMD ["pytest", "Test_cases.py", "--maxfail=1", "--disable-warnings", "-q"]
