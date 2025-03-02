# Dockerfile for Python Tutorial Generator
FROM python:3.10-slim

# Install system dependencies 
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    imagemagick \
    zlib1g-dev \
    libjpeg-dev \
    libfreetype6-dev \
    espeak \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install moviepy to ensure it's properly installed
RUN pip install --no-cache-dir moviepy==1.0.3

# Explicitly install OpenAI with the correct version
RUN pip install --no-cache-dir openai==0.28.0

# Install the latest version of elevenlabs
RUN pip install --no-cache-dir elevenlabs --upgrade

# Set up ImageMagick policy to allow PDF operations and increase resource limits
RUN mkdir -p /etc/ImageMagick-6 && \
    echo '<policymap> \
    <policy domain="resource" name="memory" value="2GiB"/> \
    <policy domain="resource" name="disk" value="4GiB"/> \
    <policy domain="resource" name="map" value="1GiB"/> \
    <policy domain="resource" name="width" value="32KP"/> \
    <policy domain="resource" name="height" value="32KP"/> \
    <policy domain="resource" name="area" value="1GB"/> \
    <policy domain="resource" name="time" value="unlimited"/> \
    <policy domain="resource" name="throttle" value="0"/> \
    <policy domain="resource" name="thread" value="8"/> \
    <policy domain="coder" rights="read|write" pattern="PDF" /> \
    <policy domain="coder" rights="read|write" pattern="LABEL" /> \
    </policymap>' > /etc/ImageMagick-6/policy.xml

# Set up MoviePy to use ImageMagick
ENV IMAGEMAGICK_BINARY=/usr/bin/convert

# Copy the application code
COPY tutorial_generator.py .
COPY config.json .

# Create necessary directories
RUN mkdir -p scripts audio slides code_animations temp output assets

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
ENTRYPOINT ["python", "tutorial_generator.py"]
CMD ["--help"]