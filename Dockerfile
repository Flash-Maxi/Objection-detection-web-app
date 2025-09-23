# Use official slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system packages required for Pillow & Torchvision
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port (Render/Heroku will map this)
EXPOSE 5000

# Use gunicorn in production instead of app.run()
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
