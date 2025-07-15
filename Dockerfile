# Dockerfile for Stock Prediction App
FROM python:3.10-slim

# Create and set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY server/requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy model file and application code
COPY Latest_stock_price_model.keras ./Latest_stock_price_model.keras
COPY server/ ./

# Expose Flask port
EXPOSE 5000

# Start the application
CMD ["python", "server.py"]
