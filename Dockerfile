# Dockerfile for Stock Prediction App
FROM python:3.10-slim

# Create and set working directory
WORKDIR /app

# Use temporary writable directories for Rust toolchain cache to avoid read-only errors
ENV CARGO_HOME=/tmp/cargo \
    RUSTUP_HOME=/tmp/rustup \
    CARGO_TARGET_DIR=/tmp/cargo-target

# Copy and install Python dependencies
COPY server/requirements.txt ./requirements.txt
# Upgrade pip and install only binary wheels to avoid Rust compilation
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Copy model file and application code
COPY Latest_stock_price_model.keras ./Latest_stock_price_model.keras
COPY server/ ./

# Expose Flask port
EXPOSE 5000

# Start the application
CMD ["python", "server.py"]
