# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.12-slim AS builder

# Set environment variables to avoid writing .pyc files and to ensure output is not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system build dependencies (e.g., gcc is often needed for certain Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Runner (Final Image)
# ==========================================
FROM python:3.12-slim AS runner

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Create a non-root user and group for security
RUN addgroup --system appgroup && adduser --system --group appuser

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code and set ownership to the non-root user
COPY --chown=appuser:appgroup . .

# Switch to the non-root user
USER appuser

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application (Adjust 'main:app' to your file/app instance names)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]