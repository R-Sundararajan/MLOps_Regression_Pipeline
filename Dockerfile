FROM python:3.11-slim

# Prevent Python from creating .pyc files
# and ensure logs are immediately visible
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Application directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI application
COPY server/ ./server/

# Copy registered/champion model
COPY models/champion_model.joblib ./models/champion_model.joblib

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]