# Stage 1: Build React Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
ENV VITE_API_BASE_URL=""
RUN npm run build

# Stage 2: Python FastAPI Backend Runner
FROM python:3.11-slim

# Create user with UID 1000 (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user

# Install any basic build tools needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install Python requirements
COPY --chown=user:user backend/requirements.txt ./backend/
RUN pip install --no-cache-dir --user -r backend/requirements.txt

# Copy backend application
COPY --chown=user:user backend/ ./backend/

# Copy built frontend assets
COPY --chown=user:user --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port (default 7860, or dynamic PORT on cloud platforms like Render)
EXPOSE 7860

WORKDIR $HOME/app/backend

# Start FastAPI application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
