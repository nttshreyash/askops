# syntax=docker/dockerfile:1

# --- Stage 1: build the React frontend (Vite) ---
FROM node:18-alpine AS frontend-build

WORKDIR /app/askgen-react

# Copy only package files first for better layer caching
COPY askgen-react/package*.json ./
RUN npm ci --silent

# Copy the rest of the frontend source
COPY askgen-react/ ./

# Build the frontend (Vite outputs to dist by default)
RUN npm run build

# --- Stage 2: runtime with Python + Nginx ---
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install nginx and basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy python requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt uvicorn[standard]

# Copy backend code and RAG engine
COPY backend.py /app/backend.py
COPY rag_engine/ /app/rag_engine/
COPY .env /app/.env

# Create directory for ChromaDB persistence
RUN mkdir -p /app/chroma_db && chmod 777 /app/chroma_db

# Copy built frontend (Vite outputs to dist only)
COPY --from=frontend-build /app/askgen-react/dist /usr/share/nginx/html
COPY --from=frontend-build /app/askgen-react/dist /app/askgen-react/dist

# Copy nginx config and start script
COPY nginx.conf /etc/nginx/nginx.conf
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Create nginx directories with proper permissions for Azure
RUN mkdir -p /var/log/nginx /var/lib/nginx/body /var/lib/nginx/proxy \
    && touch /var/run/nginx.pid \
    && chmod -R 777 /var/log/nginx /var/lib/nginx /var/run/nginx.pid

EXPOSE 80 8000

# Azure Container Registry typically runs as root
CMD ["/start.sh"]
