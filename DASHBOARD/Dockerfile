# ── Stage 1: Build the Vite frontend ──
FROM node:20-slim AS frontend-build
WORKDIR /build
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build
# Vite output: /build/../app/static/dist (relative to /build)
# Resolves to /app/static/dist on disk

# ── Stage 2: Python backend ──
FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application (includes app/static/data/ atlas + mesh files)
COPY app/ ./app/

# Drop the built frontend assets into static/dist/ on top of the existing static/
COPY --from=frontend-build /app/static/dist/ ./app/static/dist/

# Expose port
EXPOSE 8050

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8050"]
