FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pinned requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and seed assets
COPY . .

EXPOSE 5000

ENV PORT=5000
ENV FALKORDB_HOST=localhost
ENV FALKORDB_PORT=6379
ENV FALKORDB_GRAPH=polypharmacy_guardian

CMD ["sh", "-c", "python -m app.ingestion && python -m app.server"]
