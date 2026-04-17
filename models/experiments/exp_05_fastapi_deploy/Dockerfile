FROM python:3.11-slim

# prevent Python from buffering stdout/stderr -- important for Cloud Run / Vertex AI logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# install system deps needed by psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# install Python deps first (separate layer -- cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app code
COPY main.py .
COPY recommend_utils.py .

# Vertex AI / Cloud Run injects PORT at runtime -- default to 8080 for local testing
ENV PORT=8080

# use shell form so $PORT is expanded at runtime, not build time
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
