# Extends the official OpenClaw image to bake in Python dependencies
# for the agentic ML pipeline (DatasetAdapter, featurizer, etc).
#
# Rebuild whenever requirements.txt changes:
#   docker compose build
#   docker compose up -d openclaw-gateway
ARG OPENCLAW_BASE_IMAGE=ghcr.io/openclaw/openclaw:latest
FROM ${OPENCLAW_BASE_IMAGE}

USER root

# Base image (node:*-bookworm-slim) ships no Python at all.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python deps. --break-system-packages is required on
# Debian's PEP 668-protected system Python; safe here since this is a
# throwaway build layer, not a host install. Keep requirements.txt pinned.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

USER node