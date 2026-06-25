# Aviation predictive-maintenance MVP — OpenClaw gateway image
#
# We do NOT build OpenClaw from source (the pnpm/bun build is heavy and
# OOM-prone). We start from the official pre-built image and add only what
# the predictive-maintenance tools need: a Python runtime. The four
# deterministic tools (`python -m aviation_mas_mvp.tools <cmd>`) are invoked
# by the agent via the gateway's `exec` tool, so Python must exist in the
# same image the gateway runs in.
#
# Base image tags: `latest`, `main`, or a pinned version e.g. `2026.2.26`.
# Pin a version once the smoke test passes, so rebuilds are reproducible.
ARG OPENCLAW_IMAGE=ghcr.io/openclaw/openclaw:latest
FROM ${OPENCLAW_IMAGE}

# apt + pip need root; the base image otherwise runs as the non-root `node`
# user (uid 1000). We drop back to `node` at the end.
USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      curl \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Python deps for the tools live here. Kept tiny for the smoke test; the real
# featurizer/classifier deps (numpy/pandas/scikit-learn/joblib) get added when
# we wire the actual tools in the next step. --break-system-packages because
# Debian marks the system Python as externally-managed.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# Drop back to the unprivileged runtime user the gateway expects.
USER node

# Inherit the base image's ENTRYPOINT (tini) and CMD (starts the gateway with
# --allow-unconfigured). We intentionally do not override them.
