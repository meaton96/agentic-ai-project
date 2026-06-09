FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y nodejs

RUN npm install -g openclaw@latest

WORKDIR /app

# Default: run gateway on port 18789
CMD ["openclaw", "gateway", "--port", "18789", "--verbose", "--allow-unconfigured"]