# Tools container for the predictive-maintenance MAS MVP.
# This image holds the DETERMINISTIC tools the orchestrator execs against
# (inspect/featurize/classify/recommend). The LLM itself runs separately via
# your vLLM endpoint; OpenClaw connects to that and shells into this container.
FROM python:3.12-slim

# System deps kept minimal; sklearn wheels are manylinux so no build toolchain needed.
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY aviation_mas_mvp /app/aviation_mas_mvp

# Editable install so `python -m aviation_mas_mvp.tools ...` works from any cwd.
RUN pip install -e .

# Mount your data and an output dir at runtime:
#   docker run --rm -it \
#     -v $PWD/data:/data:ro \
#     -v $PWD/run:/run \
#     aviation-mas-tools \
#     python -m aviation_mas_mvp.tools inspect --flight-dir /data/flights --metadata /data/metadata.csv
#
# The agent issues exactly these calls (see openclaw/tools_manifest.md):
#   inspect   --flight-dir DIR --metadata META.csv
#   featurize --flight-dir DIR --metadata META.csv --out /run/feats.csv
#   classify  --feats /run/feats.csv --model /run/model.joblib --out /run/preds.json
#   recommend --preds /run/preds.json --queue /run/maintenance_queue.jsonl --top-k 10
WORKDIR /run
CMD ["python", "-c", "import aviation_mas_mvp; print('aviation-mas-tools ready')"]
