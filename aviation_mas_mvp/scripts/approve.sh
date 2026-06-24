export MAINTENANCE_APPROVAL_TOKEN="potato"


python -m scripts.tools approve --queue work/agent_run/maintenance_queue.jsonl --approved approved_out/approved_queue.jsonl --token "potato"