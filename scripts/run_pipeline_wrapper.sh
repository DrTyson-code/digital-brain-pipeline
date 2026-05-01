#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"
LOGS_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOGS_DIR/pipeline-$(date +%Y%m%d_%H%M%S).log"
STDERR_LOG="$LOGS_DIR/pipeline-stderr.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

# Set up log rotation (keep last 7 days of logs)
find "$LOGS_DIR" -name "pipeline-*.log" -type f -mtime +7 -delete 2>/dev/null || true

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE" >&2
    exit 1
fi

# Check that venv exists with Python
if [ ! -f "$PROJECT_DIR/venv/bin/python3" ]; then
    echo "ERROR: Python venv not found at $PROJECT_DIR/venv/bin/python3" >&2
    echo "To fix, run:" >&2
    echo "  python3 -m venv venv" >&2
    echo "  source venv/bin/activate" >&2
    echo "  pip install -r requirements.txt" >&2
    exit 1
fi

# Verify the script is executable
if [ ! -x "$SCRIPT_DIR/run_pipeline_wrapper.sh" ]; then
    chmod +x "$SCRIPT_DIR/run_pipeline_wrapper.sh"
fi

# Load env vars from .env (skip comments and blank lines)
# Using 'source .env' directly instead of process substitution for /bin/sh compatibility
set -o allexport
# shellcheck disable=SC1090
source "$ENV_FILE"
set +o allexport

# Set PYTHONPATH so src module imports work
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Process latest Apple Health export (non-fatal if it fails)
{
    echo "Processing Apple Health export..."
    /opt/homebrew/bin/python3.12 ~/code/brain-tools/process_health_export.py || {
        exit_code=$?
        echo "WARNING: Apple Health export processing failed with exit code $exit_code" >&2
    }
} >> "$LOG_FILE" 2>> "$STDERR_LOG"

# Pull latest fitness bot conversations before running pipeline (non-fatal if it fails)
{
    echo "Pulling fitness bot conversations..."
    "$PROJECT_DIR/venv/bin/python3" \
        "$PROJECT_DIR/scripts/pull_bot_conversations.py" || {
        exit_code=$?
        echo "WARNING: Fitness bot conversation pull failed with exit code $exit_code" >&2
    }
} >> "$LOG_FILE" 2>> "$STDERR_LOG"

# Run main pipeline (cd to project dir so relative cache paths work)
{
    echo "Running main pipeline..."
    cd "$PROJECT_DIR"
    exec "$PROJECT_DIR/venv/bin/python3" \
        "$PROJECT_DIR/scripts/run_pipeline.py" \
        --config "$PROJECT_DIR/config/settings.yaml" \
        --extraction-mode llm_augmented \
        --budget 2.0
} >> "$LOG_FILE" 2>> "$STDERR_LOG"
