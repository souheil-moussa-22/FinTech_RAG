#!/bin/sh

set -e

EMBED_MODEL="nomic-embed-text"
CHAT_MODEL="${OLLAMA_CHAT_MODEL:-gemma2:2b}"

echo "🚀 Starting Ollama server..."
ollama serve &
SERVE_PID=$!

MAX_WAIT=120
WAITED=0
echo "⏳ Waiting for Ollama API to be ready..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "❌ Ollama did not start within ${MAX_WAIT}s"
        exit 1
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done
echo "✅ Ollama API is ready (waited ${WAITED}s)"

pull_if_missing() {
    MODEL_NAME="$1"
    echo ""
    echo "📦 Checking model: ${MODEL_NAME}"
    if ollama show "${MODEL_NAME}" > /dev/null 2>&1; then
        echo "   ✓ Already present — skipping pull"
    else
        echo "   ↓ Pulling ${MODEL_NAME} (may take several minutes on first run)..."
        ollama pull "${MODEL_NAME}"
        echo "   ✓ ${MODEL_NAME} ready"
    fi
}

pull_if_missing "${EMBED_MODEL}"

pull_if_missing "${CHAT_MODEL}"

echo ""
echo "═══════════════════════════════════════"
echo "  Ollama ready — loaded models:"
ollama list
echo "═══════════════════════════════════════"
echo ""

wait $SERVE_PID