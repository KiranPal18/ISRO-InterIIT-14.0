#!/bin/bash

echo "=============================================="
echo "  Satellite VQA Container Starting..."
echo "=============================================="
echo ""

# Start SSH server
echo "Starting OpenSSH server..."
/usr/sbin/sshd
if [ $? -eq 0 ]; then
    echo "✓ SSH server started on port 22"
else
    echo "✗ SSH server failed to start"
    service ssh start  # Fallback
fi

echo ""
echo "SSH is enabled. Use RunPod's SSH connection string."
echo ""

# =============================================================================
# Auto-start mode (set AUTO_START=1 to enable)
# =============================================================================
if [ "${AUTO_START:-0}" = "1" ]; then
    echo "Auto-start mode enabled..."
    
    # Check if models exist
    if [ ! -d "/workspace/models/hf_cache" ]; then
        echo "Creating vLLM cache directory..."
        mkdir -p /workspace/models/hf_cache
    fi
    
    echo ""
    echo "=== Starting vLLM Server on GPU 1 ==="
    
    # Start vLLM server in background on GPU 1
    # Uses HuggingFace model ID with local cache directory
    CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --host 0.0.0.0 \
        --port 8001 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.50 \
        --max-model-len 16384 \
        --dtype bfloat16 \
        --download-dir /workspace/models/hf_cache \
        --trust-remote-code \
        --disable-log-requests &
    
    VLLM_PID=$!
    echo "vLLM server starting (PID: $VLLM_PID)..."
    
    # Wait for vLLM server to be ready
    echo "Waiting for vLLM server to be ready..."
    MAX_WAIT=300
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        if curl -s http://localhost:8001/health > /dev/null 2>&1; then
            echo "✓ vLLM server is ready!"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
        echo "  Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
    done
    
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo "✗ vLLM server failed to start within $MAX_WAIT seconds"
        echo "  Check logs and try again"
        tail -f /dev/null
        exit 1
    fi
    
    echo ""
    echo "=== Starting FastAPI Server on GPU 0 ==="
    
    # Start FastAPI server on GPU 0
    CUDA_VISIBLE_DEVICES=0 python -m uvicorn api:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 &
    
    API_PID=$!
    echo "FastAPI server starting (PID: $API_PID)..."
    
    # Wait a moment for API to start
    sleep 10
    
    echo ""
    echo "=============================================="
    echo "  Services Started Successfully!"
    echo "=============================================="
    echo ""
    echo "  vLLM Server:  http://localhost:8001 (GPU 1)"
    echo "  FastAPI:      http://localhost:8000 (GPU 0)"
    echo ""
    echo "  API Docs:     http://localhost:8000/docs"
    echo "  Health:       http://localhost:8000/health"
    echo ""
    echo "=============================================="
    
    # Keep container alive and monitor processes
    wait $API_PID $VLLM_PID
    
else
    # Manual mode - just print instructions
    echo "Manual start mode (default)."
    echo ""
    echo "To start services manually:"
    echo ""
    echo "  1. Start vLLM server (GPU 1):"
    echo "     CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \\"
    echo "       --model Qwen/Qwen2.5-VL-7B-Instruct \\"
    echo "       --port 8001 --gpu-memory-utilization 0.50 --max-model-len 16384 \\"
    echo "       --download-dir /workspace/models/hf_cache --trust-remote-code &"
    echo ""
    echo "  2. Start FastAPI (GPU 0):"
    echo "     CUDA_VISIBLE_DEVICES=0 python -m uvicorn api:app --host 0.0.0.0 --port 8000"
    echo ""
    echo "  Or set AUTO_START=1 to auto-start both services."
    echo ""
    echo "=============================================="
    
    # Keep container alive
    tail -f /dev/null
fi
