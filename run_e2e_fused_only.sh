#!/usr/bin/env bash
# Fused GEMM+AR e2e benchmark — NO profiler.
set -euo pipefail

export MODEL="openai/gpt-oss-120b"
export TP=8
export CONC=32
export ISL=2048
export OSL=512
export MAX_MODEL_LEN=8192
export RANDOM_RANGE_RATIO=0.8
export NUM_PROMPTS=320
export PORT=8089

export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1
export HSA_NO_SCRATCH_RECLAIM=1

source "$(dirname "$0")/benchmark_lib.sh"

COMPILE_SIZES='[1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,256,512,1024,2048,8192]'
CUDAGRAPH_SIZES='[1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,264,272,280,288,296,304,312,320,328,336,344,352,360,368,376,384,392,400,408,416,424,432,440,448,456,464,472,480,488,496,504,512,520,528,536,544,552,560,568,576,584,592,600,608,616,624,632,640,648,656,664,672,680,688,696,704,712,720,728,736,744,752,760,768,776,784,792,800,808,816,824,832,840,848,856,864,872,880,888,896,904,912,920,928,936,944,952,960,968,976,984,992,1000,1008,1016,1024,2048,4096,8192]'

CONFIG_JSON="{\"pass_config\": {\"fuse_gemm_all_reduce\": true}, \"compile_sizes\":$COMPILE_SIZES, \"cudagraph_capture_sizes\":$CUDAGRAPH_SIZES, \"cudagraph_mode\": \"FULL_AND_PIECEWISE\"}"

echo ""
echo "========================================================================"
echo "  FUSED GEMM+AR (no profiler)"
echo "========================================================================"

CONFIG_FILE="/tmp/config_fused_noprof.yaml"
cat > "$CONFIG_FILE" << EOFCFG
compilation-config: '$CONFIG_JSON'
EOFCFG
cat "$CONFIG_FILE"

SERVER_LOG=$(mktemp /tmp/server-fused-noprof-XXXXXX.log)
echo "Server log: $SERVER_LOG"

/opt/venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size="$TP" \
    --gpu-memory-utilization 0.95 \
    --max-model-len "$MAX_MODEL_LEN" \
    --config "$CONFIG_FILE" \
    --block-size=64 \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --async-scheduling > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

echo ""
echo "--- Checking fused kernel pattern matches ---"
match_count=$(grep -c "GEMMAllReducePass replaced [1-9]" "$SERVER_LOG" || true)
echo "  Fused GEMM+AR patterns matched in $match_count log lines"
grep "GEMMAllReducePass replaced" "$SERVER_LOG" | head -8 || true
echo ""

echo "--- Running benchmark: $NUM_PROMPTS prompts, concurrency=$CONC ---"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --result-filename "e2e_fused_ar" \
    --result-dir /tmp/ || true

echo ""
echo "Shutting down server (PID: $SERVER_PID)..."
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
echo "Server shutdown complete."
