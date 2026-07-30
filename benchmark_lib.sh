#!/usr/bin/env bash
# Minimal benchmark_lib.sh stub

check_env_vars() {
    local missing=0
    for var in "$@"; do
        if [ -z "${!var}" ]; then
            echo "ERROR: Required env var $var is not set"
            missing=1
        fi
    done
    [ $missing -ne 0 ] && exit 1
    echo "All required env vars present."
}

wait_for_server_ready() {
    local port=""
    local server_log=""
    local server_pid=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port)        port="$2";       shift 2 ;;
            --server-log)  server_log="$2"; shift 2 ;;
            --server-pid)  server_pid="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    echo "Waiting for vLLM server on port $port ..."
    local max_wait=600  # 10 minutes
    local elapsed=0
    while true; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "ERROR: Server process $server_pid died. Last log lines:"
            tail -30 "$server_log"
            exit 1
        fi
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "Server is ready on port $port."
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: Server did not become ready within ${max_wait}s. Last log lines:"
            tail -30 "$server_log"
            exit 1
        fi
        echo "  still waiting... (${elapsed}s)"
    done
}

run_benchmark_serving() {
    local model="" port="" backend="" input_len="" output_len=""
    local random_range_ratio="" num_prompts="" max_concurrency=""
    local result_filename="" result_dir="."

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)               model="$2";              shift 2 ;;
            --port)                port="$2";               shift 2 ;;
            --backend)             backend="$2";            shift 2 ;;
            --input-len)           input_len="$2";          shift 2 ;;
            --output-len)          output_len="$2";         shift 2 ;;
            --random-range-ratio)  random_range_ratio="$2"; shift 2 ;;
            --num-prompts)         num_prompts="$2";        shift 2 ;;
            --max-concurrency)     max_concurrency="$2";    shift 2 ;;
            --result-filename)     result_filename="$2";    shift 2 ;;
            --result-dir)          result_dir="$2";         shift 2 ;;
            *) shift ;;
        esac
    done

    mkdir -p "$result_dir"

    /opt/venv/bin/python -m vllm.entrypoints.cli.main bench serve \
        --model "$model" \
        --host localhost \
        --port "$port" \
        --backend "$backend" \
        --dataset-name random \
        --input-len "$input_len" \
        --output-len "$output_len" \
        --random-range-ratio "$random_range_ratio" \
        --num-prompts "$num_prompts" \
        --max-concurrency "$max_concurrency" \
        --save-result \
        --result-dir "$result_dir" \
        --result-filename "${result_filename}.json"
}
