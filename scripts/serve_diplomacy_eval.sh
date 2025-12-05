#!/bin/bash
# Launch vLLM servers for Diplomacy SFT evaluation
#
# Usage:
#   ./scripts/serve_diplomacy_eval.sh <lora_adapter_path>
#
# Example:
#   ./scripts/serve_diplomacy_eval.sh results/diplomacy/sft/diplomacy_sft_train-20251204-233336-c1048be/artifacts
#
# This starts two vLLM servers:
#   - Port 8001: Base Qwen model (DEFAULT agents)
#   - Port 8002: LoRA fine-tuned model (TARGET agent)
#
# Then run eval with:
#   DEFAULT_BASE_URL=http://localhost:8001/v1 \
#   TARGET_BASE_URL=http://localhost:8002/v1 \
#   DEFAULT_MODEL=Qwen/Qwen3-8B \
#   TARGET_MODEL=Qwen/Qwen3-8B \
#   python gamescope/environments/diplomacy/scripts/eval_diplobench.py --num_games 10

set -e

LORA_PATH="${1:-}"
BASE_MODEL="${2:-Qwen/Qwen3-4b}"

if [ -z "$LORA_PATH" ]; then
    echo "Usage: $0 <lora_adapter_path> [base_model]"
    echo "Example: $0 results/diplomacy/sft/diplomacy_sft_train-20251205-021831-c1048be/artifacts"
    exit 1
fi

if [ ! -d "$LORA_PATH" ]; then
    echo "Error: LoRA adapter path does not exist: $LORA_PATH"
    exit 1
fi

echo "Starting vLLM servers..."
echo "  Base model: $BASE_MODEL"
echo "  LoRA adapter: $LORA_PATH"
echo ""

# Activate venv if not already
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

# Start base model on GPU 0, port 8001
echo "Starting base model server on port 8001 (GPU 0)..."
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --port 8001 \
    --data-parallel-size 2 \
    --max-model-len 20000 \
    --trust-remote-code \
    --max-num-seqs 16 \
    &> logs/vllm_base.log &
BASE_PID=$!
echo "  PID: $BASE_PID"

# Start LoRA model on GPU 1, port 8002
echo "Starting LoRA model server on port 8002 (GPU 1)..."
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --port 8002 \
    --tensor-parallel-size 2 \
    --max-model-len 20000 \
    --trust-remote-code \
    --enable-lora \
    --max-lora-rank 32 \
    --lora-modules "diplomacy-sft=$LORA_PATH" \
    --max-num-seqs 16 \
    &> logs/vllm_lora.log &
LORA_PID=$!
echo "  PID: $LORA_PID"

echo ""
echo "Servers starting in background. Check logs:"
echo "  tail -f logs/vllm_base.log"
echo "  tail -f logs/vllm_lora.log"
echo ""
echo "Wait for servers to be ready, then run eval:"
echo ""
echo "  DEFAULT_BASE_URL=http://localhost:8001/v1 \\"
echo "  TARGET_BASE_URL=http://localhost:8002/v1 \\"
echo "  DEFAULT_MODEL=$BASE_MODEL \\"
echo "  TARGET_MODEL=diplomacy-sft \\"
echo "  python gamescope/environments/diplomacy/scripts/eval_diplobench.py --num_games 10"
echo ""
echo "To stop servers:"
echo "  kill $BASE_PID $LORA_PID"
