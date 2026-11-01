#!/usr/bin/env bash
# Collect Llama 3.1 8B responses with Ollama on a GPU machine (e.g. the university GPU).
#
# Usage (from anywhere inside the repo, on the GPU machine):
#   bash src/run_llama_gpu.sh setup    # install Ollama in ~/ollama if missing (no sudo), start it, pull the model
#   bash src/run_llama_gpu.sh test     # 5 rows -> data/responses_llama_test.csv, printed
#   bash src/run_llama_gpu.sh full     # all 2140 rows -> data/responses_llama.csv, in the background
#   bash src/run_llama_gpu.sh status   # rows done / 2140, and the end of the log
#   bash src/run_llama_gpu.sh stop     # stop the background run (rerun "full" to resume)
#
# Talks to the local Ollama server only: OLLAMA_URL is forced to localhost and OLLAMA_API_KEY is
# cleared for this script, so the Ollama Cloud settings in .env are not used and nothing is billed.
# collect.py saves after every call and skips saved rows, so "full" can be stopped and rerun safely.
#
# Optional settings (export before running):
#   MODEL=llama3.1:8b           model tag to pull and use
#   OLLAMA_MODELS=/scratch/...  where model weights go (default ~/.ollama/models, about 5 GB)
#   PYTHON=python3
set -euo pipefail

MODEL="${MODEL:-llama3.1:8b}"
PYTHON="${PYTHON:-python3}"
PORT=11434
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export OLLAMA_URL="http://localhost:$PORT"
export OLLAMA_API_KEY=""          # dotenv does not override variables that are already set
export PATH="$HOME/ollama/bin:$PATH"

server_up() { curl -s --max-time 3 "$OLLAMA_URL/api/tags" >/dev/null; }

ensure_ollama() {
    if ! command -v ollama >/dev/null; then
        echo "Ollama not found: installing to ~/ollama (no sudo)"
        mkdir -p "$HOME/ollama"
        curl -fL https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C "$HOME/ollama"
    fi
    echo "ollama: $(command -v ollama) ($(ollama --version 2>/dev/null | tail -1))"
}

ensure_server() {
    if server_up; then echo "Ollama server already running on port $PORT"; return; fi
    echo "starting ollama serve (log: logs/ollama_serve.log)"
    nohup ollama serve > logs/ollama_serve.log 2>&1 &
    for _ in $(seq 1 30); do server_up && { echo "server up"; return; }; sleep 1; done
    echo "server did not start, see logs/ollama_serve.log"; exit 1
}

ensure_python() {
    "$PYTHON" -c "import pandas, requests, dotenv" 2>/dev/null \
        || "$PYTHON" -m pip install --user -q pandas requests python-dotenv
}

collect() {  # extra args are passed to collect.py
    "$PYTHON" -u src/collect.py --provider ollama --model "$MODEL" --rpm 0 "$@"
}

case "${1:-}" in
    setup)
        command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv \
            || echo "warning: nvidia-smi not found, Ollama may run on CPU (much slower)"
        ensure_ollama
        ensure_server
        ollama pull "$MODEL"
        ensure_python
        echo "setup done. Next: bash src/run_llama_gpu.sh test"
        ;;
    test)
        ensure_server
        rm -f data/responses_llama_test.csv logs/usage_llama_test.csv
        collect --out data/responses_llama_test.csv --limit 5 --max-tokens 300 --usage-log logs/usage_llama_test.csv
        "$PYTHON" - <<'EOF'
import pandas as pd
t = pd.read_csv("data/responses_llama_test.csv", dtype=str, keep_default_na=False)
g = pd.read_csv("data/responses_gemini-3.1-flash-lite.csv", dtype=str, nrows=1)
print("columns:", list(t.columns), "| same as Gemini file:", list(t.columns) == list(g.columns))
for r in t.itertuples():
    print(f"\n### {r.prompt_id} | {r.model}\n{r.response[:400]}")
u = pd.read_csv("logs/usage_llama_test.csv")
print(f"\ntokens per call: mean {u.total_tokens.mean():.0f}, max {u.total_tokens.max()}")
EOF
        ;;
    full)
        ensure_server
        if pgrep -f "collect.py --provider ollama --model $MODEL" >/dev/null; then
            echo "a run is already going: bash src/run_llama_gpu.sh status"; exit 0
        fi
        nohup "$PYTHON" -u src/collect.py --provider ollama --model "$MODEL" --rpm 0 \
            --out data/responses_llama.csv --usage-log logs/usage_llama.csv >> logs/collect_llama.log 2>&1 &
        echo "started (pid $!). Progress: bash src/run_llama_gpu.sh status"
        echo "It keeps running after you log out, as long as the machine/job stays up."
        ;;
    status)
        "$PYTHON" -c "import pandas as pd, os; f='data/responses_llama.csv'; \
print((len(pd.read_csv(f)) if os.path.exists(f) else 0), '/ 2140 rows saved')"
        pgrep -f "collect.py --provider ollama" >/dev/null && echo "run: going" || echo "run: not running"
        [ -f logs/collect_llama.log ] && tail -5 logs/collect_llama.log
        ;;
    stop)
        pkill -f "collect.py --provider ollama" && echo "stopped; rerun 'full' to resume" || echo "no run going"
        ;;
    *)
        sed -n '2,19p' "$0"; exit 1
        ;;
esac
