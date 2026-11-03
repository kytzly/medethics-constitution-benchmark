#!/usr/bin/env bash
# run_all.sh — run the full pipeline (one process per model/judge to avoid GPU memory not being released)
# Usage:
#   bash run_all.sh smoke     # smoke test: 2 models x first 10 questions (a few minutes, run this first!)
#   bash run_all.sh full      # full run: 10 models x 4 constitutions x 400 questions
set -e

MODELS=(Qwen2.5-32B MiMo-7B-RL Yi-1.5-34B Phi-4 Ministral-3-14B \
        DeepSeek-V2-Lite Llama-3-8B gemma-3-27b GLM-4-32B OLMo-2-32B)

MODE=${1:-smoke}

# The benchmark (data/benchmark_400.json) is shipped with the repo.

if [ "$MODE" = "smoke" ]; then
  echo "==== Smoke test mode ===="
  SM=(Qwen2.5-32B Llama-3-8B)
  echo "-- (1) inference (2 models x first 10 questions) --"
  for m in "${SM[@]}"; do python src/run_inference.py --models "$m" --limit 10; done
  echo "-- (2) judge (2 judges x first 40 answers) --"
  for j in "${SM[@]}"; do python src/run_judge.py --judges "$j" --limit 40; done
else
  echo "==== Full run mode ===="
  echo "-- (1) inference (one process per model) --"
  for m in "${MODELS[@]}"; do python src/run_inference.py --models "$m"; done
  echo "-- (2) judge (one process per judge) --"
  for j in "${MODELS[@]}"; do python src/run_judge.py --judges "$j"; done
fi

echo "==== (3) Aggregate ===="
python src/aggregate.py
echo "==== Done. Results: outputs/scores.json ===="
