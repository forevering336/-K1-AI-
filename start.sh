#!/bin/bash
set -e
cd "$(dirname "$0")"

MODEL="models/best_award_int8.onnx"
if [ ! -f "$MODEL" ]; then
  echo "[ERROR] Missing $MODEL"
  exit 1
fi

python3 weld_live.py --web-video --no-display --model "$MODEL"
