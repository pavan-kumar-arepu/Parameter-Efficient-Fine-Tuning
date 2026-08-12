#!/bin/bash
set -e

cd "$(dirname "$0")"

python3 merge_json.py
python3 task1_pipeline.py
python3 scripts/run_assignment.py
python3 scripts/train_lora.py
python3 scripts/finalize_assignment.py
python3 scripts/build_submission_report.py

echo "All assignment steps completed."
