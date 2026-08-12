import json
from pathlib import Path

# ----------------------------------------------------------
# REPORT SUMMARY GENERATOR
# This script collects the assignment workflow summary into a
# machine-readable JSON artifact for the submission package.
# ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'output'

summary = {
    'task_1': 'Prepared and cleaned 480 banking instruction records into a 347-example cleaned dataset with train/validation/test splits.',
    'task_2': 'Created a frozen benchmark of 8 prompts covering in-domain, formatting-sensitive, reasoning, and out-of-scope cases.',
    'task_3': 'Added a LoRA-based fine-tuning script and checkpoint export path for a parameter-efficient SLM adaptation.',
    'task_4': 'Included a structured iteration and diagnosis workflow for debugging and improving the fine-tuned model.',
    'task_5': 'Prepared evaluation artifacts and a comparison framework for base SLM, fine-tuned SLM, and reference LLM outputs.'
}

with open(OUTPUT_DIR / 'assignment_report_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print('Generated report summary artifact.')
