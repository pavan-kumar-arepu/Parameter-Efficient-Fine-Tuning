import json
import os
import sys
from pathlib import Path

# ----------------------------------------------------------
# TASK 1/2/5 SUPPORT SCRIPT
# This script prepares the common assignment artifacts so the
# workflow can be run and documented from a single entry point.
# ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'output'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print('Preparing assignment artifacts...')

# Create the frozen benchmark prompts used for Task 2 and Task 5.
benchmark = [
    {
        'id': 1,
        'prompt': 'How do I open a savings account in this bank?',
        'type': 'in_domain_factual'
    },
    {
        'id': 2,
        'prompt': 'List the required documents to apply for a credit card.',
        'type': 'in_domain_factual'
    },
    {
        'id': 3,
        'prompt': 'I need help with a disputed transaction. Explain the steps in bullet points.',
        'type': 'formatting_sensitive'
    },
    {
        'id': 4,
        'prompt': 'How can I transfer money internationally and what fees should I expect?',
        'type': 'reasoning_style'
    },
    {
        'id': 5,
        'prompt': 'What is the capital of France?',
        'type': 'out_of_scope'
    },
    {
        'id': 6,
        'prompt': 'Explain how to reset my online banking password safely.',
        'type': 'in_domain_factual'
    },
    {
        'id': 7,
        'prompt': 'Give me a short response that starts with Yes and includes exactly three bullet points.',
        'type': 'formatting_sensitive'
    },
    {
        'id': 8,
        'prompt': 'Why might a loan application be declined?',
        'type': 'reasoning_style'
    }
]

with open(OUTPUT_DIR / 'benchmark_prompts.json', 'w', encoding='utf-8') as f:
    json.dump(benchmark, f, indent=2)

# Create a compact summary report
summary = {
    'assignment': 'Conversational AI Assignment 2',
    'domain': 'banking support',
    'slm': 'google/flan-t5-small',
    'reference_llm': 'gpt-4o-mini or comparable hosted model',
    'dataset_source': 'local banking instruction JSON files',
    'dataset_size': 347,
    'train_split': 277,
    'validation_split': 35,
    'test_split': 35,
    'benchmark_prompts': len(benchmark),
    'checkpoint_dir': str(CHECKPOINT_DIR),
    'notes': [
        'The workflow uses LoRA-based parameter-efficient fine-tuning.',
        'The benchmark prompts are frozen and reused across tasks.',
        'The scripts generate reports and outputs that can be used directly in the submission.'
    ]
}

with open(OUTPUT_DIR / 'assignment_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print('Created benchmark prompts and summary artifacts.')
print('Outputs written to:', OUTPUT_DIR)
