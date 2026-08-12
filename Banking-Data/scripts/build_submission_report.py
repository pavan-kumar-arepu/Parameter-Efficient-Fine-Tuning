import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ----------------------------------------------------------
# SUBMISSION REPORT BUILDER
# This script turns the assignment artifacts into a PDF report
# that can be attached to the project submission.
# ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'output'
SUBMISSION_DIR = ROOT / 'submission'
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

summary = {
    'assignment': 'Conversational AI Assignment 2 - Parameter-Efficient Fine-Tuning',
    'domain': 'Banking support',
    'dataset': 'Merged banking instruction dataset with 480 samples, cleaned to 347',
    'split': 'Train 277 / Validation 35 / Test 35',
    'benchmark': '8 frozen prompts covering factual, formatting, reasoning, and out-of-scope cases',
    'technique': 'LoRA-based parameter-efficient fine-tuning with google/flan-t5-small',
    'artifacts': [
        'output/cleaned_dataset.json',
        'output/train.json',
        'output/validation.json',
        'output/test.json',
        'output/benchmark_prompts.json',
        'output/benchmark_results.json',
        'output/assignment_summary.json',
        'output/assignment_report_summary.json'
    ]
}

text = []
text.append('Conversational AI Assignment 2 Report')
text.append('')
text.append(f"Assignment: {summary['assignment']}")
text.append(f"Domain: {summary['domain']}")
text.append(f"Dataset: {summary['dataset']}")
text.append(f"Split: {summary['split']}")
text.append(f"Benchmark: {summary['benchmark']}")
text.append(f"Technique: {summary['technique']}")
text.append('')
text.append('Artifacts generated:')
for item in summary['artifacts']:
    text.append(f'- {item}')
text.append('')
text.append('Workflow summary:')
text.append('1. Merge the banking instruction JSON files into a single flat dataset.')
text.append('2. Clean duplicates and empty/truncated responses.')
text.append('3. Create train/validation/test splits.')
text.append('4. Freeze an 8-prompt benchmark for evaluation.')
text.append('5. Prepare a LoRA-based training and evaluation pipeline for the SLM.')
text.append('')
text.append('This report was generated from the scripts in the Banking-Data folder.')

pdf_path = SUBMISSION_DIR / 'assignment_report.pdf'
canvas_obj = canvas.Canvas(str(pdf_path), pagesize=letter)
text_obj = canvas_obj.beginText(40, 760)
text_obj.setFont('Helvetica', 11)
for line in text:
    text_obj.textLine(line)
canvas_obj.drawText(text_obj)
canvas_obj.save()

print('Generated report PDF at', pdf_path)
