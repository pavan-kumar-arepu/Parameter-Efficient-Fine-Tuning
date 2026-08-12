# Parameter-Efficient Fine-Tuning (Assignment)

This repository contains code and data for the assignment on Parameter-Efficient Fine-Tuning.

Contents
- `Banking-Data/` — dataset, scripts, outputs, and training checkpoints.

Quick start
1. Create a Python environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Banking-Data/requirements.txt
```

2. Inspect data and run training (example):

```bash
# run a training script (adjust args as needed)
python Banking-Data/scripts/train_lora.py
```

Notes
- This repo includes pre-prepared LoRA checkpoints under `Banking-Data/output/checkpoints/`.
- Update `Banking-Data/README.md` or the assignment scripts for dataset-specific instructions.

License
- See `SUBMISSION_CHECKLIST.txt` for assignment submission details.
