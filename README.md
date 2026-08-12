# Parameter-Efficient Fine-Tuning

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

## Architecture & Purpose

### Purpose
- Demonstrate parameter-efficient fine-tuning for instruction-following banking data using LoRA adapters.
- Keep compute and storage costs low by training small adapter weights instead of full model weights.
- Provide reproducible scripts, checkpoints, and evaluation for interview discussion and demos.

### Architecture (high-level)
This project follows a straightforward data-to-deployment pipeline:

- Data ingestion and cleaning -> dataset splits -> training with LoRA adapters on a base LM -> checkpointing -> evaluation and reporting.

See `docs/architecture.mmd` for a simple diagram you can show in interviews.

### Interview talking points
- Explain why LoRA reduces trainable parameters and speeds up iteration.
- Describe dataset preparation choices and evaluation metrics used in `Banking-Data/scripts`.
- Point to `Banking-Data/output/checkpoints` to show real adapter checkpoints and explain how they attach to the base model at inference.


