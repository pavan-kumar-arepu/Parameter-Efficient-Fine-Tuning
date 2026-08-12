# Conversational AI Assignment 2 - Virtual Lab Guide

This folder is prepared to run the full assignment workflow in a college Virtual Lab environment from start to finish.

## 1. What you will produce
By the end of this workflow, you should have:
- a cleaned banking instruction dataset
- train/validation/test splits
- a frozen benchmark of 8 prompts
- a LoRA fine-tuned small model checkpoint
- evaluation and summary artifacts
- a PDF report for submission

## 2. Folder structure
- [Banking-Data](.) - main assignment folder
- [Banking-Data/sets](sets) - original JSON batches for the banking dataset
- [Banking-Data/output](output) - generated datasets, summaries, and checkpoints
- [Banking-Data/scripts](scripts) - Python scripts for each task
- [Banking-Data/submission](submission) - final report artifact
- [Banking-Data/requirements.txt](requirements.txt) - Python dependencies

## 3. Virtual Lab setup
Use the terminal inside the Virtual Lab and work from the assignment folder.

### Step A: Open the terminal
Go to the project folder:

```bash
cd /home/<your-user>/ConAI/Banking-Data
```

If your folder path is different, replace it accordingly.

### Step B: Create a Python environment
Use a virtual environment so the installation stays clean:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step C: Install dependencies
Install everything required for dataset preparation, training, and report generation:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the workflow in order
Run the following commands one by one.

### Step 1: Merge the dataset chunks
This creates the unified 480-example banking dataset:

```bash
python3 merge_json.py
```

### Step 2: Prepare and clean the dataset
This creates the cleaned dataset and the train/validation/test splits:

```bash
python3 task1_pipeline.py
```

### Step 3: Create the benchmark and assignment artifacts
This prepares the frozen benchmark prompts and summary JSON files:

```bash
python3 scripts/run_assignment.py
```

### Step 4: Fine-tune the small model using LoRA
This is the main training step for the assignment:

```bash
python3 scripts/train_lora.py
```

### Step 5: Generate evaluation results
This creates the benchmark outputs and evaluation summary:

```bash
python3 scripts/finalize_assignment.py
```

### Step 6: Build the submission PDF report
This generates the PDF report for submission:

```bash
python3 scripts/build_submission_report.py
```

## 5. Expected artifacts after execution
After all steps finish, these files should exist:
- [Banking-Data/output/cleaned_dataset.json](output/cleaned_dataset.json)
- [Banking-Data/output/train.json](output/train.json)
- [Banking-Data/output/validation.json](output/validation.json)
- [Banking-Data/output/test.json](output/test.json)
- [Banking-Data/output/dataset_statistics.csv](output/dataset_statistics.csv)
- [Banking-Data/output/cleaning_report.csv](output/cleaning_report.csv)
- [Banking-Data/output/benchmark_prompts.json](output/benchmark_prompts.json)
- [Banking-Data/output/benchmark_results.json](output/benchmark_results.json)
- [Banking-Data/output/evaluation_summary.json](output/evaluation_summary.json)
- [Banking-Data/output/checkpoints/lora_v1](output/checkpoints/lora_v1)
- [Banking-Data/submission/assignment_report.pdf](submission/assignment_report.pdf)

## 6. What to submit
For the assignment, submit the following:
1. The PDF report from [Banking-Data/submission/assignment_report.pdf](submission/assignment_report.pdf)
2. The code scripts from [Banking-Data/scripts](scripts)
3. The generated datasets and summaries from [Banking-Data/output](output)
4. The fine-tuned adapter/checkpoint from [Banking-Data/output/checkpoints/lora_v1](output/checkpoints/lora_v1)
5. Screenshots from the Virtual Lab terminal showing the execution steps

## 7. Screenshots to capture in Virtual Lab
Take full-screen screenshots at these moments:
- terminal showing the virtual environment activation
- terminal showing dependency installation
- terminal showing dataset preparation
- terminal showing LoRA fine-tuning start and completion
- terminal showing report generation

## 8. Quick one-shot run
If you want to run everything in one go, use:

```bash
bash run_all.sh
```

## 9. Notes
- The scripts were prepared to work in the Virtual Lab environment.
- If the training takes time, let it run without interruption.
- If you see a memory issue, reduce batch size in the training script and rerun.
