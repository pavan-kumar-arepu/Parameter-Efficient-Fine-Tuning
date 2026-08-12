import json
import os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

# ----------------------------------------------------------
# TASK 3: PARAMETER-EFFICIENT FINE-TUNING (LoRA)
# This script loads the banking instruction data, prepares the
# prompt-response pairs, and fine-tunes a small model using LoRA.
# ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'output'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints' / 'lora_v1'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'google/flan-t5-small'
TRAIN_PATH = OUTPUT_DIR / 'train.json'
VALID_PATH = OUTPUT_DIR / 'validation.json'

print('Loading tokenizer and model...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['q', 'v'],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.SEQ_2_SEQ_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def load_records(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        response = item.get('response', '')
        prompt = instruction if not input_text else f"{instruction}\n\n{input_text}"
        rows.append({'prompt': prompt, 'response': response})
    return rows

train_rows = load_records(TRAIN_PATH)
valid_rows = load_records(VALID_PATH)

train_dataset = Dataset.from_list(train_rows)
valid_dataset = Dataset.from_list(valid_rows)


def preprocess(batch):
    inputs = tokenizer(batch['prompt'], truncation=True, max_length=128, padding='max_length')
    outputs = tokenizer(batch['response'], truncation=True, max_length=128, padding='max_length')
    batch['input_ids'] = inputs['input_ids']
    batch['attention_mask'] = inputs['attention_mask']
    batch['labels'] = outputs['input_ids']
    return batch

train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=['prompt', 'response'])
valid_dataset = valid_dataset.map(preprocess, batched=True, remove_columns=['prompt', 'response'])

training_args = Seq2SeqTrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    evaluation_strategy='epoch',
    predict_with_generate=True,
    fp16=False,
    report_to='none',
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

print('Starting training...')
trainer.train()
trainer.save_model(str(CHECKPOINT_DIR))
tokenizer.save_pretrained(str(CHECKPOINT_DIR))
print('Saved checkpoint to', CHECKPOINT_DIR)
