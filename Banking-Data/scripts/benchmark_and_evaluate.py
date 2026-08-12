import json
import os
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ==========================================================
# TASK 2 & TASK 5: BENCHMARKING AND EVALUATION
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / "lora_v2"
BENCHMARK_PATH = OUTPUT_DIR / "benchmark_prompts.json"
RESULTS_PATH = OUTPUT_DIR / "benchmark_results.json"
SUMMARY_PATH = OUTPUT_DIR / "evaluation_summary.json"
CSV_PATH = OUTPUT_DIR / "evaluation_summary.csv"

# Use a small base model for the SLM baseline and a larger reference model.
SLM_MODEL_NAME = "google/flan-t5-small"
REFERENCE_MODEL_NAME = "google/flan-t5-base"


def load_benchmark(path):
    """Load the frozen benchmark prompts created earlier."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_prompt(prompt):
    """Format the prompt for instruction-style generation."""
    return f"Answer the following request:\n{prompt}"


def generate_answer(model, tokenizer, prompt, device, max_new_tokens=100):
    """Generate one response for a single prompt."""
    encoded = tokenizer(build_prompt(prompt), return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def score_response(prompt, response):
    """Apply a light heuristic rubric for the four requested evaluation criteria."""
    prompt_lower = prompt.lower()
    response_lower = response.lower()

    # Criterion 1: domain correctness.
    if "france" in prompt_lower:
        domain_score = 2 if any(word in response_lower for word in ["cannot", "not", "don't", "i don't", "help with"]) else 0
    elif any(word in prompt_lower for word in ["account", "loan", "card", "transfer", "transaction", "password", "bank"]):
        domain_score = 2 if any(word in response_lower for word in ["account", "loan", "card", "transfer", "transaction", "password", "bank", "money", "document"]) else 1
    else:
        domain_score = 1

    # Criterion 2: instruction following.
    instruction_score = 0
    if "bullet points" in prompt_lower or "bullet" in prompt_lower:
        instruction_score = 2 if any(token in response_lower for token in ["- ", "* ", "•", "1.", "2.", "3."]) else 0
    elif "starts with yes" in prompt_lower:
        instruction_score = 2 if response_lower.startswith("yes") else 0
    elif "exactly three bullet points" in prompt_lower:
        bullet_count = response_lower.count("- ") + response_lower.count("* ") + response_lower.count("•")
        instruction_score = 2 if bullet_count >= 3 else 1 if bullet_count > 0 else 0
    else:
        instruction_score = 2

    # Criterion 3: completeness.
    word_count = len(response.split())
    completeness_score = 2 if word_count >= 15 else 1 if word_count >= 8 else 0

    # Criterion 4: hallucination control.
    hallucination_score = 2 if any(word in response_lower for word in ["cannot", "not sure", "don't know", "i can't", "unable", "i can help"] ) else 1

    return {
        "domain_correctness": domain_score,
        "instruction_following": instruction_score,
        "completeness": completeness_score,
        "hallucination_control": hallucination_score,
    }


def build_summary(results):
    """Aggregate the benchmark results into a concise per-system summary."""
    summary_rows = []
    for system_name in ["base_slm", "reference_llm", "fine_tuned_slm"]:
        entries = [item for item in results if item["system"] == system_name]
        if not entries:
            continue

        scores = []
        for entry in entries:
            scores.append(entry["scores"])

        averages = {
            "domain_correctness": round(sum(item["domain_correctness"] for item in scores) / len(scores), 2),
            "instruction_following": round(sum(item["instruction_following"] for item in scores) / len(scores), 2),
            "completeness": round(sum(item["completeness"] for item in scores) / len(scores), 2),
            "hallucination_control": round(sum(item["hallucination_control"] for item in scores) / len(scores), 2),
        }
        averages["total"] = round(sum(averages.values()), 2)
        summary_rows.append({"system": system_name, **averages})

    return summary_rows


def run_benchmark():
    """Run the benchmark for the base SLM, reference model, and fine-tuned SLM."""
    print("Loading models for benchmark execution...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the base SLM tokenizer and model.
    slm_tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL_NAME)
    base_slm = AutoModelForSeq2SeqLM.from_pretrained(SLM_MODEL_NAME).to(device)
    base_slm.eval()

    # Load the reference model.
    reference_tokenizer = AutoTokenizer.from_pretrained(REFERENCE_MODEL_NAME)
    reference_model = AutoModelForSeq2SeqLM.from_pretrained(REFERENCE_MODEL_NAME).to(device)
    reference_model.eval()

    # Load the LoRA-adapted fine-tuned SLM from the saved checkpoint.
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(SLM_MODEL_NAME).to(device)
    fine_tuned_model = PeftModel.from_pretrained(fine_tuned_model, str(CHECKPOINT_DIR)).to(device)
    fine_tuned_model.eval()

    benchmark_items = load_benchmark(BENCHMARK_PATH)
    results = []

    # Run the same set of prompts through each system.
    for item in benchmark_items:
        prompt = item["prompt"]
        print(f"Generating response for prompt {item['id']}...")

        base_output = generate_answer(base_slm, slm_tokenizer, prompt, device)
        reference_output = generate_answer(reference_model, reference_tokenizer, prompt, device)
        tuned_output = generate_answer(fine_tuned_model, fine_tuned_tokenizer, prompt, device)

        results.append(
            {
                "id": item["id"],
                "prompt": prompt,
                "type": item["type"],
                "system": "base_slm",
                "response": base_output,
                "scores": score_response(prompt, base_output),
            }
        )
        results.append(
            {
                "id": item["id"],
                "prompt": prompt,
                "type": item["type"],
                "system": "reference_llm",
                "response": reference_output,
                "scores": score_response(prompt, reference_output),
            }
        )
        results.append(
            {
                "id": item["id"],
                "prompt": prompt,
                "type": item["type"],
                "system": "fine_tuned_slm",
                "response": reference_output if False else tuned_output,
                "scores": score_response(prompt, tuned_output),
            }
        )

    # Save the detailed benchmark outputs.
    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    # Save the summarized evaluation metrics.
    summary_rows = build_summary(results)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    pd.DataFrame(summary_rows).to_csv(CSV_PATH, index=False)
    print("Benchmark results written to", RESULTS_PATH)
    print("Evaluation summary written to", SUMMARY_PATH)
    print("Evaluation CSV written to", CSV_PATH)


if __name__ == "__main__":
    run_benchmark()
