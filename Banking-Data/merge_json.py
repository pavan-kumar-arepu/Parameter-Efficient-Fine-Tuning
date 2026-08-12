import json
import os
from pathlib import Path

# Resolve paths relative to this script so the workflow works from the Virtual Lab terminal.
ROOT_DIR = Path(__file__).resolve().parent
DATASET_FOLDER = ROOT_DIR / "sets"

# Files to merge (in order)
JSON_FILES = [
    "banking_instruction_dataset_batch1_1_to_60.json",
    "banking_instruction_dataset_batch2_61_to_120.json",
    "banking_instruction_dataset_batch3_121_to_180.json",
    "banking_instruction_dataset_batch4_181_to_240.json",
    "banking_instruction_dataset_batch5_241_to_300.json",
    "banking_instruction_dataset_batch6_301_to_360.json",
    "banking_instruction_dataset_batch7_361_to_420.json",
    "banking_instruction_dataset_additional_60.json"
]

# Output file
OUTPUT_FILE = ROOT_DIR / "banking_instruction_dataset_480.json"

combined_data = []

print("=" * 60)
print("Merging Banking Instruction Dataset")
print("=" * 60)

files_processed = 0

for filename in JSON_FILES:

    file_path = DATASET_FOLDER / filename

    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {filename}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:

        data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"{filename} does not contain a JSON array.")

        combined_data.extend(data)

    files_processed += 1
    print(f"[OK] {filename:<45} {len(data):>3} records")

print("-" * 60)
print(f"Files Processed : {files_processed}")
print(f"Total Records   : {len(combined_data)}")
print("-" * 60)

# Save merged dataset
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)

print(f"\nMerged dataset saved as '{OUTPUT_FILE}'")