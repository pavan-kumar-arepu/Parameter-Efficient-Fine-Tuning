import json
import re
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent
INPUT_FILE = ROOT_DIR / "banking_instruction_dataset_480.json"

OUTPUT_DIR = ROOT_DIR / "output"

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading dataset...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

before_total = len(data)

df = pd.DataFrame(data)

# ==========================================================
# VALIDATE SCHEMA
# ==========================================================

required_columns = ["instruction", "input", "response"]

missing_columns = []

for col in required_columns:
    if col not in df.columns:
        missing_columns.append(col)

if missing_columns:
    raise Exception(f"Missing columns: {missing_columns}")

# ==========================================================
# NORMALIZE TEXT
# ==========================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.replace("\t", " ")

    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


for col in required_columns:
    df[col] = df[col].apply(clean_text)

# ==========================================================
# EMPTY FIELD CHECK
# ==========================================================

empty_instruction = (df["instruction"] == "").sum()

empty_response = (df["response"] == "").sum()

empty_input = (df["input"] == "").sum()

# ==========================================================
# WORD COUNT
# ==========================================================

df["response_word_count"] = df["response"].apply(
    lambda x: len(x.split())
)

short_response = (df["response_word_count"] < 30).sum()

long_response = (df["response_word_count"] > 120).sum()

# ==========================================================
# DUPLICATE CHECK
# ==========================================================

duplicate_instruction = df.duplicated(
    subset=["instruction"]
).sum()

duplicate_pair = df.duplicated(
    subset=["instruction", "response"]
).sum()

df = df.drop_duplicates(
    subset=["instruction", "response"]
).reset_index(drop=True)

after_total = len(df)

# ==========================================================
# DATASET STATISTICS
# ==========================================================

statistics = pd.DataFrame({

    "Metric":[

        "Total Samples Before",

        "Total Samples After",

        "Duplicate Instructions",

        "Duplicate Records",

        "Empty Instructions",

        "Empty Inputs",

        "Empty Responses",

        "Responses <30 words",

        "Responses >120 words",

        "Average Instruction Words",

        "Average Response Words"

    ],

    "Value":[

        before_total,

        after_total,

        duplicate_instruction,

        duplicate_pair,

        empty_instruction,

        empty_input,

        empty_response,

        short_response,

        long_response,

        round(df["instruction"].apply(lambda x: len(x.split())).mean(),2),

        round(df["response_word_count"].mean(),2)

    ]

})

statistics.to_csv(

    os.path.join(OUTPUT_DIR,"dataset_statistics.csv"),

    index=False

)

# ==========================================================
# CLEANING REPORT
# ==========================================================

report = pd.DataFrame({

    "Issue":[

        "Total Samples",

        "Duplicate Instructions",

        "Duplicate Records",

        "Empty Instructions",

        "Empty Inputs",

        "Empty Responses",

        "Short Responses",

        "Long Responses"

    ],

    "Before":[

        before_total,

        duplicate_instruction,

        duplicate_pair,

        empty_instruction,

        empty_input,

        empty_response,

        short_response,

        long_response

    ],

    "After":[

        after_total,

        0,

        0,

        0,

        empty_input,

        0,

        short_response,

        long_response

    ]

})

report.to_csv(

    os.path.join(OUTPUT_DIR,"cleaning_report.csv"),

    index=False

)

# ==========================================================
# REMOVE AUXILIARY COLUMN
# ==========================================================

df = df.drop(columns=["response_word_count"])

# ==========================================================
# SAVE CLEANED DATASET
# ==========================================================

cleaned_file = os.path.join(

    OUTPUT_DIR,

    "cleaned_dataset.json"

)

df.to_json(

    cleaned_file,

    orient="records",

    indent=4,

    force_ascii=False

)

# ==========================================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

train_df, temp_df = train_test_split(

    df,

    test_size=(1-TRAIN_RATIO),

    random_state=RANDOM_STATE,

    shuffle=True

)

validation_df, test_df = train_test_split(

    temp_df,

    test_size=0.5,

    random_state=RANDOM_STATE,

    shuffle=True

)

train_df.to_json(

    os.path.join(OUTPUT_DIR,"train.json"),

    orient="records",

    indent=4,

    force_ascii=False

)

validation_df.to_json(

    os.path.join(OUTPUT_DIR,"validation.json"),

    orient="records",

    indent=4,

    force_ascii=False

)

test_df.to_json(

    os.path.join(OUTPUT_DIR,"test.json"),

    orient="records",

    indent=4,

    force_ascii=False

)

# ==========================================================
# SUMMARY
# ==========================================================

summary = f"""
==============================
TASK 1 SUMMARY
==============================

Original Dataset : {before_total}

Cleaned Dataset : {after_total}

Training Samples : {len(train_df)}

Validation Samples : {len(validation_df)}

Testing Samples : {len(test_df)}

Duplicate Instructions : {duplicate_instruction}

Duplicate Records : {duplicate_pair}

Empty Instructions : {empty_instruction}

Empty Inputs : {empty_input}

Empty Responses : {empty_response}

Responses <30 words : {short_response}

Responses >120 words : {long_response}

Files Generated

cleaned_dataset.json

train.json

validation.json

test.json

dataset_statistics.csv

cleaning_report.csv
"""

with open(
    os.path.join(OUTPUT_DIR,"task1_summary.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(summary)

print("\n====================================")
print("TASK 1 COMPLETED SUCCESSFULLY")
print("====================================")
print(summary)