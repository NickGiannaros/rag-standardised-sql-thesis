"""
1_load_data.py
What this does: downloads the Spider training set and saves it locally as
JSON. This is the pool of (question, SQL) examples the memory will later
retrieve from.
Input: none -- pulls directly from HuggingFace ('xlangai/spider').
Output: sample_train.json (7000 question/SQL/db_id records).
"""
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
import json

print("Loading Spider dataset...")
spider = load_dataset("xlangai/spider")
train = spider["train"]

train_sample = list(train)

print("Train examples (full):", len(train_sample))

with open("sample_train.json", "w") as f:
    json.dump(train_sample, f, indent=2)

print("Saved sample_train.json (full train set)")