import os
import random
from tqdm import tqdm
from datasets import load_dataset

output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"

# Load the dataset
dataset = load_dataset("openwebtext", split="train")

# Split into train/val (90/10)
split = dataset.train_test_split(test_size=0.1)
train_data = split["train"]
val_data = split["test"]

# Sample 1% for realistic testing
sample_rate = 0.3
train_sample = train_data.select(random.sample(range(len(train_data)), int(len(train_data) * sample_rate)))
val_sample = val_data.select(random.sample(range(len(val_data)), int(len(val_data) * sample_rate)))

# Clear output files
open(output_file_train, 'w').close()
open(output_file_val, 'w').close()

# Write train data and collect vocab
vocab = set()

print("Processing training data...")
with open(output_file_train, "a", encoding="utf-8") as f:
    for item in tqdm(train_sample):
        text = item["text"]
        f.write(text)
        vocab.update(set(text))

# Write val data and collect vocab
print("Processing validation data...")
with open(output_file_val, "a", encoding="utf-8") as f:
    for item in tqdm(val_sample):
        text = item["text"]
        f.write(text)
        vocab.update(set(text))

# Write vocab file
with open(vocab_file, "w", encoding="utf-8") as f:
    for char in sorted(vocab):
        f.write(char + '\n')

print(f"Done! Train: {len(train_sample)} docs, Val: {len(val_sample)} docs, Vocab size: {len(vocab)}")