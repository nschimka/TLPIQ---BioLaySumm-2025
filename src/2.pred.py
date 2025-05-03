import os
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

# Configuration
MODEL_CHECKPOINT = "checkpoint-3500"  # Your checkpoint folder
MODEL_PATH = os.path.join("/content/drive/MyDrive/573_BioLaySumm/checkpoints/biosum_flan_t5_base/", MODEL_CHECKPOINT)
ELIFE_TEST_PATH = "/content/drive/MyDrive/573_BioLaySumm/data/elife_validation.jsonl"
PLOS_TEST_PATH = "/content/drive/MyDrive/573_BioLaySumm/data/plos_validation.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/573_BioLaySumm/predictions/dev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # Increase this to utilize more GPU memory

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer and model
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
print("Model loaded successfully.")

# Switch to eval mode for inference
model.eval()

# Optimized diverse beam search parameters
diverse_beam_params = {
    "max_length": 400,             # Ensure full coverage of abstract
    "min_length": 120,             # Avoid short, uninformative summaries
    "num_beams": 8,                # Strong baseline for quality-diversity tradeoff
    "num_beam_groups": 4,          # Increase for more distinct summaries
    "diversity_penalty": 0.8,      # Moderate diversity; avoids disjoint/unnatural output
    "no_repeat_ngram_size": 3,     # Standard for reducing redundancy
    "repetition_penalty": 1.2,     # Mild penalty to encourage novelty
    "length_penalty": 1.4,         # Balanced between verbosity and conciseness
    "early_stopping": True,        # Stop when best beam ends; good for speed
    "do_sample": False             # Deterministic — good for eval consistency
}


def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_source_text(row):
    """Extract source text from row, handling different data formats."""
    if 'source' in row:
        return row['source']
    elif 'text' in row:
        return row['text']
    elif 'input_text' in row:
        return row['input_text']
    else:
        # Try to find the text field
        text_fields = [field for field in row.keys()
                      if isinstance(row[field], str) and len(row[field]) > 100]
        if text_fields:
            return row[text_fields[0]]
        else:
            return ""

def batch_generate_summaries(texts, progress_bar=None):
    """Generate summaries for a batch of texts."""
    # Prepare prompts
    prompts = [f"create a lay summary of this scientific research for a general audience who has no background in biology: {text}" for text in texts]

    # Tokenize inputs in batch
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    ).to(DEVICE)

    # Generate summaries
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **diverse_beam_params
        )

    # Decode summaries
    summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Update progress bar
    if progress_bar is not None:
        progress_bar.update(len(texts))

    return summaries

def process_dataset(data, dataset_name):
    """Process a dataset and generate summaries in batches."""
    print(f"Processing {dataset_name} dataset...")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Extract source texts
    source_texts = [get_source_text(row) for _, row in df.iterrows()]

    # Generate summaries in batches
    summaries = []

    # Set up progress bar
    total_samples = len(source_texts)
    with tqdm(total=total_samples, desc=f"Generating {dataset_name} summaries") as pbar:
        for i in range(0, total_samples, BATCH_SIZE):
            batch_texts = source_texts[i:i+BATCH_SIZE]
            batch_summaries = batch_generate_summaries(batch_texts, pbar)
            summaries.extend(batch_summaries)

    # Add predicted summaries to DataFrame
    df["predicted_summary"] = summaries

    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    return df

def main():
    # Load datasets
    print("Loading datasets...")
    try:
        elife_data = load_jsonl(ELIFE_TEST_PATH)
        plos_data = load_jsonl(PLOS_TEST_PATH)
        print(f"Loaded {len(elife_data)} eLife samples and {len(plos_data)} PLOS samples.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Process PLOS dataset
    print("\n==== PLOS Dataset ====")
    plos_df = process_dataset(plos_data, "plos")

    # Process eLife dataset
    print("\n==== eLife Dataset ====")
    elife_df = process_dataset(elife_data, "elife")

    # Combine datasets
    print("\n==== Combined Dataset ====")
    combined_data = elife_data + plos_data
    combined_df = process_dataset(combined_data, "combined")

    print("\nAll processing complete!")

if __name__ == "__main__":
    main()
