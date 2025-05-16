#!/usr/bin/env python3
"""
BioLaySumm Inference Script

This script loads a finetuned model and generates summaries for test datasets.
"""

import os
import json
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate lay summaries using a finetuned model")

    # Model paths
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint folder (e.g., checkpoint-3500)")

    # Input data
    parser.add_argument("--elife_path", type=str, required=True,
                        help="Path to eLife test/validation data (JSONL)")
    parser.add_argument("--plos_path", type=str, required=True,
                        help="Path to PLOS test/validation data (JSONL)")

    # Output settings
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated summaries")
    parser.add_argument("--split_name", type=str, default="test",
                        help="Name of the data split (e.g., 'test', 'dev')")

    # Generation parameters
    parser.add_argument("--prompt", type=str,
                        default="create a lay summary of this scientific research for a general audience who has no background in biology: ",
                        help="Prompt template for generation")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation")
    parser.add_argument("--max_length", type=int, default=400,
                        help="Maximum length of generated summaries")
    parser.add_argument("--min_length", type=int, default=120,
                        help="Minimum length of generated summaries")
    parser.add_argument("--num_beams", type=int, default=8,
                        help="Number of beams for beam search")
    parser.add_argument("--num_beam_groups", type=int, default=4,
                        help="Number of beam groups for diverse beam search")
    parser.add_argument("--diversity_penalty", type=float, default=0.8,
                        help="Diversity penalty for diverse beam search")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                        help="Size of n-grams to avoid repeating")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Penalty for repetition")
    parser.add_argument("--length_penalty", type=float, default=1.4,
                        help="Length penalty (>1 favors longer sequences)")
    parser.add_argument("--do_sample", action="store_true",
                        help="Use sampling instead of deterministic decoding")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Maximum input length for the model")

    # Misc settings
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, cpu, or auto for automatic detection)")
    parser.add_argument("--skip_combined", action="store_true",
                        help="Skip generation for the combined dataset")

    return parser.parse_args()


def load_jsonl(file_path):
    """
    Load data from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries containing the data
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading JSONL file {file_path}: {e}")
        return []


def get_source_text(row):
    """
    Extract source text from row, handling different data formats.

    Args:
        row: Dictionary containing the data

    Returns:
        Source text from the row
    """
    if 'input_text' in row:  # Our processed format
        return row['input_text']
    elif 'text' in row:  # Alternative format
        return row['text']
    elif 'article' in row:  # Raw dataset format
        return row['article']
    else:
        # Try to find the text field by looking for long strings
        text_fields = [field for field in row.keys()
                       if isinstance(row[field], str) and len(row[field]) > 100]
        if text_fields:
            return row[text_fields[0]]
        else:
            print(f"Warning: Could not find text field in row: {list(row.keys())}")
            return ""


def get_reference_summary(row):
    """
    Extract reference summary from row, handling different data formats.

    Args:
        row: Dictionary containing the data

    Returns:
        Reference summary from the row
    """
    if 'summary' in row:
        return row['summary']
    else:
        print(f"Warning: Could not find summary field in row: {list(row.keys())}")
        return ""


def batch_generate_summaries(texts, tokenizer, model, generation_params, prompt, device, progress_bar=None):
    """
    Generate summaries for a batch of texts.

    Args:
        texts: List of source texts
        tokenizer: Tokenizer to use
        model: Model to use
        generation_params: Parameters for generation
        prompt: Prompt template to use
        device: Device to use
        progress_bar: Progress bar to update

    Returns:
        List of generated summaries
    """
    # Prepare prompts
    prompts = [f"{prompt}{text}" for text in texts]

    # Tokenize inputs in batch
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=args.max_input_length,
        return_tensors="pt"
    ).to(device)

    # Generate summaries
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_params
        )

    # Decode summaries
    summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Update progress bar
    if progress_bar is not None:
        progress_bar.update(len(texts))

    return summaries


def process_dataset(data, dataset_name, tokenizer, model, args, device):
    """
    Process a dataset and generate summaries in batches.

    Args:
        data: List of dictionaries containing the data
        dataset_name: Name of the dataset
        tokenizer: Tokenizer to use
        model: Model to use
        args: Command line arguments
        device: Device to use

    Returns:
        DataFrame containing the data with generated summaries
    """
    print(f"Processing {dataset_name} dataset ({len(data)} samples)...")

    # Convert to DataFrame
    df_data = []
    for item in data:
        df_item = {
            "input_text": get_source_text(item),
            "summary": get_reference_summary(item)
        }
        df_data.append(df_item)

    df = pd.DataFrame(df_data)

    # Prepare generation parameters
    generation_params = {
        "max_length": args.max_length,
        "min_length": args.min_length,
        "num_beams": args.num_beams,
        "num_beam_groups": args.num_beam_groups,
        "diversity_penalty": args.diversity_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "early_stopping": True,
        "do_sample": args.do_sample
    }

    # Generate summaries in batches
    source_texts = df["input_text"].tolist()
    summaries = []

    # Set up progress bar
    total_samples = len(source_texts)
    with tqdm(total=total_samples, desc=f"Generating {dataset_name} summaries") as pbar:
        for i in range(0, total_samples, args.batch_size):
            batch_texts = source_texts[i:i + args.batch_size]
            batch_summaries = batch_generate_summaries(
                batch_texts,
                tokenizer,
                model,
                generation_params,
                args.prompt,
                device,
                pbar
            )
            summaries.extend(batch_summaries)

    # Add predicted summaries to DataFrame
    df["predicted_summary"] = summaries

    # Calculate statistics about summary lengths
    summary_lengths = [len(summary.split()) for summary in summaries]
    print(f"Summary length statistics (in words):")
    print(f"  Min: {min(summary_lengths)}")
    print(f"  Max: {max(summary_lengths)}")
    print(f"  Avg: {np.mean(summary_lengths):.1f}")
    print(f"  Median: {np.median(summary_lengths):.1f}")

    # Save to CSV
    output_path = os.path.join(args.output_dir, f"{dataset_name}_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    return df


def main(args):
    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Construct model path
    if args.checkpoint:
        full_model_path = os.path.join(args.model_path, args.checkpoint)
    else:
        full_model_path = args.model_path

    # Load tokenizer and model
    print(f"Loading model from {full_model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(full_model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_path).to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    try:
        elife_data = load_jsonl(args.elife_path)
        plos_data = load_jsonl(args.plos_path)
        print(f"Loaded {len(elife_data)} eLife samples and {len(plos_data)} PLOS samples.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Switch to eval mode for inference
    model.eval()

    # Process PLOS dataset
    print("\n==== PLOS Dataset ====")
    plos_df = process_dataset(plos_data, "plos", tokenizer, model, args, device)

    # Process eLife dataset
    print("\n==== eLife Dataset ====")
    elife_df = process_dataset(elife_data, "elife", tokenizer, model, args, device)

    # Process combined dataset if not skipped
    if not args.skip_combined:
        print("\n==== Combined Dataset ====")
        combined_data = elife_data + plos_data
        combined_df = process_dataset(combined_data, "combined", tokenizer, model, args, device)

    print("\nAll processing complete!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
