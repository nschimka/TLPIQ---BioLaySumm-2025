"""
BioLaySumm Finetuning Script with Combined Datasets and W&B Integration
"""

import argparse
import json
import os
import random
import numpy as np
import evaluate
import torch
import wandb
from datasets import Dataset, DatasetDict, concatenate_datasets
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    GenerationConfig
)
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune FLAN-T5 on BioLaySumm dataset")

    # Data paths
    parser.add_argument("--drive_path", type=str, default="/content/drive/MyDrive/573_BioLaySumm/preprocessed_data",
                      help="Path to directory containing preprocessed JSONL files")

    parser.add_argument("--dataset_source", type=str, default="both",
                  choices=["both", "plos", "elife"],
                  help="Which dataset to use: both, plos, or elife")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base",
                      help="Pretrained model name or path")
    parser.add_argument("--prompt_template", type=str,
                      default="create a lay summary of this scientific research for a general audience who has no background in biology: ",
                      help="Prompt template to use for instruction tuning")
    parser.add_argument("--output_dir", type=str, default="biosum_model",
                      help="Directory to save the model")

    # Training parameters
    parser.add_argument("--max_input_length", type=int, default=1024,
                      help="Maximum input sequence length")
    parser.add_argument("--max_output_length", type=int, default=128,
                      help="Maximum output sequence length")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                  help="Batch size per device during evaluation (if None, uses batch_size)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                      help="Ratio of steps for warmup")
    parser.add_argument("--grad_accumulation", type=int, default=2,
                      help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--fp16", action="store_true",
                      help="Use mixed precision training")

    # Evaluation and logging
    parser.add_argument("--eval_strategy", type=str, default="epoch",
                      help="Evaluation strategy (steps or epoch)")
    parser.add_argument("--eval_steps", type=int, default=500,
                      help="Number of update steps between evaluations if using steps strategy")
    parser.add_argument("--logging_steps", type=int, default=100,
                      help="Number of steps between logging")
    parser.add_argument("--save_total_limit", type=int, default=3,
                      help="Max number of checkpoints to keep")
    parser.add_argument("--patience", type=int, default=3,
                      help="Early stopping patience")

    # W&B configuration
    parser.add_argument("--use_wandb", action="store_true",
                      help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="biosum-finetuning",
                      help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="flan-t5-biosum",
                      help="W&B run name")

    return parser.parse_args()

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl_data(file_path):
    """Load data from a JSONL file for BioLaySumm."""
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return data

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                example = json.loads(line.strip())
                # Extract the fields we need
                processed_example = {
                    "text": example.get("input_text", ""),
                    "summary": example.get("summary", ""),
                    "source": os.path.basename(file_path).split('_')[0]  # Extract source (plos or elife)
                }
                # Only add examples that have both text and summary
                if processed_example["text"] and processed_example["summary"]:
                    data.append(processed_example)
            except json.JSONDecodeError:
                print(f"Error parsing JSON line in {file_path}")
                continue
    return data


def load_combined_datasets(args):
    """Load datasets based on source selection."""
    data_dir = args.drive_path

    # Define file paths
    files = {
        'plos_train': os.path.join(data_dir, 'plos_train.jsonl'),
        'plos_val': os.path.join(data_dir, 'plos_validation.jsonl'),
        'plos_test': os.path.join(data_dir, 'plos_test.jsonl'),
        'elife_train': os.path.join(data_dir, 'elife_train.jsonl'),
        'elife_val': os.path.join(data_dir, 'elife_validation.jsonl'),
        'elife_test': os.path.join(data_dir, 'elife_test.jsonl')
    }

    print(files)

    # Check for file existence and print status
    print("\nChecking dataset files:")
    for name, path in files.items():
        status = "✓ Found" if os.path.exists(path) else "✗ Not found"
        print(f"{name}: {status}")

    # Load datasets based on source selection
    combined_data = {'train': [], 'val': [], 'test': []}

    # Determine which sources to load
    sources_to_load = []
    if args.dataset_source == "both":
        sources_to_load = ['plos', 'elife']
    elif args.dataset_source == "plos":
        sources_to_load = ['plos']
    elif args.dataset_source == "elife":
        sources_to_load = ['elife']

    # Load training data
    for source in sources_to_load:
        train_file = files[f'{source}_train']
        if os.path.exists(train_file):
            print(f"\nLoading {source} training data from {train_file}")
            train_data = load_jsonl_data(train_file)
            print(f"  Loaded {len(train_data)} examples")
            combined_data['train'].extend(train_data)

    # Load validation data
    for source in sources_to_load:
        val_file = files[f'{source}_val']
        if os.path.exists(val_file):
            print(f"Loading {source} validation data from {val_file}")
            val_data = load_jsonl_data(val_file)
            print(f"  Loaded {len(val_data)} examples")
            combined_data['val'].extend(val_data)

    # Load test data
    for source in sources_to_load:
        test_file = files[f'{source}_test']
        if os.path.exists(test_file):
            print(f"Loading {source} test data from {test_file}")
            test_data = load_jsonl_data(test_file)
            print(f"  Loaded {len(test_data)} examples")
            combined_data['test'].extend(test_data)

    # Shuffle the data
    for split in combined_data:
        random.shuffle(combined_data[split])

    # Convert to HuggingFace datasets
    hf_datasets = {}
    for split, examples in combined_data.items():
        if examples:
            print(f"Creating {split} dataset with {len(examples)} examples")
            hf_datasets[split] = Dataset.from_pandas(pd.DataFrame(examples))

    return DatasetDict(hf_datasets)


def main():
    # Parse arguments
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

    # Ensure the output directory exists in Google Drive
    output_dir = os.path.join(args.drive_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load combined datasets
    print("Loading and combining datasets...")
    datasets = load_combined_datasets(args)

    # Print dataset statistics
    print("\nDataset Statistics:")
    total_examples = 0
    for split, dataset in datasets.items():
        print(f"  {split}: {len(dataset)} examples")
        total_examples += len(dataset)
    print(f"Total examples: {total_examples}")

    # Load the tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    model.gradient_checkpointing_enable()

    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define preprocessing function using the specified prompt template
    def preprocess_function(examples):
        inputs = [args.prompt_template + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True)
        labels = tokenizer(text_target=examples["summary"], max_length=args.max_output_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Tokenize datasets
    print("Tokenizing datasets...")
    print(f"Using prompt template: '{args.prompt_template}'")
    tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=["text", "summary", "source"])

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8 if args.fp16 else None
    )

    # Set up evaluation metrics
    rouge = evaluate.load("rouge")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 with pad token id for decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Clean up predictions and references
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Compute ROUGE scores
        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )

        # Add mean generated length
        prediction_lengths = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lengths)

        # Round all scores
        return {k: round(v, 4) for k, v in result.items()}

    # Determine evaluation strategy settings
    if args.eval_strategy == "steps":
        eval_strategy = "steps"
        eval_steps = args.eval_steps
    else:
        eval_strategy = "epoch"
        eval_steps = None


    generation_config = GenerationConfig(
        # Core parameters
        max_new_tokens=args.max_output_length,
        min_new_tokens=100,  # Enforce minimum length for comprehensive summaries

        # Beam search configuration
        num_beams=6,  # Must be divisible by num_beam_groups
        num_beam_groups=2,  # So 6÷2=3 beams per group
        diversity_penalty=1.0,  # Optimal value for biomedical content

        # Quality controls
        early_stopping=True,
        length_penalty=1.2,  # Slightly favor longer sequences (>1.0)
        repetition_penalty=1.3,  # Penalize repetition
        no_repeat_ngram_size=3,  # Avoid repeating trigrams

        # Sampling parameters (with beam search)
        do_sample=False,  # Must be False with diverse beam search

        # Force certain generation patterns
        forced_bos_token_id=None,  # Let the model determine the beginning naturally

        # Logical flow
        encoder_repetition_penalty=1.2,  # Reduce repetition of source content

        # Stop conditions
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id
    )


    # Configure training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=eval_strategy,
        save_steps=eval_steps if eval_steps else None,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size else args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        gradient_accumulation_steps=args.grad_accumulation,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.wandb_run_name if args.use_wandb else None,
        generation_config=generation_config
    )

    # Initialize the trainer with early stopping
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("val", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # Print training configuration
    print("\nTraining Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Prompt: '{args.prompt_template}'")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Gradient accumulation: {args.grad_accumulation}")
    print(f"  Evaluation strategy: {args.eval_strategy}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Using W&B: {args.use_wandb}")

    # Start the training
    print(f"\nStarting training...")
    trainer.train()

    # Save the final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save the prompt template with the model for future reference
    with open(os.path.join(output_dir, "prompt_template.txt"), "w") as f:
        f.write(args.prompt_template)

    # Run evaluation on test set
    if "test" in tokenized_datasets:
        print("\nEvaluating on test set...")
        test_results = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")

        # Print test results
        print("\nTest Results:")
        for key, value in test_results.items():
            print(f"  {key}: {value}")

        # Log test results to W&B
        if args.use_wandb:
            wandb.log(test_results)

        # Save test results to a file
        with open(os.path.join(output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)

    # Generate examples from test set
    if "test" in datasets:
        print("\n=== Test Set Examples ===")

        # Use the model device
        device = trainer.model.device

        # Sample a few examples
        num_examples = min(5, len(datasets["test"]))
        test_examples = [datasets["test"][i] for i in range(num_examples)]

        examples_output = []
        for i, example in enumerate(test_examples):
            # Prepare input
            input_text = example["text"]
            input_ids = tokenizer(
                f"{args.prompt_template}{input_text}",
                return_tensors="pt",
                max_length=args.max_input_length,
                truncation=True
            ).input_ids.to(device)

            # Generate summary
            with torch.no_grad():
                outputs = trainer.model.generate(
                    input_ids,
                    max_length=args.max_output_length,
                    num_beams=4,
                    early_stopping=True
                )
            generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Print example
            print(f"\nExample {i+1}:")
            print(f"Original summary: {example['summary'][:150]}...")
            print(f"Generated summary: {generated_summary[:150]}...")

            # Save for the output file
            examples_output.append({
                "source": example["source"],
                "input_text": input_text[:500] + "..." if len(input_text) > 500 else input_text,
                "reference_summary": example["summary"],
                "generated_summary": generated_summary
            })

        # Save examples to a file
        with open(os.path.join(output_dir, "example_generations.json"), "w") as f:
            json.dump(examples_output, f, indent=2)

    print("\nTraining complete!")
    print(f"Model saved to {output_dir}")

    # Finish W&B run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    # Mount Google Drive if running in Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully")
    except:
        print("Not running in Colab or Drive already mounted")

    main()
