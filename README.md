# BioLaySumm 2025 1.1 [The Lay People in Question]

https://biolaysumm.org/

## Environment Setup

### Setting up with Miniconda

We recommend using a Miniconda environment for this project to manage dependencies:

1. **Install Miniconda** (if not already installed):
   - Download the installer from [Miniconda website](https://docs.conda.io/en/latest/miniconda.html)
   - Follow the installation instructions for your operating system

2. **Create a new conda environment**:
   ```bash
   conda create -n biolaysumm python=3.12
   conda activate biolaysumm
   ```

3. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

This will create an isolated environment with all necessary dependencies for preprocessing and model training.

## Preprocessing

The preprocessing pipeline prepares scientific articles from the PLOS and eLife datasets for the biomedical lay summarization task.

### 0.preprocess.py

This script extracts key sections from scientific articles (abstract, introduction, results, discussion, methods), removes citations, simplifies technical terminology, and formats the text with appropriate section markers. It handles token limitations by intelligently truncating less important sections while preserving complete sentences.

Usage:

```bash
python src/0.preprocess.py \
    --output_dir /data \
    --dataset both \
    --max_tokens 1024 \
    --batch_size 32 \
    --output_format jsonl
 ```

Key parameters:
- `--output_dir`: Directory to save processed data
- `--dataset`: Which dataset to process (`plos`, `elife`, or `both`)
- `--max_tokens`: Maximum token limit per article (default: 1024)
- `--batch_size`: Batch size for processing (adjust based on available memory)
- `--output_format`: Output format (`jsonl` or `csv`)

The script produces structured files containing preprocessed articles with their corresponding summaries, ready for model training and evaluation.

## Finetuning (Optional)

### 1.finetune.py

This script finetunes a pretrained language model (default: FLAN-T5) on the preprocessed BioLaySumm dataset to generate lay summaries of scientific articles.

Usage:
```bash
python src/1.finetune.py \
    # Data paths
    --drive_path /data \
    --output_dir /models/biosum_flan_t5 \
    
    # Model configuration
    --model_name google/flan-t5-base \
    --prompt_template "create a lay summary of this scientific research for a general audience who has no background in biology: " \
    
    # Sequence length settings
    --max_input_length 1024 \
    --max_output_length 128 \
    
    # Training hyperparameters
    --batch_size 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --epochs 3 \
    --warmup_ratio 0.1 \
    --grad_accumulation 2 \
    --seed 42 \
    --fp16 \
    
    # Evaluation and checkpoint settings
    --eval_strategy epoch \
    --logging_steps 100 \
    --save_total_limit 3 \
    --patience 3 \
    
    # Optional Weights & Biases integration
    --use_wandb \
    --wandb_project biosum-finetuning \
    --wandb_run_name flan-t5-biosum
```

Key parameters:
- `--drive_path`: Path to directory containing preprocessed JSONL files from Step 0
- `--output_dir`: Directory to save the model checkpoints and outputs
- `--model_name`: Base model to finetune (default: google/flan-t5-base)
- `--prompt_template`: Text prompt to prepend to input articles
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate for training
- `--epochs`: Number of training epochs
- `--fp16`: Enable mixed precision training (faster on compatible GPUs)
- `--use_wandb`: Enable Weights & Biases logging (optional)

The script will:
1. Load preprocessed data from both PLOS and eLife datasets
2. Tokenize inputs with the specified prompt template
3. Train the model with early stopping based on ROUGE scores
4. Save model checkpoints to the specified output directory
5. Evaluate on test data and generate example summaries

Output files:
- Model checkpoint files in the specified output directory
- `prompt_template.txt`: Record of the prompt used during training
- `test_results.json`: Evaluation metrics on the test set
- `example_generations.json`: Sample generated summaries from test cases

The finetuned model can then be used for inference on new articles or for evaluation.


## Inference & Generation

### 2.generate.py
This script loads a finetuned model and generates lay summaries for the test datasets.
Usage:
```bash
python src/2.generate.py \
    # Model paths
    --model_path /ibrahimsharaf/biolaysumm573 \
    
    # Input data
    --elife_path /data/elife_dev.jsonl \
    --plos_path /data/plos_dev.jsonl \
    
    # Output settings
    --output_dir /data/predictions \
    --split_name test \
    
    # Generation parameters
    --prompt "create a lay summary of this scientific research for a general audience who has no background in biology: " \
    --batch_size 16 \
    --max_length 400 \
    --min_length 120 \
    --num_beams 8 \
    --num_beam_groups 4 \
    --diversity_penalty 0.8 \
    --repetition_penalty 1.2 \
    --length_penalty 1.4 \
    --no_repeat_ngram_size 3
```
The script will:

Load the finetuned model from the specified checkpoint
Process both PLOS and eLife test datasets
Generate summaries using diverse beam search with the configured parameters
Save predictions to CSV files in the output directory.


## Evaluation

### 3.evaluate.py
This script evaluates generated summaries using multiple metrics across three dimensions:

Relevance: ROUGE, BERTScore, METEOR, BLEU
Readability: Flesch-Kincaid, Dale-Chall, Coleman-Liau
Factuality: LENS, AlignScore, SummaC
Usage:

```bash
python src/3.evaluate.py \
    # Input/output paths
    --predictions_dir /data/predictions/dev \
    --output_dir /data/evaluation/dev \
    
    # Dataset options
    --dataset both \
    
    # Column names in prediction files
    --input_col input_text \
    --reference_col summary \
    --prediction_col predicted_summary \
    
    # Metrics selection
    --relevance_metrics \
    --readability_metrics \
    --factuality_metrics \
    
    # Skip slow metrics if needed
    --skip_lens \
    
    # Technical settings
    --batch_size 16 \
    --alignscore_path ./models/AlignScore/AlignScore-base.ckpt
```

For quick testing, you can run:

```bash
python src/3.evaluate.py --test_mode
```

There are some conflicting dependencies in the evaluation systems; for example, LENS and AlignScore want different versions of pytorch-lightning. You'll get some error messages when installing but they don't seem to break anything. Follow these steps to get everything set up:

Make sure your system has the wget command available for the terminal; if you're on a Mac with Homebrew, you can `brew install wget`
`pip install -r requirements-eval.txt` to install all dependencies.
Run bash `get_models.sh` to clone and install AlignScore
Note that while all the metrics can run on a CPU, the model-based ones (AlignScore, Summac, especially LENS) will want CUDA-enabled GPU to run efficiently (https://pytorch.org/get-started/locally/).




