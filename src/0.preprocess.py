#!/usr/bin/env python3
"""
BioLaySumm Dataset Preprocessing Pipeline

This script processes scientific articles from the BioLaySumm dataset (PLOS and eLife)
for biomedical lay summarization tasks. It extracts relevant sections from articles,
cleans the text, and prepares them in a structured format suitable for model input.
"""

import os
import re
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

import nltk
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm
import pandas as pd
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def setup_nltk_resources() -> None:
    """Download required NLTK resources."""
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


def truncate_to_sentence_boundary(text: str, max_tokens: int) -> str:
    """
    Truncate text to stay under max_tokens, preserving complete sentences.

    Args:
        text: Input text to truncate
        max_tokens: Maximum allowed tokens

    Returns:
        Truncated text ending at a sentence boundary
    """
    if not text:
        return ""

    sentences = sent_tokenize(text)
    tokens_so_far = 0
    selected_sentences = []

    for sentence in sentences:
        # Simple whitespace-based tokenization for counting
        sentence_tokens = len(sentence.split())
        if tokens_so_far + sentence_tokens <= max_tokens:
            selected_sentences.append(sentence)
            tokens_so_far += sentence_tokens
        else:
            break

    return " ".join(selected_sentences)


def remove_citations(text: str) -> str:
    """
    Remove citations from text to reduce noise.

    Args:
        text: Text containing citations

    Returns:
        Text with citations removed
    """
    # Remove in-text citations like (Author et al., 2020)
    text = re.sub(r'\([^)]*\d{4}[^)]*\)', ' ', text)

    # Remove numbered citations like [1] or [1,2,3]
    text = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', text)

    # Remove footnote indicators
    text = re.sub(r'\s*\[\w+\]\s*', ' ', text)

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def simplify_technical_terms(text: str) -> str:
    """
    Simplify technical terms in biomedical text to improve readability.

    Args:
        text: Text containing technical terms

    Returns:
        Text with simplified technical terminology
    """
    # Examples of simplifications (can be expanded)
    replacements = {
        r'\bp(-?value|<0\.0\d+)\b': 'statistical significance measure',
        r'\b95%(\s+)?CI\b': '95% confidence interval',
        r'\bstatistically significant\b': 'meaningful',
        r'\bin\s+vitro\b': 'in laboratory conditions',
        r'\bin\s+vivo\b': 'in living organisms',
        r'\bmethodology\b': 'methods'
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def determine_dataset_source(article_data: Dict[str, Any]) -> str:
    """
    Determine if the article is from PLOS or eLife based on available information.

    Args:
        article_data: Dictionary containing article metadata and content

    Returns:
        Dataset source: 'plos', 'elife', or 'unknown'
    """
    source = article_data.get('source', '').lower()

    # Check explicit source field
    if 'plos' in source:
        return "plos"
    elif 'elife' in source:
        return "elife"

    # Try to detect from the content
    article_text = article_data.get('article', '')
    if article_text:
        first_1000_chars = article_text[:1000].lower()
        if 'plos' in first_1000_chars:
            return "plos"
        elif 'elife' in first_1000_chars:
            return "elife"

    return "unknown"


# ============================================================================
# Section Extraction Functions
# ============================================================================

def extract_sections_from_newline_split(article_text: str) -> Dict[str, str]:
    """
    Extract article sections from text that is split by newlines.

    Args:
        article_text: Full article text with sections separated by newlines

    Returns:
        Dictionary mapping section names to content
    """
    # Split content by newlines
    sections_content = article_text.split('\n')
    sections_content = [s.strip() for s in sections_content if s.strip()]

    # Default empty sections
    sections = {
        "abstract": "",
        "introduction": "",
        "results": "",
        "discussion": "",
        "methods": ""
    }

    # If no content, return empty sections
    if not sections_content:
        return sections

    # Map sections based on position in the text
    # First section is typically the abstract
    sections["abstract"] = sections_content[0] if sections_content else ""

    # Second section is typically introduction
    if len(sections_content) > 1:
        sections["introduction"] = sections_content[1]

    # Last section is typically discussion/conclusion
    if len(sections_content) > 2:
        sections["discussion"] = sections_content[-1]

    # Middle sections are typically results and methods
    if len(sections_content) > 3:
        # Assign results section (middle content)
        results_section = ' '.join(sections_content[2:-1])
        sections["results"] = results_section

        # If there are enough sections, try to identify methods
        if len(sections_content) > 4:
            # Methods might be second-to-last
            sections["methods"] = sections_content[-2]

    return sections


def extract_sections_from_article(article_text: str, section_headings: List[str]) -> Dict[str, str]:
    """
    Extract sections from article using section_headings list as guide.

    Args:
        article_text: The full article text with sections separated by newlines
        section_headings: List of section headings from the dataset

    Returns:
        Dictionary mapping standardized section names to content
    """
    # Default empty sections
    sections = {
        "abstract": "",
        "introduction": "",
        "results": "",
        "discussion": "",
        "methods": ""
    }

    # Split content by newlines
    sections_content = article_text.split('\n')
    sections_content = [s.strip() for s in sections_content if s.strip()]

    # If no content or section headings, return empty sections
    if not sections_content or not section_headings:
        return sections

    # Clean section headings - make lowercase and remove punctuation
    cleaned_headings = []
    for heading in section_headings:
        if not isinstance(heading, str):
            continue
        heading = heading.lower().strip()
        heading = re.sub(r'[^\w\s]', '', heading)
        cleaned_headings.append(heading)

    # Map sections based on position in the text and headings
    # First content section is usually abstract
    sections["abstract"] = sections_content[0] if sections_content else ""

    # Second content section is usually introduction
    if len(sections_content) > 1:
        sections["introduction"] = sections_content[1]

    # Last content section is usually discussion/conclusion
    if len(sections_content) > 2:
        sections["discussion"] = sections_content[-1]

    # Map results (typically between intro and discussion)
    if len(sections_content) > 3:
        # All sections between intro and discussion/methods could be results
        results_index = min(2, len(sections_content) - 2)
        results_sections = sections_content[results_index:-1]
        sections["results"] = " ".join(results_sections)

    # Map methods based on section headings
    method_keywords = ["methods", "method", "materials and methods", "experimental procedures"]
    for keyword in method_keywords:
        if any(keyword in heading for heading in cleaned_headings):
            # Methods could be the last or second-to-last section
            if len(sections_content) > 3:
                sections["methods"] = sections_content[-2]
            elif len(sections_content) > 2:
                sections["methods"] = sections_content[-1]
            break

    return sections


# ============================================================================
# Article Preprocessing Functions
# ============================================================================

def preprocess_article(article_data: Dict[str, Any], max_tokens: int = 1024,
                       section_weights: Optional[Dict[str, int]] = None) -> Tuple[str, int]:
    """
    Extract key sections from scientific articles and format for model input.

    Args:
        article_data: Dictionary containing article text and metadata
        max_tokens: Maximum number of tokens in the preprocessed output
        section_weights: Dictionary with section weights (percentages)

    Returns:
        Tuple of (input_text, token_count)
        - input_text: Preprocessed text ready for model input
        - token_count: Number of tokens in the preprocessed text
    """
    # Determine source for source-specific processing
    dataset_name = determine_dataset_source(article_data)

    # Adjust max_tokens based on source
    if dataset_name == "plos":
        max_tokens = min(max_tokens, 900)  # PLOS articles are typically shorter

    # Adjust default section weights based on source
    if section_weights is None:
        if dataset_name == "plos":
            section_weights = {
                'abstract': 40,
                'introduction': 20,
                'results': 10,
                'discussion': 25,
                'methods': 5
            }
        else:  # eLife
            section_weights = {
                'abstract': 35,
                'introduction': 20,
                'results': 15,
                'discussion': 25,
                'methods': 5
            }

    # Validate weights sum to 100
    total_weight = sum(section_weights.values())
    if total_weight != 100:
        # Normalize to 100%
        for section in section_weights:
            section_weights[section] = (section_weights[section] / total_weight) * 100

    # Get basic metadata
    title = article_data.get('title', '')

    # Process keywords if available
    keywords = []
    if isinstance(article_data.get('keywords', []), list):
        keywords = article_data.get('keywords', [])
    keywords_str = ', '.join(keywords) if keywords else ''

    # Get section headings
    section_headings = article_data.get('section_headings', [])

    # Extract article text and clean spacing
    article_text = article_data.get('article', '')
    article_text = re.sub(r'\s+([.,;:])', r'\1', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Extract sections from the article text
    sections = extract_sections_from_newline_split(article_text)

    # Clean and simplify sections
    for section_name in sections:
        # Remove citations
        sections[section_name] = remove_citations(sections[section_name])

        # Simplify technical terms for abstract, introduction, and discussion
        if section_name in ['abstract', 'introduction', 'discussion']:
            sections[section_name] = simplify_technical_terms(sections[section_name])

    # Calculate token counts for each section
    token_counts = {
        section: len(text.split()) for section, text in sections.items()
    }

    # Estimate metadata tokens (title, section markers, etc.)
    metadata_tokens = len(title.split()) + len(keywords_str.split()) + 30

    # Calculate total tokens
    content_tokens = sum(token_counts.values())
    total_tokens = content_tokens + metadata_tokens

    # If over token limit, allocate proportionally based on weights
    if total_tokens > max_tokens:
        available_tokens = max_tokens - metadata_tokens

        # Allocate tokens based on specified weights
        allocations = {}
        for section in section_weights:
            allocations[section] = min(
                token_counts.get(section, 0),
                int(available_tokens * section_weights[section] / 100)
            )

        # Adjust if allocations don't add up to available tokens
        total_allocated = sum(allocations.values())
        if total_allocated < available_tokens:
            # Distribute remaining tokens proportionally
            remaining = available_tokens - total_allocated
            for section in section_weights:
                if section_weights[section] > 0:
                    allocations[section] += int(remaining * section_weights[section] / 100)

        # Truncate each section to sentence boundaries
        for section in allocations:
            if section in sections and sections[section]:
                sections[section] = truncate_to_sentence_boundary(
                    sections[section], allocations[section]
                )

    # Assemble the final input text with sections in the correct order
    input_text = f"<{dataset_name}> [TITLE] {title}\n"

    # Add sections in the correct order
    section_order = ["abstract", "introduction", "results", "discussion", "methods"]
    for section_name in section_order:
        if sections[section_name]:
            input_text += f"[{section_name.upper()}] {sections[section_name]}\n"

    # Add keywords if available and space permits
    if keywords_str:
        input_text += f"[KEYWORDS] {keywords_str}\n"

    # Calculate final token count
    final_token_count = len(input_text.split())

    # Final check to ensure we're within token limit
    if final_token_count > max_tokens:
        # If still over limit, truncate the entire text to max_tokens
        input_text = ' '.join(input_text.split()[:max_tokens])
        final_token_count = max_tokens

    return input_text, final_token_count


# ============================================================================
# Quality Analysis Functions
# ============================================================================

def analyze_preprocessing_quality(processed_item: Dict[str, Any]) -> Tuple[int, List[str]]:
    """
    Analyze the quality of the preprocessing for a given item.

    Args:
        processed_item: Dictionary containing preprocessed article data

    Returns:
        Tuple of (quality_score, issues)
        - quality_score: Score from 0-100 indicating preprocessing quality
        - issues: List of identified issues
    """
    issues = []
    input_text = processed_item["input_text"]

    # Check if key sections are present
    if "[ABSTRACT]" not in input_text:
        issues.append("Missing abstract section")

    if "[INTRODUCTION]" not in input_text:
        issues.append("Missing introduction section")

    if "[DISCUSSION]" not in input_text:
        issues.append("Missing discussion section")

    # Check token utilization
    token_count = processed_item["token_count"]
    max_tokens = 1024  # Standard token limit

    if token_count < max_tokens * 0.5:
        issues.append(f"Low token utilization ({token_count}/{max_tokens})")
    elif token_count > max_tokens:
        issues.append(f"Exceeds token limit ({token_count}/{max_tokens})")

    # Check for technical language
    technical_patterns = [
        r'\bp(-?value|<0\.0\d+)\b',
        r'\b95%(\s+)?CI\b',
        r'\bstatistically significant\b',
        r'\bin\s+vitro\b',
        r'\bin\s+vivo\b',
        r'\bmethodology\b'
    ]

    tech_term_count = 0
    for pattern in technical_patterns:
        if re.search(pattern, input_text, re.IGNORECASE):
            tech_term_count += 1

    if tech_term_count > 3:
        issues.append(f"Contains {tech_term_count} technical terms")

    # Calculate quality score
    base_score = 100

    # Deduct for issues
    base_score -= len(issues) * 10

    # Adjust for token utilization
    token_utilization = min(token_count, max_tokens) / max_tokens
    base_score *= (0.5 + 0.5 * token_utilization)  # Scale factor for utilization

    quality_score = max(0, min(100, int(base_score)))

    return quality_score, issues


# ============================================================================
# Dataset Processing Functions
# ============================================================================

def process_batch(examples: Dict[str, List], source: str) -> Dict[str, List]:
    """
    Process a batch of examples from a dataset.

    Args:
        examples: Dictionary with batched examples from a dataset
        source: Dataset source ('plos' or 'elife')

    Returns:
        Dictionary with processed results
    """
    results = {
        "input_text": [],
        "summary": [],
        "token_count": [],
        "quality_score": [],
        "issues": []
    }

    # Process each example in the batch
    for i in range(len(examples["article"])):
        example_dict = {
            "article": examples["article"][i],
            "title": examples["title"][i],
            "source": source,
            "keywords": examples.get("keywords", [None])[i] if "keywords" in examples else [],
            "section_headings": examples.get("section_headings", [None])[i] if "section_headings" in examples else []
        }

        # Apply preprocessing
        input_text, token_count = preprocess_article(example_dict, max_tokens=1024)

        # Get preprocessing quality
        quality_score, issues = analyze_preprocessing_quality({
            "input_text": input_text,
            "token_count": token_count
        })

        # Store results
        results["input_text"].append(input_text)
        results["summary"].append(examples["summary"][i])
        results["token_count"].append(token_count)
        results["quality_score"].append(quality_score)
        results["issues"].append("|".join(issues) if issues else "None")

    return results


def process_dataset(dataset, source: str, sample_size: Optional[int] = None,
                    batch_size: int = 32) -> Dict[str, List]:
    """
    Process a dataset with batch processing to prevent memory issues.

    Args:
        dataset: The HuggingFace dataset
        source: Dataset source ('plos' or 'elife')
        sample_size: Number of examples to process (None for all)
        batch_size: Batch size for processing

    Returns:
        Dictionary with processed results
    """
    processed_data = {
        "input_text": [],
        "summary": [],
        "token_count": [],
        "quality_score": [],
        "issues": []
    }

    # Process only a sample if specified
    if sample_size is not None:
        if sample_size > len(dataset):
            sample_size = len(dataset)
        indices = list(range(sample_size))
    else:
        indices = list(range(len(dataset)))

    # Process in batches
    for i in tqdm(range(0, len(indices), batch_size), desc=f"Processing {source}"):
        batch_indices = indices[i:i + batch_size]
        batch = dataset.select(batch_indices)
        batch_results = process_batch(batch, source)

        # Extend results
        for key in processed_data:
            processed_data[key].extend(batch_results[key])

    return processed_data


def save_processed_data(processed_data: Dict[str, List], output_path: str,
                        format: str = "jsonl") -> None:
    """
    Save processed data to disk in the specified format.

    Args:
        processed_data: Dictionary with processed results
        output_path: Path to save the output file
        format: Output format ('jsonl' or 'csv')
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if format == "jsonl":
        with open(output_path, 'w') as f:
            for i in range(len(processed_data["input_text"])):
                record = {
                    "input_text": processed_data["input_text"][i],
                    "summary": processed_data["summary"][i],
                    "token_count": processed_data["token_count"][i],
                    "quality_score": processed_data["quality_score"][i],
                    "issues": processed_data["issues"][i]
                }
                f.write(json.dumps(record) + "\n")
        logger.info(f"Saved {len(processed_data['input_text'])} records to {output_path}")
    elif format == "csv":
        pd.DataFrame(processed_data).to_csv(output_path, index=False)
        logger.info(f"Saved {len(processed_data['input_text'])} records to {output_path}")
    else:
        raise ValueError(f"Unsupported format: {format}")


# ============================================================================
# Main Script Functions
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess BioLaySumm dataset for training and inference'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='processed_data',
        help='Directory to save processed data'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['plos', 'elife', 'both'],
        default='both',
        help='Which dataset to process'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (None for all)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=1024,
        help='Maximum tokens for each processed article'
    )
    parser.add_argument(
        '--verify_sample_size',
        type=int,
        default=10,
        help='Number of samples to process for verification'
    )
    parser.add_argument(
        '--skip_verification',
        action='store_true',
        help='Skip the verification step'
    )
    parser.add_argument(
        '--output_format',
        type=str,
        choices=['jsonl', 'csv'],
        default='jsonl',
        help='Output file format'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode (ask for confirmation)'
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    # Parse arguments
    args = parse_args()

    # Set up NLTK resources
    setup_nltk_resources()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load datasets
    logger.info("Loading datasets from Hugging Face...")
    datasets_to_process = []

    if args.dataset in ['plos', 'both']:
        try:
            plos = load_dataset("BioLaySumm/BioLaySumm2025-PLOS")
            logger.info(f"PLOS dataset loaded: {plos}")
            datasets_to_process.append(('plos', plos))
        except Exception as e:
            logger.error(f"Failed to load PLOS dataset: {e}")

    if args.dataset in ['elife', 'both']:
        try:
            elife = load_dataset("BioLaySumm/BioLaySumm2025-eLife")
            logger.info(f"eLife dataset loaded: {elife}")
            datasets_to_process.append(('elife', elife))
        except Exception as e:
            logger.error(f"Failed to load eLife dataset: {e}")

    if not datasets_to_process:
        logger.error("No datasets were loaded successfully. Exiting.")
        return

    # Verification step
    if not args.skip_verification:
        logger.info(f"Processing {args.verify_sample_size} samples from each dataset for verification...")

        for dataset_name, dataset in datasets_to_process:
            try:
                samples = process_dataset(
                    dataset['train'],
                    dataset_name,
                    sample_size=args.verify_sample_size,
                    batch_size=args.batch_size
                )

                sample_output_path = os.path.join(args.output_dir, f"{dataset_name}_samples.{args.output_format}")
                save_processed_data(samples, sample_output_path, format=args.output_format)

                # Display an example
                logger.info(f"Example from {dataset_name.upper()}:")
                truncated_example = samples["input_text"][0][:500] + "..." if samples[
                    "input_text"] else "No example available"
                logger.info(truncated_example)

            except Exception as e:
                logger.error(f"Error during verification for {dataset_name}: {e}")

        # Check if we should continue
        if args.interactive:
            process_full = input("\nDo you want to process the full datasets? (y/n): ")
            if process_full.lower() != 'y':
                logger.info("Skipping full dataset processing. Sample results are available in the output directory.")
                return

    # Process full datasets
    for dataset_name, dataset in datasets_to_process:
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                try:
                    logger.info(f"Processing {dataset_name} {split} split...")
                    sample_size = min(len(dataset[split]), args.max_samples) if args.max_samples else None
                    processed_data = process_dataset(
                        dataset[split],
                        dataset_name,
                        sample_size=sample_size,
                        batch_size=args.batch_size
                    )

                    output_path = os.path.join(args.output_dir, f"{dataset_name}_{split}.{args.output_format}")
                    save_processed_data(processed_data, output_path, format=args.output_format)

                except Exception as e:
                    logger.error(f"Error processing {dataset_name} {split} split: {e}")

    logger.info("All processing complete!")


if __name__ == "__main__":
    main()
