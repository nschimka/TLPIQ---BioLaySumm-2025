import os
import re
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from datasets import load_dataset
import re
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt_tab')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


def truncate_to_sentence_boundary(text, max_tokens):
    """
    Truncate text to stay under max_tokens, preserving complete sentences.
    """
    if not text:
        return ""

    sentences = sent_tokenize(text)
    tokens_so_far = 0
    selected_sentences = []

    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if tokens_so_far + sentence_tokens <= max_tokens:
            selected_sentences.append(sentence)
            tokens_so_far += sentence_tokens
        else:
            break

    return " ".join(selected_sentences)


def determine_dataset_source(article_data):
    """
    Determine if the article is from PLOS or eLife based on available information.
    """
    source = article_data.get('source', '')
    if 'plos' in source.lower():
        return "plos"
    elif 'elife' in source.lower():
        return "elife"
    else:
        # Try to detect from the content
        article_text = article_data.get('article', '')
        if article_text:
            first_1000_chars = article_text[:1000].lower()
            if 'plos' in first_1000_chars:
                return "plos"
            elif 'elife' in first_1000_chars:
                return "elife"
        return "unknown"


def extract_sections_from_article(article_text, section_headings):
    """
    Extract sections from article using section_headings list as guide.
    Since actual section headings aren't in the text, we use the position in the
    newline-separated content to infer sections.

    Parameters:
    - article_text: The full article text with sections separated by newlines
    - section_headings: List of section headings from the dataset

    Returns:
    - Dictionary mapping standardized section names to content
    """
    # Split content by newlines
    sections_content = article_text.split('\n')
    sections_content = [s.strip() for s in sections_content if s.strip()]

    # If no content or section headings, return empty dict
    if not sections_content or not section_headings:
        return {
            "abstract": "",
            "introduction": "",
            "results": "",
            "discussion": "",
            "methods": ""
        }

    # Clean section headings - make lowercase and remove punctuation
    cleaned_headings = []
    for heading in section_headings:
        heading = heading.lower().strip()
        heading = re.sub(r'[^\w\s]', '', heading)
        cleaned_headings.append(heading)

    # Create mapping of extracted sections based on position in text
    # Assuming sections are in the same order in the text as in section_headings
    sections = {}

    # Always map first content section to abstract if it exists
    sections["abstract"] = sections_content[0] if sections_content else ""

    # Map introduction (usually second content section)
    if len(sections_content) > 1:
        sections["introduction"] = sections_content[1]
    else:
        sections["introduction"] = ""

    # Map discussion/conclusion (usually last content section)
    if len(sections_content) > 2:
        sections["discussion"] = sections_content[-1]
    else:
        sections["discussion"] = ""

    # Map results (if exists, typically between intro and discussion)
    sections["results"] = ""
    if len(sections_content) > 3:
        # If there are more than 3 sections, all middle sections (except the last 2) could be results
        if "results" in cleaned_headings or "result" in cleaned_headings:
            # The middle sections are typically results
            results_index = min(2, len(sections_content)-2)  # Start after intro, but before the last section
            results_sections = sections_content[results_index:-1]
            sections["results"] = " ".join(results_sections)

    # Map methods (usually last or second-to-last section)
    sections["methods"] = ""
    if "methods" in cleaned_headings or "method" in cleaned_headings or "materials and methods" in cleaned_headings:
        # Look for methods in the last few sections
        if len(sections_content) > 2:
            # Methods could be the last section or second-to-last
            sections["methods"] = sections_content[-2] if len(sections_content) > 3 else sections_content[-1]

    return sections

def preprocess_article(article_data, max_tokens=1024, section_weights=None):
    """
    Extract key sections from scientific articles with customizable section weighting
    based on the insight that sections are divided by newlines without headings.

    Parameters:
    - article_data: Dictionary containing article text and metadata
    - max_tokens: Maximum number of tokens in the preprocessed output (default 1024)
    - section_weights: Dictionary with section weights (percentages)

    Returns:
    - input_text: Preprocessed text ready for model input
    - token_count: Number of tokens in the preprocessed text
    """
    # Determine source for source-specific processing
    dataset_name = determine_dataset_source(article_data)

    # Adjust max_tokens based on source
    if dataset_name == "plos":
        max_tokens = min(max_tokens, 1000)  # PLOS articles are shorter

    # Adjust default section weights based on EDA findings
    if section_weights is None:
        # Based on EDA: abstract and discussion are most important,
        # followed by introduction
        section_weights = {
            'abstract': 40,      # Most important for summarization
            'introduction': 25,  # Second most important
            'results': 10,       # Less important for summarization
            'discussion': 25,    # Important for conclusions
            'methods': 0         # Least important for summarization
        }

    # Validate weights sum to 100
    total_weight = sum(section_weights.values())
    if total_weight != 100:
        # Normalize to 100%
        for section in section_weights:
            section_weights[section] = (section_weights[section] / total_weight) * 100

    # Get basic metadata
    title = article_data.get('title', '')
    section_headings = article_data.get('section_headings', [])

    # Extract article text and clean spacing
    article_text = article_data.get('article', '')
    article_text = re.sub(r'\s+([.,;:])', r'\1', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Extract sections using section_headings as a guide
    sections = extract_sections_from_article(article_text, section_headings)

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
    metadata_tokens = len(title.split()) + 20

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

    # Add sections with proper tags
    if sections["abstract"]:
        input_text += f"[ABSTRACT] {sections['abstract']}\n"

    if sections["introduction"]:
        input_text += f"[INTRODUCTION] {sections['introduction']}\n"

    if sections["results"]:
        input_text += f"[RESULTS] {sections['results']}\n"

    if sections["discussion"]:
        input_text += f"[DISCUSSION] {sections['discussion']}\n"

    if sections["methods"] and section_weights.get('methods', 0) > 0:
        input_text += f"[METHODS] {sections['methods']}\n"

    # Calculate final token count
    final_token_count = len(input_text.split())

    # Final check to ensure we're within token limit
    if final_token_count > max_tokens:
        # If still over limit, truncate the entire text to max_tokens
        input_text = ' '.join(input_text.split()[:max_tokens])
        final_token_count = max_tokens

    return input_text, final_token_count

def analyze_preprocessing_quality(processed_item):
    """
    Analyze the quality of the preprocessing for a given item

    Parameters:
    - processed_item: A dictionary containing the preprocessed article

    Returns:
    - quality_score: A score from 0-100 indicating preprocessing quality
    - issues: List of identified issues
    """
    issues = []
    input_text = processed_item["input_text"]

    # Check if key sections are present
    if "[ABSTRACT]" not in input_text:
        issues.append("Missing abstract section")

    # Check if either discussion or conclusion section is present
    if "[DISCUSSION]" not in input_text:
        issues.append("Missing discussion section")

    # Check token utilization
    token_count = processed_item["token_count"]
    if token_count < 512:
        issues.append(f"Low token utilization ({token_count}/1024)")
    elif token_count > 1024:
        issues.append(f"Exceeds token limit ({token_count}/1024)")

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
    token_utilization = min(token_count, 1024) / 1024
    base_score *= (0.5 + 0.5 * token_utilization)  # Scale factor for utilization

    quality_score = max(0, min(100, int(base_score)))

    return quality_score, issues
def remove_citations(text):
    """Remove citations from text"""
    # Remove in-text citations like (Author et al., 2020)
    text = re.sub(r'\([^)]*\d{4}[^)]*\)', ' ', text)

    # Remove numbered citations like [1] or [1,2,3]
    text = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', text)

    # Remove footnote indicators
    text = re.sub(r'\s*\[\w+\]\s*', ' ', text)

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_article(article_data, max_tokens=1024, section_weights=None):
    """
    Extract key sections from scientific articles using correct section ordering

    Parameters:
    - article_data: Dictionary containing article text and metadata
    - max_tokens: Maximum number of tokens in the preprocessed output
    - section_weights: Dictionary with section weights (percentages)

    Returns:
    - input_text: Preprocessed text ready for model input
    - token_count: Number of tokens in the preprocessed text
    """
    # Determine source for source-specific processing
    dataset_name = determine_dataset_source(article_data)

    # Adjust max_tokens based on source
    if dataset_name == "plos":
        max_tokens = min(max_tokens, 800)  # PLOS articles are shorter

    # Adjust default section weights based on source and EDA findings
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
    keywords = []
    if isinstance(article_data.get('keywords', []), list):
        keywords = article_data.get('keywords', [])
    keywords_str = ', '.join(keywords) if keywords else ''
    section_headings = article_data.get('section_headings', [])

    # Extract article text and clean spacing
    article_text = article_data.get('article', '')
    article_text = re.sub(r'\s+([.,;:])', r'\1', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Extract sections from the newline-split article
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

    # Add sections in the correct order based on actual structure of the articles
    if sections["abstract"]:
        input_text += f"[ABSTRACT] {sections['abstract']}\n"

    if sections["introduction"]:
        input_text += f"[INTRODUCTION] {sections['introduction']}\n"

    if sections["results"]:
        input_text += f"[RESULTS] {sections['results']}\n"

    if sections["discussion"]:
        input_text += f"[DISCUSSION] {sections['discussion']}\n"

    if sections["methods"]:
        input_text += f"[METHODS] {sections['methods']}\n"

    # Add keywords if available and space permits
    if keywords_str:
        input_text += f"[KEYWORDS] {keywords_str}\n"

    # Add section headings as additional context if space permits
    if section_headings:
        if isinstance(section_headings, list):
            headings_text = ', '.join(section_headings)
        else:
            headings_text = section_headings

        # Check if adding section headings would exceed token limit
        if len(input_text.split()) + len(headings_text.split()) <= max_tokens:
            input_text += f"[SECTIONS] {headings_text}"

    # Calculate final token count
    final_token_count = len(input_text.split())

    # Final check to ensure we're within token limit
    if final_token_count > max_tokens:
        # If still over limit, truncate the entire text to max_tokens
        input_text = ' '.join(input_text.split()[:max_tokens])
        final_token_count = max_tokens

    return input_text, final_token_count


def preprocess_dataset(dataset, max_tokens=1024, section_weights=None):
    """
    Process the entire dataset, adding dataset source tokens

    Parameters:
    - dataset: List of article dictionaries
    - max_tokens: Maximum tokens for each preprocessed article
    - section_weights: Dictionary with section weights (percentages)

    Returns:
    - processed_dataset: List of processed articles with input text and metadata
    """
    processed_dataset = []

    for item in dataset:
        # For PLOS vs eLife-specific processing
        dataset_name = determine_dataset_source(item)

        # Adjust section weights based on source if not provided
        item_section_weights = section_weights
        if item_section_weights is None:
            if dataset_name == "plos":
                item_section_weights = {
                    'abstract': 40,
                    'introduction': 20,
                    'results': 10,
                    'discussion': 25,
                    'methods': 5
                }
            else:  # eLife
                item_section_weights = {
                    'abstract': 35,
                    'introduction': 20,
                    'results': 15,
                    'discussion': 25,
                    'methods': 5
                }

        # Process the article
        input_text, token_count = preprocess_article(
            item,
            max_tokens=max_tokens,
            section_weights=item_section_weights
        )

        # Create processed item
        processed_item = {
            "input_text": input_text,
            "summary": item.get("summary", ""),
            "token_count": token_count,
            "dataset": dataset_name
        }

        processed_dataset.append(processed_item)

    return processed_dataset


def analyze_preprocessing_quality(processed_item):
    """
    Analyze the quality of the preprocessing for a given item

    Parameters:
    - processed_item: A dictionary containing the preprocessed article

    Returns:
    - quality_score: A score from 0-100 indicating preprocessing quality
    - issues: List of identified issues
    """
    issues = []
    input_text = processed_item["input_text"]

    # Check if key sections are present
    if "[ABSTRACT]" not in input_text:
        issues.append("Missing abstract section")

    if "[CONCLUSION]" not in input_text:
        issues.append("Missing conclusion section")

    # Check token utilization
    token_count = processed_item["token_count"]
    if token_count < 512:
        issues.append(f"Low token utilization ({token_count}/1024)")
    elif token_count > 1024:
        issues.append(f"Exceeds token limit ({token_count}/1024)")

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
    token_utilization = min(token_count, 1024) / 1024
    base_score *= (0.5 + 0.5 * token_utilization)  # Scale factor for utilization

    quality_score = max(0, min(100, int(base_score)))

    return quality_score, issues


# Function to process a batch of examples from a dataset
def process_batch(examples, source):
    results = {
        "input_text": [],
        "summary": [],
        "token_count": [],
        "quality_score": [],
        "issues": []
    }

    # Add source to each example
    for i in range(len(examples["article"])):
        examples_dict = {
            "article": examples["article"][i],
            "title": examples["title"][i],
            "source": source,
            "keywords": examples.get("keywords", [None])[i] if "keywords" in examples else [],
            "section_headings": examples.get("section_headings", [None])[i] if "section_headings" in examples else []
        }

        # Apply preprocessing
        input_text, token_count = preprocess_article(examples_dict, max_tokens=1024)

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

# Process a dataset with batch processing
def process_dataset(dataset, source, sample_size=None, batch_size=32):
    """
    Process a dataset with batch processing to prevent memory issues

    Parameters:
    - dataset: The HuggingFace dataset
    - source: 'plos' or 'elife'
    - sample_size: Number of examples to process (None for all)
    - batch_size: Batch size for processing

    Returns:
    - processed_data: Dictionary with processed results
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
        batch_indices = indices[i:i+batch_size]
        batch = dataset.select(batch_indices)
        batch_results = process_batch(batch, source)

        # Extend results
        for key in processed_data:
            processed_data[key].extend(batch_results[key])

    return processed_data

# Save data to disk
def save_processed_data(processed_data, output_path, format="jsonl"):
    """Save processed data to disk in the specified format"""
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
    elif format == "csv":
        pd.DataFrame(processed_data).to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def run_preprocessing():

    # Set up the T5 tokenizer to check actual token counts
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    print("Loading datasets from Hugging Face...")
    plos = load_dataset("BioLaySumm/BioLaySumm2025-PLOS")
    elife = load_dataset("BioLaySumm/BioLaySumm2025-eLife")

    print(f"PLOS dataset: {plos}")
    print(f"eLife dataset: {elife}")

    # Create output directory
    os.makedirs("processed_data", exist_ok=True)

    # Process a small sample first to verify
    sample_size = 10  # Small sample for verification

    print(f"\nProcessing {sample_size} samples from each dataset for verification...")
    plos_samples = process_dataset(plos['train'], 'plos', sample_size=sample_size)
    elife_samples = process_dataset(elife['train'], 'elife', sample_size=sample_size)

    # Save samples
    save_processed_data(plos_samples, "processed_data/plos_samples.jsonl")
    save_processed_data(elife_samples, "processed_data/elife_samples.jsonl")

    # Display some examples
    print("\nExample from PLOS:")
    print(plos_samples["input_text"][0][:500] + "...")
    print("\nExample from eLife:")
    print(elife_samples["input_text"][0][:500] + "...")

    # Ask to continue with full dataset
    process_full = input("\nDo you want to process the full datasets? (y/n): ")

    if process_full.lower() == 'y':
        # Ask for number of samples
        max_samples_input = input("Enter number of samples to process (leave empty for all): ")
        max_samples = int(max_samples_input) if max_samples_input.strip() else None

        # Process full datasets or specified number of samples
        for split in ['train', 'validation', 'test']:
            if split in plos:
                print(f"\nProcessing PLOS {split} split...")
                sample_size = min(len(plos[split]), max_samples) if max_samples else None
                plos_data = process_dataset(plos[split], 'plos', sample_size=sample_size)
                save_processed_data(plos_data, f"processed_data/plos_{split}.jsonl")
                print(f"Saved {len(plos_data['input_text'])} processed examples")

            if split in elife:
                print(f"\nProcessing eLife {split} split...")
                sample_size = min(len(elife[split]), max_samples) if max_samples else None
                elife_data = process_dataset(elife[split], 'elife', sample_size=sample_size)
                save_processed_data(elife_data, f"processed_data/elife_{split}.jsonl")
                print(f"Saved {len(elife_data['input_text'])} processed examples")

        print("\nAll processing complete!")
    else:
        print("\nSkipping full dataset processing. Sample results are available in the processed_data directory.")

if __name__ == "__main__":
    run_preprocessing()

