import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import json
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

nltk.download('punkt_tab')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


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

     # If 'and' in section headings, combine the two adjacent headings
    headings = []
    i = 0
    while i < len(cleaned_headings):
      if cleaned_headings[i] != 'and' and cleaned_headings[i] != '&':
        headings.append(cleaned_headings[i])
        i += 1
      else:
        first = headings[-1]
        if i + 1 < len(cleaned_headings):
          headings[-1] = first + ' ' + cleaned_headings[i] + ' ' + cleaned_headings[i + 1]
          cleaned_headings.pop(i + 1)
          i += 1
        else:
          headings[-1] = first + ' ' + cleaned_headings[i]
          break

    controled_headings = []

    for section in headings:
      section = section.lower()  # Convert to lowercase for consistency
      if 'abstract' in section:
        controled_headings.append('abstract')
      elif 'introduction' in section:
        controled_headings.append('introduction')
      elif 'discussion' in section:
        controled_headings.append('discussion')
      elif 'conclusion' in section:
        controled_headings.append('discussion')
      elif 'method' in section:
        controled_headings.append('methods')
      elif 'result' in section:
        controled_headings.append('results')
      else:
        controled_headings.append('other')

    # Create mapping of extracted sections based on position in text
    # Assuming sections are in the same order in the text as in section_headings
    sections = defaultdict(str)

    zipped = zip(controled_headings, sections_content)
    for heading, content in zipped:
        sections[heading] = content

    return sections

def preprocess_article(article_data, max_tokens=1024, section_weights=None):
    """
    Preprocesses scientific article text with section weighting and citation removal.

    Parameters:
    - article_data: Dictionary with article metadata and full text.
    - max_tokens: Max number of tokens allowed in final text.
    - section_weights: Optional dict assigning token percentages to each section.

    Returns:
    - input_text: Structured string for model input.
    - token_count: Number of tokens in the final string.
    """
    def remove_citations(text):
        #text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)
        #text = re.sub(r'\(([^()]*\d{4}[^()]*)\)', '', text)
        # now removing anything in parentheses
        text = re.sub(r"\(\s*.*?\s*\)|\[\s*.*?\s*\]", "", text)
        return text

    def clean_text(text):
        text = text.replace('\n', '<<NEWLINE>>')
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('<<NEWLINE>>', '\n')
        return text

    dataset_name = article_data.get('source', '').lower()

    # Set default weights if not provided
    if section_weights is None:
        if dataset_name == "plos":
            section_weights = {
                'abstract': 50,
                'introduction': 20,
                'results': 7,
                'discussion': 18,
                'methods': 5
            }
        else:  # eLife
            section_weights = {
                'abstract': 40,
                'introduction': 32,
                'results': 10,
                'discussion': 13,
                'methods': 5
            }

    # Normalize weights if they don't sum to 100
    total_weight = sum(section_weights.values())
    if total_weight != 100:
        for section in section_weights:
            section_weights[section] = (section_weights[section] / total_weight) * 100

    # Clean article and extract metadata
    raw_text = clean_text(article_data.get('article', ''))
    raw_text = remove_citations(raw_text)
    title = clean_text(article_data.get('title', ''))
    title = remove_citations(title)

    # Extract and clean sections
    # Extract and clean sections using section headings
    # sections is a default dict with actual headings, 'and' conjoined
    sections = extract_sections_from_article(raw_text, article_data.get("section_headings", []))

    # Compute token allocations per section
    available_tokens = max_tokens - len(title.split()) - 10  # Reserve some for metadata
    section_tokens = {
        sec: min(len(word_tokenize(sections[sec])), int((section_weights.get(sec, 0) / 100) * available_tokens))
        for sec in sections
    }

    # Adjust if allocations don't add up to available tokens
    total_allocated = sum(section_tokens.values())
    if total_allocated < available_tokens:
        # Distribute remaining tokens proportionally
        remaining = available_tokens - total_allocated
        for section in section_weights:
            if len(word_tokenize(sections[section])) > int((section_weights.get(section, 0) / 100) * available_tokens):
                if section_weights[section] > 0:
                    section_tokens[section] += int(remaining * section_weights[section] / 100)

    # Build input string
    input_parts = [f"<{dataset_name}> [TITLE] {title}"]
    input_parts.append(f"[KEYWORDS] {' '.join(article_data.get('keywords', ''))}")
    for sec in ['abstract', 'introduction', 'results', 'discussion', 'methods']:
        if sections[sec]:

            sentences = sent_tokenize(sections[sec])
            vectorizer = TfidfVectorizer()
            vectorizer.fit(sentences)
            sentence_vectors = vectorizer.transform(sentences)
            sentence_scores = [(i, sentence_vectors[i].sum()) for i in range(len(sentences))]
            top_sentences_indices = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
            top_sentences = [sentences[i] for i, score in top_sentences_indices]

            words = word_tokenize(" ".join(top_sentences))[:section_tokens[sec]]
            input_parts.append(f"[{sec.upper()}] {' '.join(words)}")

    input_text = '\n'.join(input_parts)
    token_count = len(word_tokenize(input_text))

    # Truncate final text if needed
    # if token_count > max_tokens:
    #     input_text = ' '.join(word_tokenize(input_text)[:max_tokens])
    #     token_count = max_tokens

    return input_text, token_count

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
            "keywords": examples["keywords"][i],
            "section_headings": examples["section_headings"][i]
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
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(len(processed_data["input_text"])):
                record = {
                    "input_text": processed_data["input_text"][i],
                    "summary": processed_data["summary"][i],
                    "token_count": processed_data["token_count"][i],
                    "quality_score": processed_data["quality_score"][i],
                    "issues": processed_data["issues"][i]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    elif format == "csv":
        pd.DataFrame(processed_data).to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def run_preprocessing():

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
