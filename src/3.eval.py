#!/usr/bin/env python3
"""
BioLaySumm Evaluation Script

This script evaluates the quality of generated lay summaries using multiple metrics:
1. Relevance: ROUGE, BERTScore, METEOR, BLEU
2. Readability: Flesch-Kincaid, Dale-Chall, Coleman-Liau
3. Factuality: LENS, AlignScore, SummaC

Usage:
  python src/3.evaluate.py --predictions_dir /data/predictions --output_dir /data/evaluation
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
import torch
import nltk
import evaluate

# Import specialized metrics
import textstat
from lens import download_model, LENS
from summac.model_summac import SummaCConv
from alignscore import AlignScore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate generated summaries with multiple metrics')

    # Input/output paths
    parser.add_argument('--predictions_dir', type=str, default='./predictions',
                        help='Directory containing prediction CSV files')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Directory to save evaluation results')

    # Dataset options
    parser.add_argument('--dataset', type=str, choices=['plos', 'elife', 'both'], default='both',
                        help='Which dataset to evaluate')
    parser.add_argument('--split', type=str, default='test',
                        help='Split name used in prediction filenames (e.g., "test" for plos_test_predictions.csv)')

    # Column names
    parser.add_argument('--input_col', type=str, default='input_text',
                        help='Column name for source articles')
    parser.add_argument('--reference_col', type=str, default='summary',
                        help='Column name for reference summaries')
    parser.add_argument('--prediction_col', type=str, default='predicted_summary',
                        help='Column name for generated summaries')

    # Metrics selection
    parser.add_argument('--relevance_metrics', action='store_true', default=True,
                        help='Calculate relevance metrics (ROUGE, BERTScore, METEOR, BLEU)')
    parser.add_argument('--readability_metrics', action='store_true', default=True,
                        help='Calculate readability metrics (FK, DC, CLI)')
    parser.add_argument('--factuality_metrics', action='store_true', default=True,
                        help='Calculate factuality metrics (LENS, AlignScore, SummaC)')
    parser.add_argument('--skip_lens', action='store_true',
                        help='Skip LENS evaluation (can be slow)')
    parser.add_argument('--skip_alignscore', action='store_true',
                        help='Skip AlignScore evaluation (can be slow)')
    parser.add_argument('--skip_summac', action='store_true',
                        help='Skip SummaC evaluation (can be slow)')

    # Sample size
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to evaluate (useful for testing)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run quick test with minimal samples')

    # Technical settings
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for neural metrics (auto if None)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for neural metrics')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # AlignScore specifics
    parser.add_argument('--alignscore_path', type=str, default='./models/AlignScore/AlignScore-base.ckpt',
                        help='Path to AlignScore checkpoint')

    return parser.parse_args()


class Evaluator:
    """
    Evaluator class for calculating multiple metrics on generated summaries.

    Metrics include:
    - Relevance: ROUGE, BERTScore, METEOR, BLEU
    - Readability: Flesch-Kincaid, Dale-Chall, Coleman-Liau
    - Factuality: LENS, AlignScore, SummaC
    """

    def __init__(self,
                 references: List[str],
                 predictions: List[str],
                 articles: List[str],
                 device: Optional[str] = None,
                 batch_size: int = 16,
                 alignscore_path: str = './models/AlignScore/AlignScore-base.ckpt'):
        """
        Initialize evaluator with references, predictions, and source articles.

        Args:
            references: List of reference summaries
            predictions: List of generated summaries
            articles: List of source articles
            device: Device to use for neural models
            batch_size: Batch size for processing
            alignscore_path: Path to AlignScore checkpoint
        """
        self.references = references
        self.predictions = predictions
        self.articles = articles
        self.batch_size = batch_size
        self.alignscore_path = alignscore_path

        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        logger.info(f"Using device: {self.device}")
        self.results = {}

    def evaluate(self, relevance=True, readability=True, factuality=True,
                 skip_lens=False, skip_alignscore=False, skip_summac=False):
        """
        Evaluate generated summaries using multiple metrics.

        Args:
            relevance: Whether to calculate relevance metrics
            readability: Whether to calculate readability metrics
            factuality: Whether to calculate factuality metrics
            skip_lens: Whether to skip LENS evaluation
            skip_alignscore: Whether to skip AlignScore evaluation
            skip_summac: Whether to skip SummaC evaluation

        Returns:
            Dictionary with calculated metrics
        """
        # Step one: relevance metrics
        if relevance:
            logger.info("Calculating relevance metrics...")

            logger.info("Calculating ROUGE scores...")
            self.add_rouge_score_to_result()

            logger.info("Calculating BERTScore...")
            self.add_bert_score_to_result()

            logger.info("Calculating METEOR...")
            self.add_meteor_score_to_result()

            logger.info("Calculating BLEU...")
            self.add_bleu_score_to_result()

        # Step 2: readability metrics
        if readability:
            logger.info("Calculating readability metrics...")
            self.add_most_readability_scores_to_result()

        # Step 3: factuality metrics
        if factuality:
            if not skip_lens:
                logger.info("Calculating LENS score...")
                self.add_lens_score_to_result()

            if not skip_alignscore:
                logger.info("Calculating AlignScore...")
                self.add_alignscore_to_result()

            if not skip_summac:
                logger.info("Calculating SummaC score...")
                self.add_summac_score_to_result()

        return self.results

    def add_rouge_score_to_result(self):
        """
        Calculate and store ROUGE scores.
        ROUGE compares n-grams in reference and prediction, emphasizing recall over precision.
        """
        rouge1, rouge2, rougeL = self.calculate_rouge_score()
        self.results["rouge1"] = rouge1
        self.results["rouge2"] = rouge2
        self.results["rougeL"] = rougeL

        logger.info(f"ROUGE-1: {rouge1:.4f}")
        logger.info(f"ROUGE-2: {rouge2:.4f}")
        logger.info(f"ROUGE-L: {rougeL:.4f}")

    def calculate_rouge_score(self) -> List[float]:
        """Calculate ROUGE scores."""
        rouge = evaluate.load("rouge")
        scores = rouge.compute(predictions=self.predictions, references=self.references)
        return [scores["rouge1"], scores["rouge2"], scores["rougeL"]]

    def add_bert_score_to_result(self):
        """
        Calculate and store BERTScore.
        BERTScore uses contextual embeddings to match words by cosine similarity.
        """
        self.results["bertscore"] = self.calculate_bert_score()
        logger.info(f"BERTScore: {self.results['bertscore']:.4f}")

    def calculate_bert_score(self) -> float:
        """Calculate BERTScore."""
        bertscore = evaluate.load("bertscore")
        scores = bertscore.compute(predictions=self.predictions, references=self.references, lang="en")
        return np.mean(scores["f1"])

    def add_meteor_score_to_result(self):
        """
        Calculate and store METEOR score.
        METEOR matches unigrams between reference and candidate, balancing recall and precision.
        """
        self.results["meteor"] = self.calculate_meteor_score()
        logger.info(f"METEOR: {self.results['meteor']:.4f}")

    def calculate_meteor_score(self) -> float:
        """Calculate METEOR score."""
        meteor = evaluate.load("meteor")
        return meteor.compute(predictions=self.predictions, references=self.references)["meteor"]

    def add_bleu_score_to_result(self):
        """
        Calculate and store BLEU score.
        BLEU averages per-sentence scores of shared n-grams, emphasizing precision over recall.
        """
        self.results["bleu"] = self.calculate_bleu_score()
        logger.info(f"BLEU: {self.results['bleu']:.4f}")

    def calculate_bleu_score(self) -> float:
        """Calculate BLEU score."""
        bleu = evaluate.load("bleu")
        return bleu.compute(predictions=self.predictions, references=self.references)["bleu"]

    def add_most_readability_scores_to_result(self):
        """Calculate and store readability scores."""
        fkgl, dcrs, cli = self.calculate_most_readability_scores()
        self.results["flesch_kincaid"] = fkgl
        self.results["dale_chall"] = dcrs
        self.results["coleman_liau"] = cli

        logger.info(f"Flesch-Kincaid Grade Level: {fkgl:.2f}")
        logger.info(f"Dale-Chall Readability Score: {dcrs:.2f}")
        logger.info(f"Coleman-Liau Index: {cli:.2f}")

    def calculate_most_readability_scores(self) -> List[float]:
        """
        Calculate readability scores:
        - Flesch-Kincaid Grade Level: estimates US grade level needed to understand the text
        - Dale-Chall: based on word familiarity
        - Coleman-Liau: based on characters per word and words per sentence
        """
        fkcg_scores = []
        dcrs_scores = []
        cli_scores = []
        for prediction in self.predictions:
            fkcg_scores.append(textstat.flesch_kincaid_grade(prediction))
            cli_scores.append(textstat.coleman_liau_index(prediction))
            dcrs_scores.append(textstat.dale_chall_readability_score(prediction))
        return [np.mean(fkcg_scores), np.mean(dcrs_scores), np.mean(cli_scores)]

    def add_lens_score_to_result(self):
        """
        Calculate and store LENS score.
        LENS evaluates the quality of simplification by comparing source, simplified, and reference texts.
        """
        lens_score, _ = self.calculate_lens_score()
        self.results["lens"] = lens_score
        logger.info(f"LENS score: {lens_score:.2f}")

    def calculate_lens_score(self) -> Tuple[float, float]:
        """
        Calculate LENS score.
        Returns the rescaled score (0-100) for better interpretability.
        """
        lens_path = download_model("davidheineman/lens")
        lens = LENS(lens_path, rescale=True)

        complex_texts = self.articles
        simple_texts = self.predictions
        references = self.references

        device = [0] if self.device == "cuda:0" else None

        return lens.score(complex_texts, simple_texts, references,
                          batch_size=self.batch_size, devices=device)

    def add_alignscore_to_result(self):
        """
        Calculate and store AlignScore.
        AlignScore measures factual consistency between generated and reference summaries.
        """
        self.results["alignscore"] = self.calculate_alignscore()
        logger.info(f"AlignScore: {self.results['alignscore']:.4f}")

    def calculate_alignscore(self) -> float:
        """Calculate AlignScore."""
        alignscorer = AlignScore(
            model="roberta-base",
            batch_size=self.batch_size,
            device=self.device,
            ckpt_path=self.alignscore_path,
            evaluation_mode="nli_sp"
        )
        return np.mean(alignscorer.score(contexts=self.articles, claims=self.predictions))

    def add_summac_score_to_result(self):
        """
        Calculate and store SummaC score.
        SummaC measures the factual consistency of summaries using NLI models.
        """
        self.results["summac"] = self.calculate_summac_score()
        logger.info(f"SummaC score: {self.results['summac']:.4f}")

    def calculate_summac_score(self) -> float:
        """Calculate SummaC score."""
        model_conv = SummaCConv(
            models=["vitc"],
            bins="percentile",
            granularity="sentence",
            nli_labels="e",
            device=self.device,
            start_file="default",
            agg="mean"
        )
        return np.mean(model_conv.score(self.articles, self.predictions)["scores"])


def load_test_data() -> Tuple[List[str], List[str], List[str]]:
    """
    Load minimal test data for quick testing.

    Returns:
        Tuple of (articles, references, predictions)
    """
    articles = [
        ("Kidney function depends on the nephron, which comprises a blood filter, a tubule that is "
         "subdivided into functionally distinct segments, and a collecting duct. How these regions "
         "arise during development is poorly understood. The zebrafish pronephros consists of two "
         "linear nephrons that develop from the intermediate mesoderm along the length of the trunk. "
         "Here we show that, contrary to current dogma, these nephrons possess multiple proximal and "
         "distal tubule domains that resemble the organization of the mammalian nephron."),

        ("White-nose syndrome is one of the most lethal wildlife diseases, killing over 5 million "
         "North American bats since it was first reported in 2006. The causal agent of the disease is "
         "a psychrophilic filamentous fungus, Pseudogymnoascus destructans. The fungus is widely "
         "distributed in North America and Europe and has recently been found in some parts of Asia, "
         "but interestingly, no mass mortality is observed in European or Asian bats.")
    ]

    references = [
        ("In the kidney, structures known as nephrons are responsible for collecting metabolic waste. "
         "Nephrons are composed of a blood filter (glomerulus) followed by a series of specialized "
         "tubule regions, or segments, which recover solutes such as salts, and finally terminate "
         "with a collecting duct. The genetic mechanisms that establish nephron segmentation in "
         "mammals have been a challenge to study because of the kidney's complex organogenesis."),

        ("Many species of bats in North America have been severely impacted by a fungal disease, "
         "white-nose syndrome, that has killed over 5 million bats since it was first identified in "
         "2006. The fungus is believed to have been introduced into a cave in New York where bats "
         "hibernate, and has now spread to 29 states and 4 Canadian provinces.")
    ]

    predictions = [
        ("In the kidney, tiny units called nephrons remove waste from the blood. Each nephron has a "
         "filter (called the glomerulus), followed by different tube sections that reabsorb useful "
         "substances like salts, and ends with a collecting duct. Studying how these sections form "
         "in mammals is difficult because kidney development is very complex."),

        ("Many types of bats in North America have been badly affected by a disease called white-nose "
         "syndrome, caused by a fungus. Since it was first found in 2006, the disease has killed over "
         "5 million bats. It likely started in a New York cave where bats hibernate and has now "
         "spread to 29 U.S. states and 4 provinces in Canada.")
    ]

    return articles, references, predictions


def load_data_from_csv(file_path: str, input_col: str, reference_col: str,
                       prediction_col: str, max_samples: Optional[int] = None) -> Tuple[
    List[str], List[str], List[str]]:
    """
    Load data from CSV file.

    Args:
        file_path: Path to CSV file
        input_col: Column name for source articles
        reference_col: Column name for reference summaries
        prediction_col: Column name for generated summaries
        max_samples: Maximum number of samples to load

    Returns:
        Tuple of (articles, references, predictions)
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} samples from {file_path}")

        # Verify columns exist
        for col in [input_col, reference_col, prediction_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {file_path}")

        # Sample if needed
        if max_samples and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} examples for evaluation")

        articles = df[input_col].tolist()
        references = df[reference_col].tolist()
        predictions = df[prediction_col].tolist()

        return articles, references, predictions

    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return [], [], []


def evaluate_dataset(args: argparse.Namespace, dataset_name: str) -> Dict[str, Any]:
    """
    Evaluate a single dataset.

    Args:
        args: Command line arguments
        dataset_name: Name of the dataset ('plos' or 'elife')

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"\n{'=' * 20} Evaluating {dataset_name.upper()} dataset {'=' * 20}\n")

    # Determine prediction file path
    prediction_file = os.path.join(args.predictions_dir, f"{dataset_name}_predictions.csv")
    if not os.path.exists(prediction_file):
        logger.error(f"Prediction file not found: {prediction_file}")
        return {}

    # Load data
    articles, references, predictions = load_data_from_csv(
        prediction_file,
        args.input_col,
        args.reference_col,
        args.prediction_col,
        args.max_samples
    )

    if not articles:
        logger.error(f"No data loaded for {dataset_name}")
        return {}

    # Initialize evaluator
    evaluator = Evaluator(
        references=references,
        predictions=predictions,
        articles=articles,
        device=args.device,
        batch_size=args.batch_size,
        alignscore_path=args.alignscore_path
    )

    # Run evaluation
    results = evaluator.evaluate(
        relevance=args.relevance_metrics,
        readability=args.readability_metrics,
        factuality=args.factuality_metrics,
        skip_lens=args.skip_lens,
        skip_alignscore=args.skip_alignscore,
        skip_summac=args.skip_summac
    )

    # Add dataset name to results
    results['dataset'] = dataset_name

    # Print summary
    logger.info(f"\n{'-' * 20} Summary for {dataset_name.upper()} {'-' * 20}")
    for metric, value in results.items():
        if metric != 'dataset':
            logger.info(f"{metric}: {value:.4f}")

    return results


def main(args):
    """Main function."""
    # Set up NLTK resources
    nltk.download("punkt", quiet=True)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    if args.test_mode:
        logger.info("Running in test mode with minimal data")
        articles, references, predictions = load_test_data()

        evaluator = Evaluator(
            references=references,
            predictions=predictions,
            articles=articles,
            device=args.device,
            batch_size=args.batch_size,
            alignscore_path=args.alignscore_path
        )

        results = evaluator.evaluate(
            relevance=args.relevance_metrics,
            readability=args.readability_metrics,
            factuality=args.factuality_metrics,
            skip_lens=args.skip_lens,
            skip_alignscore=args.skip_alignscore,
            skip_summac=args.skip_summac
        )

        results['dataset'] = 'test'
        all_results.append(results)

    else:
        # Evaluate datasets as requested
        if args.dataset in ['plos', 'both']:
            plos_results = evaluate_dataset(args, 'plos')
            if plos_results:
                all_results.append(plos_results)

        if args.dataset in ['elife', 'both']:
            elife_results = evaluate_dataset(args, 'elife')
            if elife_results:
                all_results.append(elife_results)

    # Save results to JSON
    if all_results:
        output_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        logger.info(f"Results saved to {output_file}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
