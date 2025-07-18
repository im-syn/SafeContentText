#!/usr/bin/env python3
"""
bad_text_detector.py

A professional text “bad-content” detector using a zero‑shot AI classifier.
Detects categories like profanity, hate, gore, etc., in single texts, files,
or entire directories. Outputs flagged items and can save full JSON results.
"""

import os
import sys
import argparse
import json
import logging
from typing import List, Dict, Any

from transformers import pipeline, Pipeline

# Categories we want to detect
DEFAULT_LABELS = [
    "profanity",
    "hate speech",
    "graphic violence",
    "self-harm",
    "sexual content",
    "insult",
    "terrorism"
]

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level
    )

def load_texts_from_dir(directory: str) -> Dict[str, str]:
    """
    Walk a directory and load all .txt files. Returns a map path->content.
    """
    texts: Dict[str, str] = {}
    for root, _, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(".txt"):
                path = os.path.join(root, name)
                try:
                    with open(path, encoding="utf-8") as f:
                        texts[path] = f.read()
                except Exception as e:
                    logging.warning(f"Could not read {path}: {e}")
    return texts

def detect_bad_content(
    classifier: Pipeline,
    texts: List[str],
    labels: List[str]
) -> List[Dict[str, Any]]:
    """
    Runs zero-shot classification. Returns a list of dicts:
    { "text": ..., "labels": [...], "scores": [...] }
    """
    results = classifier(texts, candidate_labels=labels, multi_label=True)
    # Ensure consistent output format
    if isinstance(results, dict):
        # single input
        results = [results]
    return results

def main() -> None:
    p = argparse.ArgumentParser(description="AI‑powered bad‑content detector")
    p.add_argument("--text", "-t", help="Text string to analyze", type=str)
    p.add_argument("--file", "-f", help="Path to a .txt file to analyze", type=str)
    p.add_argument("--dir", "-d", help="Path to a directory of .txt files", type=str)
    p.add_argument(
        "--labels", "-l",
        help="Comma-separated list of labels to detect",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=DEFAULT_LABELS
    )
    p.add_argument(
        "--threshold", "-T",
        help="Score threshold (0.0–1.0) above which a label is flagged",
        type=float,
        default=0.5
    )
    p.add_argument(
        "--output", "-o",
        help="Path to save full JSON results (optional)",
        type=str
    )
    p.add_argument(
        "--verbose", "-v",
        help="Enable debug logging",
        action="store_true"
    )
    args = p.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    configure_logging(log_level)

    # Prepare inputs
    inputs: Dict[str, str] = {}
    if args.text:
        inputs["<input>"] = args.text
    if args.file:
        if not os.path.isfile(args.file):
            logging.error(f"File not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            inputs[args.file] = f.read()
    if args.dir:
        if not os.path.isdir(args.dir):
            logging.error(f"Directory not found: {args.dir}")
            sys.exit(1)
        inputs.update(load_texts_from_dir(args.dir))

    if not inputs:
        p.error("Please specify --text, --file, or --dir with at least one input")

    logging.info(f"Loading zero‑shot classifier model...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    logging.info(f"Detecting labels {args.labels} with threshold {args.threshold}")
    results = detect_bad_content(classifier, list(inputs.values()), args.labels)

    flagged: Dict[str, Dict[str, float]] = {}
    safe: Dict[str, Dict[str, float]] = {}

    # Process and display
    for (source, text), res in zip(inputs.items(), results):
        label_scores = dict(zip(res["labels"], res["scores"]))
        high = {lbl: sc for lbl, sc in label_scores.items() if sc >= args.threshold}
        if high:
            flagged[source] = high
            print(f"[FLAGGED] {source}")
            for lbl, sc in high.items():
                print(f"   - {lbl}: {sc:.2f}")
        else:
            safe[source] = label_scores
            print(f"[ SAFE ] {source}")

    # Optionally save JSON
    if args.output:
        out_data = {
            "threshold": args.threshold,
            "labels": args.labels,
            "results": [
                {"source": src, "scores": dict(zip(res["labels"], res["scores"]))}
                for src, res in zip(inputs.keys(), results)
            ]
        }
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Results written to {args.output}")
        except Exception as e:
            logging.error(f"Could not write output file: {e}")

if __name__ == "__main__":
    main()
