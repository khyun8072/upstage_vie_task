#!/usr/bin/env python3
"""Deep Analysis of TOTAL Prediction Failures"""

import os
import json
import csv
from collections import defaultdict, Counter

def analyze_training_total(train_file):
    """Analyze TOTAL labels in training data"""
    print("=" * 80)
    print("TRAINING DATA: TOTAL LABEL ANALYSIS")
    print("=" * 80)

    total_words = []
    total_contexts = []  # Words before TOTAL

    with open(train_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    prev_words = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            prev_words = []
            continue

        parts = line.split("\t")
        if len(parts) >= 2:
            word, label = parts[0], parts[1]

            if "TOTAL" in label:
                total_words.append(word)
                if prev_words:
                    total_contexts.append(" ".join(prev_words[-3:]))

            prev_words.append(word)

    print(f"\n  Total S-TOTAL labels in training: {len(total_words)}")

    # Most common TOTAL values
    word_counts = Counter(total_words)
    print(f"\n  Most common TOTAL tokens:")
    for word, count in word_counts.most_common(20):
        print(f"    '{word}': {count}")

    # Context patterns
    context_counts = Counter(total_contexts)
    print(f"\n  Most common contexts before TOTAL:")
    for ctx, count in context_counts.most_common(10):
        print(f"    '{ctx}': {count}")

def analyze_prediction_patterns(output_file, gt_dir):
    """Analyze what the model predicts instead of TOTAL"""
    print("\n" + "=" * 80)
    print("PREDICTION PATTERN ANALYSIS")
    print("=" * 80)

    # Load ground truth
    gt_totals = {}
    for filepath in os.listdir(gt_dir):
        if filepath.endswith(".txt"):
            filename = os.path.splitext(filepath)[0]
            with open(os.path.join(gt_dir, filepath), "r") as f:
                data = json.load(f)
            gt_totals[filename] = data.get("total", "")

    # Load predictions
    predictions = defaultdict(lambda: {"all_labels": [], "total_pred": []})

    with open(output_file, "r", encoding="utf-8") as f:
        for line in csv.reader(f):
            if len(line) == 3:
                text, label, filename = line
                predictions[filename]["all_labels"].append((text, label))
                if label == "S-TOTAL":
                    predictions[filename]["total_pred"].append(text)

    # Find files with empty TOTAL predictions
    empty_total_files = []
    for filename in gt_totals:
        if filename in predictions:
            if not predictions[filename]["total_pred"]:
                empty_total_files.append(filename)

    print(f"\n  Files with GT TOTAL but empty prediction: {len(empty_total_files)}")

    # Analyze what labels appear in files with empty TOTAL
    label_in_empty_total = Counter()
    for filename in empty_total_files:
        for text, label in predictions[filename]["all_labels"]:
            label_in_empty_total[label] += 1

    print(f"\n  Label distribution in files with empty TOTAL prediction:")
    total_labels = sum(label_in_empty_total.values())
    for label, count in label_in_empty_total.most_common():
        print(f"    {label}: {count} ({count/total_labels*100:.1f}%)")

    # Check if TOTAL value appears in text but labeled as O
    print(f"\n  Checking if TOTAL values appear but labeled as O:")
    found_as_o = 0
    for filename in empty_total_files[:20]:  # Sample
        gt_total = gt_totals.get(filename, "")
        all_texts = [t for t, l in predictions[filename]["all_labels"]]
        all_text_str = " ".join(all_texts).lower()

        # Check if total value is in the text
        gt_total_normalized = gt_total.replace(" ", "").lower()
        if gt_total_normalized in all_text_str.replace(" ", ""):
            found_as_o += 1
            print(f"    {filename}: GT='{gt_total}' found in text but not labeled")

    print(f"\n  TOTAL values found in text but labeled as O: {found_as_o}/20 samples")

def analyze_confusion(output_file, op_test_file):
    """Analyze what tokens near TOTAL are labeled as"""
    print("\n" + "=" * 80)
    print("CONFUSION ANALYSIS: Tokens near TOTAL keyword")
    print("=" * 80)

    # Load op_test to find TOTAL keyword positions
    file_tokens = defaultdict(list)
    with open(op_test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                word, _, filename = parts[0], parts[1], parts[-1]
                file_tokens[filename].append(word)

    # Load predictions
    file_preds = defaultdict(list)
    with open(output_file, "r", encoding="utf-8") as f:
        for line in csv.reader(f):
            if len(line) == 3:
                text, label, filename = line
                file_preds[filename].append((text, label))

    # Find "TOTAL" keyword and check nearby labels
    total_keyword_contexts = []
    for filename, tokens in file_tokens.items():
        if filename not in file_preds:
            continue

        preds = file_preds[filename]
        for i, token in enumerate(tokens):
            if "total" in token.lower() and i < len(preds):
                # Get context around TOTAL keyword
                start = max(0, i-2)
                end = min(len(preds), i+5)
                context = preds[start:end]
                total_keyword_contexts.append({
                    "filename": filename,
                    "keyword_idx": i - start,
                    "context": context
                })

    print(f"\n  Found {len(total_keyword_contexts)} 'TOTAL' keyword occurrences")

    # Analyze labels around TOTAL keyword
    label_at_total = Counter()
    label_after_total = Counter()

    for ctx in total_keyword_contexts:
        keyword_idx = ctx["keyword_idx"]
        context = ctx["context"]

        if keyword_idx < len(context):
            label_at_total[context[keyword_idx][1]] += 1

        # Labels after TOTAL keyword (likely the amount)
        for i in range(keyword_idx + 1, len(context)):
            label_after_total[context[i][1]] += 1

    print(f"\n  Labels AT 'TOTAL' keyword:")
    for label, count in label_at_total.most_common():
        print(f"    {label}: {count}")

    print(f"\n  Labels AFTER 'TOTAL' keyword (next 2-3 tokens):")
    for label, count in label_after_total.most_common():
        print(f"    {label}: {count}")

def main():
    train_file = "../data/train.txt"
    output_file = "./output/output.csv"
    gt_dir = "../data/test/entities"
    op_test_file = "../data/op_test.txt"

    analyze_training_total(train_file)
    analyze_prediction_patterns(output_file, gt_dir)
    analyze_confusion(output_file, op_test_file)

if __name__ == "__main__":
    main()
