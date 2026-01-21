#!/usr/bin/env python3
"""Sequence Length Analysis for VIE Task"""

import json
import os
from collections import defaultdict

def count_tokens_per_file(data_file):
    """Count tokens per file in the dataset"""
    file_tokens = defaultdict(int)
    current_file = None

    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 3:
                filename = parts[-1]
                file_tokens[filename] += 1

    return file_tokens

def analyze_seq_length(data_dir, max_seq_length=512):
    """Analyze sequence length distribution"""
    op_test_file = os.path.join(data_dir, "op_test.txt")
    test_file = os.path.join(data_dir, "test.txt")

    print("=" * 80)
    print("SEQUENCE LENGTH ANALYSIS")
    print("=" * 80)

    for name, filepath in [("op_test", op_test_file), ("test", test_file)]:
        if not os.path.exists(filepath):
            continue

        file_tokens = count_tokens_per_file(filepath)

        total_files = len(file_tokens)
        exceeded_files = [f for f, count in file_tokens.items() if count > max_seq_length - 2]

        print(f"\n[{name.upper()}]")
        print(f"  Total files: {total_files}")
        print(f"  Files exceeding {max_seq_length} tokens: {len(exceeded_files)} ({len(exceeded_files)/total_files*100:.1f}%)")

        # Token distribution
        token_counts = list(file_tokens.values())
        token_counts.sort()

        print(f"\n  Token count distribution:")
        print(f"    Min: {min(token_counts)}")
        print(f"    Max: {max(token_counts)}")
        print(f"    Median: {token_counts[len(token_counts)//2]}")
        print(f"    Mean: {sum(token_counts)/len(token_counts):.1f}")

        # Files with most tokens
        sorted_files = sorted(file_tokens.items(), key=lambda x: -x[1])
        print(f"\n  Top 10 longest files:")
        for filename, count in sorted_files[:10]:
            print(f"    {filename}: {count} tokens")

        # Return exceeded files for further analysis
        if name == "op_test":
            return exceeded_files, file_tokens

    return [], {}

def cross_reference_errors(exceeded_files, error_file):
    """Cross-reference exceeded files with error analysis"""
    with open(error_file, "r") as f:
        error_data = json.load(f)

    print("\n" + "=" * 80)
    print("CROSS-REFERENCE: Sequence Length vs Errors")
    print("=" * 80)

    exceeded_set = set(exceeded_files)

    for entity, data in error_data.items():
        errors = data.get("errors", [])

        # Errors in exceeded files
        errors_in_exceeded = [e for e in errors if e["filename"] in exceeded_set]
        empty_in_exceeded = [e for e in errors_in_exceeded if not e["pred"].strip()]

        print(f"\n[{entity.upper()}]")
        print(f"  Total errors: {len(errors)}")
        print(f"  Errors in exceeded files: {len(errors_in_exceeded)}")
        print(f"  Empty predictions in exceeded files: {len(empty_in_exceeded)}")

        if empty_in_exceeded:
            print(f"\n  Empty predictions in exceeded files (sample):")
            for e in empty_in_exceeded[:3]:
                print(f"    - {e['filename']}: gold='{e['gold'][:50]}...'")

def analyze_total_position(data_file):
    """Analyze where TOTAL labels appear in documents"""
    print("\n" + "=" * 80)
    print("TOTAL LABEL POSITION ANALYSIS")
    print("=" * 80)

    # For training data with labels
    file_data = defaultdict(lambda: {"total_count": 0, "tokens": [], "total_positions": []})

    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_file = None
    current_position = 0

    for line in lines:
        line = line.strip()
        if not line:
            current_position = 0
            continue

        parts = line.split("\t")
        if len(parts) >= 2:
            token, label = parts[0], parts[1]
            # Assuming filename might be in the line
            if len(parts) >= 3:
                filename = parts[-1]
                if filename != current_file:
                    current_file = filename
                    current_position = 0

                current_position += 1
                file_data[filename]["total_count"] += 1

                if "TOTAL" in label:
                    file_data[filename]["total_positions"].append(current_position)

    # Analyze TOTAL positions
    all_total_positions = []
    files_with_total_after_512 = []

    for filename, data in file_data.items():
        for pos in data["total_positions"]:
            all_total_positions.append(pos)
            if pos > 510:
                files_with_total_after_512.append((filename, pos))

    if all_total_positions:
        print(f"\n  TOTAL label position statistics:")
        print(f"    Total occurrences: {len(all_total_positions)}")
        print(f"    Min position: {min(all_total_positions)}")
        print(f"    Max position: {max(all_total_positions)}")
        print(f"    Mean position: {sum(all_total_positions)/len(all_total_positions):.1f}")
        print(f"    Labels after position 510: {len(files_with_total_after_512)}")

def main():
    data_dir = "../data"
    error_file = "./output/error_analysis.json"

    exceeded_files, file_tokens = analyze_seq_length(data_dir)

    if os.path.exists(error_file):
        cross_reference_errors(exceeded_files, error_file)

    # Analyze training data for TOTAL positions
    train_file = os.path.join(data_dir, "train.txt")
    if os.path.exists(train_file):
        analyze_total_position(train_file)

if __name__ == "__main__":
    main()
