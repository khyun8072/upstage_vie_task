#!/usr/bin/env python3
"""Tokenization Analysis - Check actual token counts after subword tokenization"""

import os
from collections import defaultdict
from transformers import BertTokenizer

def analyze_tokenization(data_dir, max_seq_length=512):
    """Analyze actual token counts after BertTokenizer subword tokenization"""

    tokenizer = BertTokenizer.from_pretrained('microsoft/layoutlm-base-uncased', do_lower_case=True)

    print("=" * 80)
    print("SUBWORD TOKENIZATION ANALYSIS")
    print("=" * 80)

    for name in ["op_test", "test"]:
        filepath = os.path.join(data_dir, f"{name}.txt")
        if not os.path.exists(filepath):
            continue

        file_tokens = defaultdict(list)
        current_file = None

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= 1:
                    word = parts[0]
                    filename = parts[-1] if len(parts) >= 3 else "unknown"

                    # Count subword tokens
                    word_tokens = tokenizer.tokenize(word)
                    file_tokens[filename].extend(word_tokens)

        print(f"\n[{name.upper()}]")

        # Analyze token lengths
        exceeded_files = []
        token_counts = []

        for filename, tokens in file_tokens.items():
            count = len(tokens) + 2  # +2 for [CLS] and [SEP]
            token_counts.append(count)

            if count > max_seq_length:
                exceeded_files.append((filename, count, count - max_seq_length))

        print(f"  Total files: {len(file_tokens)}")
        print(f"  Files exceeding {max_seq_length} tokens: {len(exceeded_files)} ({len(exceeded_files)/len(file_tokens)*100:.1f}%)")

        if token_counts:
            token_counts.sort()
            print(f"\n  Token count distribution (after subword tokenization):")
            print(f"    Min: {min(token_counts)}")
            print(f"    Max: {max(token_counts)}")
            print(f"    Median: {token_counts[len(token_counts)//2]}")
            print(f"    Mean: {sum(token_counts)/len(token_counts):.1f}")

        if exceeded_files:
            exceeded_files.sort(key=lambda x: -x[1])
            print(f"\n  Files exceeding limit (sorted by token count):")
            for filename, count, excess in exceeded_files[:10]:
                print(f"    {filename}: {count} tokens (excess: {excess})")

        # Return exceeded files for cross-reference
        if name == "op_test":
            return exceeded_files

    return []

def cross_reference_with_errors(exceeded_files, error_file):
    """Cross-reference exceeded files with TOTAL errors"""
    import json

    if not os.path.exists(error_file):
        return

    with open(error_file, "r") as f:
        error_data = json.load(f)

    exceeded_set = set(f[0] for f in exceeded_files)

    print("\n" + "=" * 80)
    print("CROSS-REFERENCE: Tokenized Length vs TOTAL Empty Predictions")
    print("=" * 80)

    total_errors = error_data.get("total", {}).get("errors", [])
    empty_total = [e for e in total_errors if not e["pred"].strip()]

    print(f"\n  Total empty TOTAL predictions: {len(empty_total)}")

    empty_in_exceeded = [e for e in empty_total if e["filename"] in exceeded_set]
    print(f"  Empty TOTAL in exceeded files: {len(empty_in_exceeded)}")

    # Sample of empty TOTALs NOT in exceeded files
    empty_not_exceeded = [e for e in empty_total if e["filename"] not in exceeded_set]
    if empty_not_exceeded:
        print(f"\n  Empty TOTAL predictions NOT in exceeded files (sample):")
        for e in empty_not_exceeded[:5]:
            print(f"    - {e['filename']}: gold='{e['gold']}'")

def main():
    data_dir = "../data"
    error_file = "./output/error_analysis.json"

    exceeded_files = analyze_tokenization(data_dir)
    cross_reference_with_errors(exceeded_files, error_file)

if __name__ == "__main__":
    main()
