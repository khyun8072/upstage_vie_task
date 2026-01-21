#!/usr/bin/env python3
"""Error Analysis Script for VIE Task"""

import json
import os
import csv
from collections import defaultdict
from glob import glob

def load_ground_truth(gt_dir):
    """Load ground truth from entities folder"""
    gt_parses = {}
    for filepath in glob(os.path.join(gt_dir, "*.txt")):
        filename = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r") as f:
            data = json.load(f)
        gt_parses[filename] = data
    return gt_parses

def load_predictions(output_path):
    """Load predictions from output.csv"""
    pr_parses = defaultdict(lambda: {"company": [], "date": [], "address": [], "total": []})
    with open(output_path, "r", encoding="utf-8") as f:
        for line in csv.reader(f):
            if len(line) == 3:
                text, pred_label, filename = line
                if pred_label != "O":
                    label_map = {
                        "S-COMPANY": "company",
                        "S-DATE": "date",
                        "S-ADDRESS": "address",
                        "S-TOTAL": "total"
                    }
                    if pred_label in label_map:
                        pr_parses[filename][label_map[pred_label]].append(text)

    # Join tokens into strings
    for filename, pr_parse in pr_parses.items():
        for label, value in pr_parse.items():
            pr_parse[label] = " ".join(value)

    return pr_parses

def normalize(s):
    """Simple normalization"""
    return " ".join(s.lower().split())

def exact_match(pred, gold):
    """Check exact match after normalization"""
    return normalize(pred) == normalize(gold)

def char_f1(pred, gold):
    """Character-level F1 score"""
    pred_chars = list(normalize(pred).replace(" ", ""))
    gold_chars = list(normalize(gold).replace(" ", ""))

    if not pred_chars and not gold_chars:
        return 1.0
    if not pred_chars or not gold_chars:
        return 0.0

    from collections import Counter
    common = Counter(pred_chars) & Counter(gold_chars)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_chars)
    recall = num_same / len(gold_chars)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def analyze_errors(gt_parses, pr_parses):
    """Analyze errors by entity type"""
    entities = ["company", "date", "address", "total"]

    results = {
        entity: {
            "total": 0,
            "exact_match": 0,
            "f1_sum": 0.0,
            "errors": []
        } for entity in entities
    }

    common_files = set(gt_parses.keys()) & set(pr_parses.keys())
    print(f"\n=== Analysis on {len(common_files)} files ===\n")

    for filename in common_files:
        gt = gt_parses[filename]
        pr = pr_parses[filename]

        for entity in entities:
            gold = " ".join(gt[entity]) if isinstance(gt[entity], list) else gt[entity]
            pred = pr.get(entity, "")

            results[entity]["total"] += 1

            em = exact_match(pred, gold)
            f1 = char_f1(pred, gold)

            results[entity]["f1_sum"] += f1
            if em:
                results[entity]["exact_match"] += 1
            else:
                results[entity]["errors"].append({
                    "filename": filename,
                    "gold": gold,
                    "pred": pred,
                    "f1": f1
                })

    return results

def print_analysis(results):
    """Print analysis results"""
    print("=" * 80)
    print("ENTITY-LEVEL PERFORMANCE ANALYSIS")
    print("=" * 80)

    for entity, data in results.items():
        total = data["total"]
        em = data["exact_match"]
        f1_avg = data["f1_sum"] / total if total > 0 else 0
        em_rate = em / total * 100 if total > 0 else 0

        print(f"\n[{entity.upper()}]")
        print(f"  Total samples: {total}")
        print(f"  Exact Match: {em}/{total} ({em_rate:.1f}%)")
        print(f"  Avg Char F1: {f1_avg*100:.1f}%")
        print(f"  Error count: {len(data['errors'])}")

    print("\n" + "=" * 80)
    print("ERROR CASE ANALYSIS (Top 5 per entity)")
    print("=" * 80)

    for entity, data in results.items():
        print(f"\n[{entity.upper()}] - Worst cases by F1 score:")
        errors = sorted(data["errors"], key=lambda x: x["f1"])[:5]

        for i, err in enumerate(errors, 1):
            print(f"\n  {i}. File: {err['filename']}")
            print(f"     Gold: '{err['gold']}'")
            print(f"     Pred: '{err['pred']}'")
            print(f"     F1: {err['f1']*100:.1f}%")

def analyze_error_patterns(results):
    """Analyze common error patterns"""
    print("\n" + "=" * 80)
    print("ERROR PATTERN ANALYSIS")
    print("=" * 80)

    for entity, data in results.items():
        errors = data["errors"]

        # Pattern analysis
        empty_pred = sum(1 for e in errors if not e["pred"].strip())
        partial_match = sum(1 for e in errors if e["f1"] > 0.5 and e["f1"] < 1.0)
        total_miss = sum(1 for e in errors if e["f1"] == 0)

        print(f"\n[{entity.upper()}] Error Patterns:")
        print(f"  - Empty predictions: {empty_pred}")
        print(f"  - Partial matches (F1 > 50%): {partial_match}")
        print(f"  - Total miss (F1 = 0): {total_miss}")

def main():
    gt_dir = "../data/test/entities"
    output_path = "./output/output.csv"

    print("Loading ground truth...")
    gt_parses = load_ground_truth(gt_dir)
    print(f"Loaded {len(gt_parses)} ground truth files")

    print("Loading predictions...")
    pr_parses = load_predictions(output_path)
    print(f"Loaded predictions for {len(pr_parses)} files")

    results = analyze_errors(gt_parses, pr_parses)
    print_analysis(results)
    analyze_error_patterns(results)

    # Save detailed results
    output_file = "./output/error_analysis.json"
    with open(output_file, "w") as f:
        # Convert errors to serializable format
        serializable_results = {}
        for entity, data in results.items():
            serializable_results[entity] = {
                "total": data["total"],
                "exact_match": data["exact_match"],
                "avg_f1": data["f1_sum"] / data["total"] if data["total"] > 0 else 0,
                "error_count": len(data["errors"]),
                "errors": data["errors"]
            }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
