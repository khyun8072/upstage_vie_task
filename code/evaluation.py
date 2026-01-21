import argparse
from collections import Counter, defaultdict
import csv
from glob import glob
import json
import os
import re
import string

# NOTE: DO NOT MODIFY THE FOLLOWING PATHS
# ---------------------------------------
data_dir = os.environ.get("SM_CHANNEL_EVAL", "../input/data")
output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "./output")
# ---------------------------------------


def read_ground_truths(test_data_dir: str):
    gt_parses = {}
    for filename in glob(test_data_dir):
        with open(filename, "r") as f:
            data = json.load(f)
        filename = os.path.splitext(os.path.basename(filename))[0]
        gt_parses[filename] = data
    return gt_parses


def gen_parsers(output_path: str):
    f = open(output_path, "r", encoding="utf-8")
    pr_parses = defaultdict(lambda: {"company": [], "date": [], "address": [], "total": []})
    for line in csv.reader(f):
        if len(line) == 3:
            text, pred_label, filename = line
            if pred_label != "O":
                if pred_label == "S-COMPANY":
                    pred_label = "company"
                elif pred_label == "S-DATE":
                    pred_label = "date"
                elif pred_label == "S-ADDRESS":
                    pred_label = "address"
                elif pred_label == "S-TOTAL":
                    pred_label = "total"
                pr_parses[filename][pred_label].append(text)
        elif len(line) == 2:
            # Test mode (text, label) - cannot perform entity-level evaluation without filename
            # Just skip or log warning
            pass

    for (filename, pr_parse) in pr_parses.items():
        for (pred_label, value) in pr_parse.items():
            joined = " ".join(value)
            
            # 후처리: TOTAL과 DATE의 중복 값 제거
            # 예: "53.00 53.00" -> "53.00", "15/01/2019 15/01/2019" -> "15/01/2019"
            if pred_label in ["total", "date"]:
                tokens = joined.split()
                if len(tokens) > 1:
                    # 모든 토큰이 동일하면 첫 번째만 사용
                    if len(set(tokens)) == 1:
                        joined = tokens[0]
                    # 일부만 동일하면 중복 제거 (순서 유지)
                    else:
                        seen = set()
                        unique_tokens = []
                        for t in tokens:
                            if t not in seen:
                                seen.add(t)
                                unique_tokens.append(t)
                        joined = " ".join(unique_tokens)
            
            pr_parse[pred_label] = joined
    f.close()
    return pr_parses


def normalize_answer(s, remove_whitespace: bool = False):
    def remove_(text):
        """불필요한 기호 제거"""
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub("《", " ", text)
        text = re.sub("》", " ", text)
        text = re.sub("<", " ", text)
        text = re.sub(">", " ", text)
        text = re.sub("〈", " ", text)
        text = re.sub("〉", " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        return " ".join(text.split())

    def white_space_remove(text):
        return "".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if remove_whitespace:
        return white_space_remove(remove_punc(lower(remove_(s))))
    else:
        return white_space_fix(remove_punc(lower(remove_(s))))


def get_char_level_f1_score(prediction, ground_truth, remove_whitespace: bool = False):
    prediction_tokens = normalize_answer(prediction, remove_whitespace).split()
    ground_truth_tokens = normalize_answer(ground_truth, remove_whitespace).split()

    # F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth, remove_whitespace: bool = False):
    return normalize_answer(prediction, remove_whitespace) == normalize_answer(
        ground_truth, remove_whitespace
    )


def evaluation(gt_parses, pr_parses: str, detailed=False, **kwargs):
    # Use intersection of gt and pr filenames for evaluation
    common_filenames = set(gt_parses.keys()) & set(pr_parses.keys())
    if len(common_filenames) == 0:
        raise ValueError("No common filenames between ground truth and predictions")

    parses = defaultdict(lambda: {"gold": dict, "infer": dict})
    f1 = exact_match = exact_match_no_space = total = 0
    entity_score_per_entity = defaultdict(
        lambda: {
            "entity_em": 0.0,
            "entity_em_no_space": 0.0,
            "entity_f1": 0.0,
        }
    )
    total_per_entity = defaultdict(int)
    
    # 상세 결과용
    predictions_list = []

    filenames = list(common_filenames)
    for filename in sorted(filenames):
        gt_parse = gt_parses[filename]
        pr_parse = pr_parses[filename]
        
        # 상세 결과용 케이스별 데이터
        case_result = {"filename": filename}

        for key in ["company", "date", "address", "total"]:
            total += 1
            total_per_entity[key] += 1
            ground_truths = " ".join(gt_parse[key]) if isinstance(gt_parse.get(key), list) else gt_parse.get(key, "")
            try:
                prediction = " ".join(pr_parse[key]) if isinstance(pr_parse.get(key), list) else pr_parse.get(key, "")
            except (KeyError, TypeError):
                prediction = ""
            parses[filename][key] = {"gold": ground_truths, "infer": prediction}
            
            em_score = exact_match_score(prediction, ground_truths)
            f1_score_val = get_char_level_f1_score(prediction, ground_truths)
            em_no_space = exact_match_score(prediction, ground_truths, remove_whitespace=True)
            
            exact_match += em_score
            f1 += f1_score_val
            exact_match_no_space += em_no_space

            entity_score_per_entity[key]["entity_em"] += em_score
            entity_score_per_entity[key]["entity_em_no_space"] += em_no_space
            entity_score_per_entity[key]["entity_f1"] += f1_score_val
            
            # 상세 결과 저장
            case_result[key] = {
                "gold": ground_truths,
                "pred": prediction,
                "f1": round(f1_score_val * 100, 2),
                "em": bool(em_score)
            }
        
        predictions_list.append(case_result)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    exact_match_no_space = 100.0 * exact_match_no_space / total

    # get entity score per entities
    per_entity_result = {}
    for key in ["company", "date", "address", "total"]:
        per_entity_result[key] = {
            "f1": round(100.0 * entity_score_per_entity[key]["entity_f1"] / total_per_entity[key], 2),
            "em": round(100.0 * entity_score_per_entity[key]["entity_em"] / total_per_entity[key], 2)
        }

    # 기본 결과 (기존 형식 호환)
    result = {
        'entity_f1': {
            'value': f1,
            'rank': True,
            'decs': True
        },
        "entity_em" : {
            'value': exact_match,
            'rank': False,
            'decs': True
        },
        "entity_em_no_space": {
            'value': exact_match_no_space,
            'rank': False,
            'decs': True
        }
    }
    
    # 상세 결과
    detailed_result = {
        "metrics": {
            "entity_f1": round(f1, 2),
            "entity_em": round(exact_match, 2),
            "per_entity": per_entity_result
        },
        "predictions": predictions_list
    }

    if detailed:
        return json.dumps(result), detailed_result
    return json.dumps(result), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=data_dir, help="path to read the test data")
    parser.add_argument(
        "--output_dir", type=str, default=output_dir, help="path to read the inference result"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Save detailed per-case results to JSON"
    )
    parser.add_argument(
        "--output_json", type=str, default=None, help="Path to save detailed results JSON"
    )
    args = parser.parse_args()

    gt_parses = read_ground_truths(f"{args.data_dir}/test/entities/*")
    pr_parses = gen_parsers(os.path.join(args.output_dir, "output.csv"))

    eval_result, detailed_result = evaluation(gt_parses, pr_parses, detailed=args.detailed or args.output_json)
    print(eval_result)
    
    # 상세 결과 저장
    if args.output_json and detailed_result:
        # 디렉토리 생성
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(detailed_result, f, ensure_ascii=False, indent=2)
        import sys
        print(f"상세 결과 저장됨: {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()

