#!/usr/bin/env python3
"""Show actual model input format"""

import torch
from transformers import BertTokenizer, LayoutLMConfig, LayoutLMForTokenClassification
from utils import SROIEDataset
from torch.nn import CrossEntropyLoss

def main():
    # 설정
    class Args:
        data_dir = "../data"
        model_name_or_path = "microsoft/layoutlm-base-uncased"
        model_type = "layoutlm"
        max_seq_length = 512
        local_rank = -1
        overwrite_cache = False

    args = Args()

    # 라벨 로드
    with open("../data/labels.txt", "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels

    print("=" * 80)
    print("1. 라벨 매핑")
    print("=" * 80)
    label_map = {i: label for i, label in enumerate(labels)}
    for i, label in enumerate(labels):
        print(f"   {i} → {label}")

    # 토크나이저
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    pad_token_label_id = CrossEntropyLoss().ignore_index  # -100

    print(f"\n   pad_token_label_id (무시되는 라벨): {pad_token_label_id}")

    # 데이터셋 로드
    print("\n" + "=" * 80)
    print("2. 데이터셋 로드")
    print("=" * 80)

    dataset = SROIEDataset(args, tokenizer, labels, pad_token_label_id, mode="train")

    print(f"   총 샘플 수: {len(dataset)}")

    # 첫 번째 샘플 가져오기
    print("\n" + "=" * 80)
    print("3. 첫 번째 샘플 (하나의 영수증 문서)")
    print("=" * 80)

    sample = dataset[0]
    input_ids, attention_mask, token_type_ids, label_ids, bboxes = sample

    print(f"\n   [텐서 형태]")
    print(f"   input_ids shape:      {input_ids.shape}  # (512,)")
    print(f"   attention_mask shape: {attention_mask.shape}  # (512,)")
    print(f"   token_type_ids shape: {token_type_ids.shape}  # (512,)")
    print(f"   label_ids shape:      {label_ids.shape}  # (512,)")
    print(f"   bboxes shape:         {bboxes.shape}  # (512, 4)")

    # 실제 토큰 개수 (패딩 제외)
    actual_length = attention_mask.sum().item()
    print(f"\n   실제 토큰 수 (패딩 제외): {actual_length}")

    print("\n" + "=" * 80)
    print("4. 실제 데이터 내용 (처음 30개 토큰)")
    print("=" * 80)

    print(f"\n   {'위치':<4} {'토큰ID':<8} {'토큰':<15} {'라벨ID':<8} {'라벨':<12} {'bbox':<20}")
    print("   " + "-" * 75)

    for i in range(min(30, actual_length)):
        token_id = input_ids[i].item()
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        label_id = label_ids[i].item()
        bbox = bboxes[i].tolist()

        if label_id == -100:
            label_str = "(무시)"
        else:
            label_str = label_map.get(label_id, "?")

        print(f"   {i:<4} {token_id:<8} {token:<15} {label_id:<8} {label_str:<12} {bbox}")

    print("\n" + "=" * 80)
    print("5. 배치 형태 (모델에 실제 들어가는 형태)")
    print("=" * 80)

    # 배치 생성 (8개 샘플)
    batch_size = min(8, len(dataset))
    batch_input_ids = torch.stack([dataset[i][0] for i in range(batch_size)])
    batch_attention_mask = torch.stack([dataset[i][1] for i in range(batch_size)])
    batch_token_type_ids = torch.stack([dataset[i][2] for i in range(batch_size)])
    batch_labels = torch.stack([dataset[i][3] for i in range(batch_size)])
    batch_bboxes = torch.stack([dataset[i][4] for i in range(batch_size)])

    print(f"\n   batch_size = {batch_size}")
    print(f"\n   [배치 텐서 형태]")
    print(f"   input_ids:      {batch_input_ids.shape}      # (batch, seq_len)")
    print(f"   attention_mask: {batch_attention_mask.shape}      # (batch, seq_len)")
    print(f"   token_type_ids: {batch_token_type_ids.shape}      # (batch, seq_len)")
    print(f"   labels:         {batch_labels.shape}      # (batch, seq_len)")
    print(f"   bbox:           {batch_bboxes.shape}   # (batch, seq_len, 4)")

    print("\n" + "=" * 80)
    print("6. 모델 forward 호출 형태")
    print("=" * 80)

    print("""
   model(
       input_ids=tensor([[101, 2728, 4492, ...],    # [8, 512] - 토큰 ID
                         [101, 3421, 2098, ...],
                         ...]),

       attention_mask=tensor([[1, 1, 1, 1, ...],    # [8, 512] - 실제 토큰=1, 패딩=0
                              [1, 1, 1, 1, ...],
                              ...]),

       token_type_ids=tensor([[0, 0, 0, 0, ...],    # [8, 512] - 세그먼트 (항상 0)
                              [0, 0, 0, 0, ...],
                              ...]),

       bbox=tensor([[[0, 0, 0, 0],                  # [8, 512, 4] - 위치 좌표
                     [155, 24, 280, 63],             # [x1, y1, x2, y2]
                     [291, 24, 460, 63],
                     ...],
                    ...]),

       labels=tensor([[-100, 4, 4, ...],            # [8, 512] - 정답 라벨
                      [-100, 2, 2, ...],             # -100은 loss 계산에서 제외
                      ...])
   )
   """)

    print("=" * 80)
    print("7. bbox 좌표 의미")
    print("=" * 80)
    print("""
   bbox = [x1, y1, x2, y2]  (0-1000 범위로 정규화됨)

   ┌─────────────────────────────────┐
   │  (x1, y1)                       │
   │     ┌───────────────┐           │
   │     │    "TAN"      │           │  ← 토큰의 bounding box
   │     └───────────────┘           │
   │                  (x2, y2)       │
   │                                 │
   │         영수증 이미지            │
   │         (1000 x 1000 정규화)    │
   └─────────────────────────────────┘

   예: bbox = [155, 24, 280, 63]
       → 왼쪽 상단 (155, 24), 오른쪽 하단 (280, 63)
   """)

if __name__ == "__main__":
    main()
