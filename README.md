# VIE (Visual Information Extraction) - SROIE 영수증 정보 추출

> Upstage 딥러닝 코딩 테스트 과제

## 프로젝트 개요

영수증 이미지에서 4가지 핵심 정보를 추출하는 Named Entity Recognition(NER) 태스크입니다.

OCR 데이터와 문서 이미지를 다루는 첫 경험이었습니다. 기존에는 일반적인 NLP 태스크나 이미지 분류를 주로 다뤄왔기에, 텍스트와 레이아웃(bounding box) 정보를 함께 활용하는 LayoutLM 계열 모델은 새로운 도전이었습니다.

- **COMPANY**: 회사/상호명
- **DATE**: 날짜
- **ADDRESS**: 주소
- **TOTAL**: 총액

## 최종 성능

| 메트릭 | Baseline | 최종 모델 |
|--------|----------|-----------|
| Entity F1 | 79.31% | **84.15%** (+4.84%p) |
| Entity EM | 48.33% | 43.08% |

---

### 데이터 준비

SROIE 데이터셋을 `data/` 폴더에 다운로드해야 합니다.

```
data/
├── train/
│   ├── img/          # 학습 이미지
│   └── entities/     # 학습 라벨 (JSON)
├── test/
│   ├── img/          # 테스트 이미지
│   └── entities/     # 테스트 라벨 (JSON)
├── labels.txt        # 라벨 정의 (S-COMPANY, S-DATE, S-ADDRESS, S-TOTAL, O)
├── train.txt         # 학습 데이터 (텍스트 + bbox)
├── test.txt          # 테스트 데이터
└── op_test.txt       # 평가용 테스트 데이터
```


## 문제 해결 과정

### 1단계: Baseline 확인

LayoutLM v1 (epochs=10, lr=5e-5)으로 학습한 결과입니다.

| Entity F1 | Entity EM |
|-----------|-----------|
| 79.31% | 48.33% |

처음에는 학습이 부족하거나 하이퍼파라미터 문제라고 생각했습니다.

### 2단계: 하이퍼파라미터 튜닝

epochs, learning rate, warmup, weight decay를 바꿔가며 실험했습니다.

| 실험 | epochs | lr | warmup | weight_decay | Entity F1 |
|------|--------|-----|--------|--------------|-----------|
| Baseline | 10 | 5e-5 | 0 | 0.0 | **79.31%** |
| epochs=20 | 20 | 5e-5 | 0 | 0.0 | 76.88% |
| epochs=30 | 30 | 5e-5 | 0 | 0.0 | 78.12% |
| lr=2e-5 | 30 | 2e-5 | 0 | 0.0 | 78.69% |
| warmup+wd | 30 | 2e-5 | 50 | 0.01 | 78.28% |

epochs를 늘리면 오히려 과적합이 발생했고, 어떤 조합을 시도해도 Baseline을 넘지 못했습니다. 하이퍼파라미터 문제가 아니라는 걸 깨달았습니다.

```bash
bash scripts/run.sh                      # Baseline
bash scripts/run.sh --epochs 30 --lr 2e-5
```

### 3단계: 데이터 분석

원인을 찾기 위해 학습 데이터를 분석했습니다. `code/analysis/` 폴더의 스크립트들을 사용했습니다.

```bash
python code/analysis/error_analysis.py      # 에러 패턴 분석
python code/analysis/total_analysis.py      # TOTAL 예측 실패 분석
```

#### 라벨 분포

```
라벨           개수        비율
─────────────────────────────────
O              61,423      84.8%
S-ADDRESS       6,906       9.5%
S-COMPANY       2,689       3.7%
S-DATE            730       1.0%
S-TOTAL           642       0.9%
```

O 라벨이 전체의 84.8%를 차지하고, S-TOTAL은 0.9%밖에 안 됩니다.

#### 엔티티별 성능 (Baseline)

| 엔티티 | Exact Match | Char F1 | Empty 예측 |
|--------|-------------|---------|------------|
| COMPANY | 51.3% | 90.4% | 6개 |
| DATE | 69.7% | 90.0% | 11개 |
| ADDRESS | 2.1% | 91.0% | 0개 |
| **TOTAL** | **26.2%** | **42.5%** | **88개** |

TOTAL만 유독 성능이 낮습니다. 347개 파일 중 88개(45%)에서 TOTAL 예측이 아예 비어있었습니다.

#### TOTAL 예측 실패 원인

`total_analysis.py`로 분석한 결과:
- 'TOTAL' 키워드 409개 중 **409개 전부** O로 예측됨
- TOTAL 뒤에 나오는 숫자도 1543개 중 83개(5.4%)만 S-TOTAL로 예측됨
- 학습 데이터에서 S-TOTAL이 0.9%밖에 없어서 패턴을 제대로 못 배운 것으로 보임

클래스 불균형이 핵심 문제였습니다.

### 4단계: Loss 함수 변경

클래스 불균형을 해결하기 위해 Loss 함수를 바꿨습니다.

#### Weighted Cross Entropy

기본 CE Loss는 모든 클래스를 동등하게 취급합니다. 그래서 84.8%를 차지하는 O 라벨 위주로 학습되고, 0.9%밖에 안 되는 S-TOTAL은 거의 무시됩니다.

Weighted CE는 클래스별로 다른 가중치를 줘서 이 문제를 해결합니다. 샘플이 적은 클래스에 높은 가중치를 주면, 해당 클래스를 틀렸을 때 loss가 더 커지므로 모델이 더 신경 쓰게 됩니다.

#### Focal Loss

Focal Loss는 모델이 "쉽게 맞추는 샘플"의 loss를 줄이고, "어려워하는 샘플"에 집중하게 만듭니다. 예측 확률이 높은(확신 있는) 샘플은 loss 기여도가 낮아지고, 확률이 낮은(헷갈리는) 샘플은 loss가 커집니다.

TOTAL처럼 모델이 자주 틀리는 클래스에 효과적일 거라고 생각했습니다.

#### 클래스 가중치 설정

샘플 수가 적은 클래스에 높은 가중치를 줬습니다.

| 클래스 | 샘플 수 | 비율 | 가중치 |
|--------|--------|------|--------|
| O | 61,423 | 84.8% | 0.5 |
| S-ADDRESS | 6,906 | 9.5% | 2.0 |
| S-COMPANY | 2,689 | 3.7% | 5.0 |
| S-DATE | 730 | 1.0% | 15.0 |
| S-TOTAL | 642 | 0.9% | 20.0 |

S-TOTAL에 가장 높은 가중치(20.0)를 준 이유는 샘플이 가장 적으면서 Baseline에서 F1 42.5%로 성능이 가장 낮았기 때문입니다.

#### 실험 결과

| 실험 | loss_type | gamma | class_weights | Entity F1 |
|------|-----------|-------|---------------|-----------|
| Baseline | ce | - | X | 78.23% |
| Weighted CE | weighted | - | O | 83.56% |
| Focal γ=2 | focal | 2.0 | X | 77.56% |
| **Focal+Weights** | focal | 2.0 | O | **84.15%** |

Weighted CE만으로도 78% → 83%로 크게 올랐습니다.

#### Focal Loss 단독 사용이 효과 없었던 이유

Focal Loss는 원래 Object Detection(RetinaNet)에서 background vs foreground 불균형 문제를 해결하기 위해 제안되었습니다. "쉬운 샘플"(background)의 loss를 줄여서 "어려운 샘플"(foreground)에 집중하게 만드는 방식입니다.

그런데 이 데이터에서는 상황이 다릅니다:
- O 라벨(84.8%)은 대부분 "쉬운 샘플"이면서 동시에 "다수 클래스"
- S-TOTAL(0.9%)은 "어려운 샘플"이면서 동시에 "소수 클래스"

Focal Loss는 confidence 기반으로 동작하는데, 모델이 S-TOTAL을 계속 틀리면(confidence가 낮으면) 해당 샘플의 loss는 커집니다. 하지만 S-TOTAL 샘플 자체가 너무 적어서 전체 loss에서 차지하는 비중이 작습니다. 결국 O 라벨의 "어려운 샘플들"이 학습을 지배하게 됩니다.

Class Weights를 함께 쓰면 S-TOTAL의 loss가 20배로 증폭되어, 적은 샘플 수를 보상할 수 있습니다. 그래서 Focal + Weights 조합이 효과적이었습니다.

```bash
bash scripts/run.sh --loss weighted               # Weighted CE
bash scripts/run.sh --loss focal --class-weights  # 최종 모델
```

### 5단계: 후처리

TOTAL에서 같은 값이 중복 예측되는 경우("53.00 53.00")를 처리하는 후처리를 추가했는데, 큰 효과는 없었습니다.

### 6단계: OCR 품질 분석

Test 데이터에서는 토큰 단위 F1이 90% 이상인데 Op_test에서는 낮게 나오는 이유를 분석했습니다.

| 데이터셋 | OCR 결과 예시 |
|---------|---------------|
| Test | `$6.60` |
| Op_test | `S6.60` |

Op_test의 OCR 품질이 다르다는 걸 발견했습니다. 모델 자체는 잘 학습되었지만 OCR 오류 때문에 성능이 낮게 측정되는 부분이 있었습니다.

---

## 추가 실험: LayoutLMv3

이미지 정보를 함께 쓰면 성능이 오를 거라고 생각해서 LayoutLMv3를 실험했습니다.

데이터가 213개밖에 없어서 Visual Encoder를 freeze하는 전략도 시도했습니다.

| 실험 | lr | Visual Encoder | Entity F1 |
|------|-----|----------------|-----------|
| v3 | 5e-5 | 전체 학습 | 33.20% |
| v3 | 2e-5 | 전체 학습 | 32.94% |
| v3 + freeze | 5e-5 | Freeze | 33.00% |
| v3 + freeze | 2e-5 | Freeze | 33.72% |

v1(84%)에 비해 v3(33%)가 훨씬 낮게 나왔습니다.

#### 실패 원인

1. 학습 데이터 213개는 ViT 기반 이미지 인코더를 학습시키기엔 너무 적음
2. LayoutLMv3는 이미지를 224x224로 리사이즈하는데, 세로로 긴 영수증이 리사이즈되면서 텍스트가 뭉개짐

```bash
bash scripts/run.sh --model layoutlmv3
bash scripts/run.sh --model layoutlmv3 --freeze-visual
```

---

## 프로젝트 구조

```
.
├── code/
│   ├── train.py              # 학습
│   ├── inference.py          # 추론
│   ├── evaluation.py         # 평가
│   ├── utils.py              # 데이터셋, 유틸리티
│   ├── requirements.txt
│   └── analysis/             # 분석 스크립트
│       ├── error_analysis.py       # 엔티티별 에러 분석
│       ├── total_analysis.py       # TOTAL 예측 실패 분석
│       ├── seq_length_analysis.py  # 시퀀스 길이 분석
│       ├── tokenization_analysis.py
│       └── show_model_input.py
├── data/
│   ├── train/
│   ├── test/
│   └── labels.txt
├── scripts/
│   └── run.sh
└── README.md
```

---

## 결론

| 단계 | 방법 | Entity F1 | 변화 |
|------|------|-----------|------|
| Baseline | CE Loss | 79.31% | - |
| 방법 1 | 하이퍼파라미터 튜닝 | 78.69% | -0.62%p |
| 방법 2 | Weighted CE | 83.56% | +4.25%p |
| **최종** | **Focal + Weights** | **84.15%** | **+4.84%p** |
| 추가 | LayoutLMv3 | 33.72% | -45.59%p |

### 실험에서 얻은 인사이트

**1. 하이퍼파라미터 튜닝의 한계**

epochs를 10 → 30으로 늘렸을 때 오히려 성능이 떨어졌습니다. 학습 데이터가 213개로 적어서 금방 수렴하고, 이후로는 과적합이 발생한 것으로 보입니다. learning rate를 낮추거나 regularization(weight decay)을 추가해도 근본적인 클래스 불균형 문제는 해결되지 않았습니다.

돌이켜보면, 성능이 안 오를 때 "학습이 부족한가?"보다 "데이터에 문제가 있나?"를 먼저 확인했어야 했습니다.

**2. Loss 함수 선택 기준**

| 상황 | 권장 방법 |
|------|----------|
| 클래스 불균형만 있는 경우 | Weighted CE |
| 쉬운/어려운 샘플 구분이 명확한 경우 | Focal Loss |
| 둘 다 해당하는 경우 | Focal + Weights |

이 데이터는 클래스 불균형이 극심해서(O: 84.8% vs S-TOTAL: 0.9%) Weighted CE만으로도 큰 효과가 있었습니다. Focal Loss는 단독으로는 효과가 없었지만, Weights와 함께 쓰면 추가로 0.6%p 향상되었습니다.

**3. 모델 복잡도와 데이터 크기의 관계**

LayoutLMv3가 v1보다 성능이 낮은 건 의외였습니다. 원인을 분석해보면:

- LayoutLM v1: 텍스트 + bbox만 사용, 파라미터 ~113M
- LayoutLMv3: 텍스트 + bbox + 이미지(ViT), 파라미터 ~133M

v3의 ViT 인코더는 ImageNet으로 사전학습되어 있지만, 영수증 도메인에 맞게 fine-tuning하려면 충분한 데이터가 필요합니다. 213개는 부족했고, freeze해도 텍스트-이미지 간 alignment를 학습하기엔 역부족이었습니다.

또한 영수증은 세로로 긴 문서인데, v3는 224x224 정사각형으로 리사이즈합니다. aspect ratio가 크게 변하면서 텍스트가 뭉개지는 문제도 있었습니다.

**4. EM(Exact Match) 하락 원인**

F1은 79% → 84%로 올랐는데, EM은 48% → 43%로 떨어졌습니다. 이유를 분석해보면:

- Weighted Loss로 S-TOTAL 예측이 늘어남 → 부분 일치 증가 → F1 상승
- 하지만 과하게 예측하는 경우도 생김 → 완전 일치 감소 → EM 하락

예: 정답이 "53.00"인데 "53.00 GST"로 예측하면 F1은 높지만 EM은 0입니다.

### 추가로 시도해볼 방법

**LayoutLMv3 개선**
- 이미지 리사이즈 대신 패딩을 추가해서 aspect ratio 유지
- 세로로 긴 이미지를 여러 타일로 분할 후 각각 처리
- 문서 도메인(DocVQA, FUNSD 등)으로 추가 사전학습된 체크포인트 사용

**데이터 증강**
- 이미지: 밝기/대비 조절, 노이즈 추가 (bbox 좌표에 영향 없는 변환만)
- 약간의 회전/스케일 변환 시 bbox 좌표도 함께 변환 필요

**TOTAL 성능 추가 개선**
- TOTAL 주변 컨텍스트 패턴 분석 후 규칙 기반 후처리 추가
- S-TOTAL 샘플만 오버샘플링
- 2-stage 접근: 1단계에서 TOTAL 후보 추출 → 2단계에서 검증
