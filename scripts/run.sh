#!/bin/bash

set -e

cd "$(dirname "$0")/../code"

# ============================================
# 기본값 설정
# ============================================
DATA_DIR="../data"
MODEL_TYPE="layoutlm"
MODEL_NAME="microsoft/layoutlm-base-uncased"
EPOCHS=10
LR=5e-5
LOSS_TYPE="ce"
FOCAL_GAMMA=2.0
USE_CLASS_WEIGHTS=false
FREEZE_VISUAL=false

# ============================================
# 인자 파싱
# ============================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TYPE="$2"
            if [ "$MODEL_TYPE" == "layoutlmv3" ]; then
                MODEL_NAME="microsoft/layoutlmv3-base"
            else
                MODEL_NAME="microsoft/layoutlm-base-uncased"
            fi
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --loss)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --gamma)
            FOCAL_GAMMA="$2"
            shift 2
            ;;
        --class-weights)
            USE_CLASS_WEIGHTS=true
            shift
            ;;
        --freeze-visual)
            FREEZE_VISUAL=true
            shift
            ;;
        -h|--help)
            echo "Usage: bash run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model TYPE       layoutlm (default) or layoutlmv3"
            echo "  --epochs N         Number of epochs (default: 10)"
            echo "  --lr RATE          Learning rate (default: 5e-5)"
            echo "  --loss TYPE        ce, weighted, or focal (default: ce)"
            echo "  --gamma VALUE      Focal loss gamma (default: 2.0)"
            echo "  --class-weights    Enable class weights for loss"
            echo "  --freeze-visual    Freeze visual encoder (layoutlmv3 only)"
            echo ""
            echo "Examples:"
            echo "  bash run.sh                                    # Baseline"
            echo "  bash run.sh --loss weighted                    # Weighted CE"
            echo "  bash run.sh --loss focal --gamma 2.0           # Focal Loss"
            echo "  bash run.sh --loss focal --class-weights       # Focal + Weights"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================
# 학습
# ============================================
echo "=========================================="
echo "1. Training ${MODEL_TYPE} model..."
echo "   - epochs: ${EPOCHS}"
echo "   - learning_rate: ${LR}"
echo "   - loss_type: ${LOSS_TYPE}"
echo "   - use_class_weights: ${USE_CLASS_WEIGHTS}"
echo "=========================================="

TRAIN_CMD="python train.py \
    --model_name_or_path '${MODEL_NAME}' \
    --model_type ${MODEL_TYPE} \
    --data_dir ${DATA_DIR} \
    --labels ${DATA_DIR}/labels.txt \
    --do_train \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --loss_type ${LOSS_TYPE} \
    --focal_gamma ${FOCAL_GAMMA} \
    --overwrite_model_dir"

if [ "$MODEL_TYPE" == "layoutlm" ]; then
    TRAIN_CMD="${TRAIN_CMD} --do_lower_case"
fi

if [ "$USE_CLASS_WEIGHTS" == "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_class_weights"
fi

if [ "$MODEL_TYPE" == "layoutlmv3" ] && [ "$FREEZE_VISUAL" == "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --freeze_visual_encoder"
fi

eval $TRAIN_CMD

# ============================================
# 추론 (Test)
# ============================================
echo "=========================================="
echo "2. Inference (test mode)..."
echo "=========================================="

INFER_CMD="python inference.py \
    --model_name_or_path '${MODEL_NAME}' \
    --model_type ${MODEL_TYPE} \
    --data_dir ${DATA_DIR} \
    --labels ${DATA_DIR}/labels.txt \
    --do_predict \
    --max_seq_length 512 \
    --mode test \
    --overwrite_output_dir"

if [ "$MODEL_TYPE" == "layoutlm" ]; then
    INFER_CMD="${INFER_CMD} --do_lower_case"
fi

eval $INFER_CMD

# ============================================
# 추론 (Op_test)
# ============================================
echo "=========================================="
echo "3. Inference (op_test mode)..."
echo "=========================================="

INFER_CMD="python inference.py \
    --model_name_or_path '${MODEL_NAME}' \
    --model_type ${MODEL_TYPE} \
    --data_dir ${DATA_DIR} \
    --labels ${DATA_DIR}/labels.txt \
    --do_predict \
    --max_seq_length 512 \
    --mode op_test \
    --overwrite_output_dir"

if [ "$MODEL_TYPE" == "layoutlm" ]; then
    INFER_CMD="${INFER_CMD} --do_lower_case"
fi

eval $INFER_CMD

# ============================================
# 평가
# ============================================
echo "=========================================="
echo "4. Evaluation..."
echo "=========================================="
python evaluation.py \
    --data_dir "$DATA_DIR" \
    --output_dir "./output"

echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
