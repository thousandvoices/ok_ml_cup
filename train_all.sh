set -e

DATA_PATH="$1"
TEMP_DATA_DIR="$2"
MODEL_DIR="$3"

if [[ -z "$MODEL_DIR" ]]; then
    echo "Usage:"
    echo "    `basename $0` DATA_PATH TEMP_DATA_DIR MODEL_DIR"
    echo "Parameters:"
    echo "    DATA_PATH - path to file with dataset"
    echo "    TEMP_DATA_DIR - path to directory where data split for training and validation will be saved"
    echo "    MODEL_DIR - path to directory where the models will be saved"
    exit 1
fi

mkdir -p "$TEMP_DATA_DIR"
python3 prepare_data.py --train-path "$DATA_PATH" --output-dir "$TEMP_DATA_DIR"

ALL_MODELS=""

SAVE_PATH="${MODEL_DIR}/intermediate/rubert"
ALL_MODELS="${ALL_MODELS} ${SAVE_PATH}"
python3 train.py --classifier DeepPavlov/rubert-base-cased-conversational \
    --train "${TEMP_DATA_DIR}/train.csv" \
    --validation "${TEMP_DATA_DIR}/validation.csv" \
    --save-path "$SAVE_PATH" \
    --layers 12 \
    --epochs 2

SAVE_PATH="${MODEL_DIR}/intermediate/xlm-roberta-large"
ALL_MODELS="${ALL_MODELS} ${SAVE_PATH}"
python3 train.py --classifier xlm-roberta-large \
    --train "${TEMP_DATA_DIR}/train.csv" \
    --validation "${TEMP_DATA_DIR}/validation.csv" \
    --save-path "$SAVE_PATH" \
    --layers 24 \
    --epochs 3

python3 inference.py --data-path "${TEMP_DATA_DIR}/train.csv" \
    --output-path "${TEMP_DATA_DIR}/train_pseudolabels.csv" \
    --fast-classifiers $ALL_MODELS \
    --write-text \
    --augment

python3 train.py --classifier DeepPavlov/rubert-base-cased-conversational \
    --train "${TEMP_DATA_DIR}/train_pseudolabels.csv" \
    --validation "${TEMP_DATA_DIR}/validation.csv" \
    --save-path "${MODEL_DIR}/final/rubert" \
    --layers 4 \
    --epochs 3 \
    --distill \
    --export-type cpu
