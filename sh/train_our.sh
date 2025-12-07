#!/bin/bash

# Script to train the OUR (SupCon) model

# Define the global variables
BASEFOLDER=$(pwd)
PYTHON="python"
SCRIPT_PATH=${BASEFOLDER}/src/OUR.py

# Model Parameters (optimized for MacBook testing)
INPUT_PATH=${BASEFOLDER}/train_2m.csv
K=4
DIM=256
EPOCHNUM=300
LR=0.001
BATCH_SIZE=10000
MAXREADNUM=10000
MAX_VIEWS_PER_READ=10
TEMPERATURE=0.1
SEED=26042024
CHECKPOINT=0
DEVICE=gpu
WORKERS=1

# Define the output path
OUTPUT_PATH=${BASEFOLDER}/models/our_k${K}_dim${DIM}_epoch=${EPOCHNUM}_LR=${LR}_batch=${BATCH_SIZE}.pt

# Define the command
CMD="$PYTHON ${SCRIPT_PATH} --input $INPUT_PATH --k ${K} --dim ${DIM} --epoch $EPOCHNUM --lr $LR"
CMD="${CMD} --batch_size ${BATCH_SIZE} --max_read_num ${MAXREADNUM}"
CMD="${CMD} --max_views_per_read ${MAX_VIEWS_PER_READ} --temperature ${TEMPERATURE}"
CMD="${CMD} --device ${DEVICE} --workers_num ${WORKERS} --output ${OUTPUT_PATH} --seed ${SEED} --checkpoint ${CHECKPOINT}"

# Run the command
echo ${OUTPUT_PATH}
$CMD
