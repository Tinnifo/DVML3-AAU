#!/usr/bin/bash

# Script to train the REVISIT model

# Define the global variables
BASEFOLDER=$(pwd)
PYTHON="python"
SCRIPT_PATH=${BASEFOLDER}/src/REVISIT.py

# Model Parameters
INPUT_PATH=${BASEFOLDER}/train_2m.csv
K=4
DIM=256
EPOCHNUM=1000
LR=0.001
BATCH_SIZE=10000
MAXREADNUM=10000
NEGSAMPLEPERPOS=1000
LOSS_NAME=bern
SEED=26042024
CHECKPOINT=0
DEVICE=cpu
WORKERS=1

# Define the output path
OUTPUT_PATH=${BASEFOLDER}/models/revisit_k${K}_dim${DIM}_epoch=${EPOCHNUM}_LR=${LR}_batch=${BATCH_SIZE}.pt

# Define the command
CMD="$PYTHON ${SCRIPT_PATH} --input $INPUT_PATH --k ${K} --dim ${DIM} --epoch $EPOCHNUM --lr $LR"
CMD="${CMD} --neg_sample_per_pos ${NEGSAMPLEPERPOS} --max_read_num ${MAXREADNUM}"
CMD="${CMD} --batch_size ${BATCH_SIZE} --loss_name ${LOSS_NAME}"
CMD="${CMD} --device ${DEVICE} --workers_num ${WORKERS} --output ${OUTPUT_PATH} --seed ${SEED} --checkpoint ${CHECKPOINT}"

# Run the command
echo ${OUTPUT_PATH}
$CMD
