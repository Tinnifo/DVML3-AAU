#!/usr/bin/bash -l
#SBATCH --job-name=REVISIT_TRAIN
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tolaso24@student.aau.dk

# --- AI-LAB container & project paths ---
PYTORCH_CONTAINER=/ceph/container/pytorch/pytorch_25.10.sif

# Adjust this path if your repo is elsewhere
BASEFOLDER=/ceph/home/student.aau.dk/db56hw/revisitingkmers-p3

PYTHON="python"
SCRIPT_PATH=${BASEFOLDER}/src/REVISIT.py

# --- Data path on AI-LAB ---
DATA_DIR=/ceph/project/p3-kmer/dataset
INPUT_PATH=${DATA_DIR}/train_2m.csv

# --- Model parameters (can be overridden via env vars when submitting) ---
K=${K:-4}
DIM=${DIM:-256}
EPOCHNUM=${EPOCHNUM:-1000}
LR=${LR:-0.001}
BATCH_SIZE=${BATCH_SIZE:-10000}
MAXREADNUM=${MAXREADNUM:-10000}
NEGSAMPLEPERPOS=${NEGSAMPLEPERPOS:-1000}
LOSS_NAME=${LOSS_NAME:-bern}
SEED=${SEED:-26042024}
CHECKPOINT=${CHECKPOINT:-0}
DEVICE=${DEVICE:-gpu}      # change to cpu if you really don't want GPU
WORKERS=${WORKERS:-1}

# --- Output path ---
OUTPUT_PATH=${BASEFOLDER}/models/revisit_k${K}_dim${DIM}_epoch=${EPOCHNUM}_LR=${LR}_batch=${BATCH_SIZE}.pt

# --- Build command ---
CMD="$PYTHON ${SCRIPT_PATH} --input ${INPUT_PATH} --k ${K} --dim ${DIM} --epoch ${EPOCHNUM} --lr ${LR}"
CMD="${CMD} --neg_sample_per_pos ${NEGSAMPLEPERPOS} --max_read_num ${MAXREADNUM}"
CMD="${CMD} --batch_size ${BATCH_SIZE} --loss_name ${LOSS_NAME}"
CMD="${CMD} --device ${DEVICE} --workers_num ${WORKERS} --output ${OUTPUT_PATH} --seed ${SEED} --checkpoint ${CHECKPOINT}"

echo "Running REVISIT training from: ${SCRIPT_PATH}"
echo "Using input data from:         ${INPUT_PATH}"
echo "Saving model to:               ${OUTPUT_PATH}"
echo "Job running on host:           $(hostname)"
echo "Start time:                    $(date)"
echo "CMD: ${CMD}"
echo

# --- Run inside the AI-LAB PyTorch container with GPU support ---
singularity exec --nv "${PYTORCH_CONTAINER}" ${CMD}

echo
echo "Finished at:                   $(date)"
