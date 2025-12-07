#!/usr/bin/bash -l
#SBATCH --job-name=OUR_TRAIN
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tolaso24@student.aau.dk

# --- AI-LAB container & project paths ---
# PyTorch container already available on AI-LAB
PYTORCH_CONTAINER=/ceph/container/pytorch/pytorch_25.10.sif

# Adjust this if your repo lives in a different folder
BASEFOLDER=/ceph/home/student.aau.dk/db56hw/revisitingkmers-p3

PYTHON="python"
SCRIPT_PATH=${BASEFOLDER}/src/OUR.py

# --- Data path on AI-LAB ---
# Using the same dataset location as your old scripts
DATA_DIR=/ceph/project/p3-kmer/dataset
INPUT_PATH=${DATA_DIR}/train_2m.csv

# --- Model parameters (can be overridden via env vars) ---
K=${K:-4}
DIM=${DIM:-256}
EPOCHNUM=${EPOCHNUM:-300}
LR=${LR:-0.001}
BATCH_SIZE=${BATCH_SIZE:-10000}
MAXREADNUM=${MAXREADNUM:-10000}
MAX_VIEWS_PER_READ=${MAX_VIEWS_PER_READ:-10}
TEMPERATURE=${TEMPERATURE:-0.1}
SEED=${SEED:-26042024}
CHECKPOINT=${CHECKPOINT:-0}
DEVICE=${DEVICE:-gpu}
WORKERS=${WORKERS:-1}

# --- Output path ---
OUTPUT_PATH=${BASEFOLDER}/models/our_k${K}_dim${DIM}_epoch=${EPOCHNUM}_LR=${LR}_batch=${BATCH_SIZE}.pt

# --- Build command ---
CMD="$PYTHON ${SCRIPT_PATH} --input ${INPUT_PATH} --k ${K} --dim ${DIM} --epoch ${EPOCHNUM} --lr ${LR}"
CMD="${CMD} --batch_size ${BATCH_SIZE} --max_read_num ${MAXREADNUM}"
CMD="${CMD} --max_views_per_read ${MAX_VIEWS_PER_READ} --temperature ${TEMPERATURE}"
CMD="${CMD} --device ${DEVICE} --workers_num ${WORKERS} --output ${OUTPUT_PATH} --seed ${SEED} --checkpoint ${CHECKPOINT}"

echo "Running training from:   ${SCRIPT_PATH}"
echo "Using input data from:   ${INPUT_PATH}"
echo "Saving model to:         ${OUTPUT_PATH}"
echo "Job running on host:     $(hostname)"
echo "Start time:              $(date)"
echo "CMD: ${CMD}"
echo

# --- Run inside the AI-LAB PyTorch container with GPU access ---
singularity exec --nv "${PYTORCH_CONTAINER}" ${CMD}

echo
echo "Finished at:             $(date)"
