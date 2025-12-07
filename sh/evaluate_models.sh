#!/usr/bin/bash -l
#SBATCH --job-name=EVAL_MODELS
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tolaso24@student.aau.dk

# --- AI-LAB container & project paths ---
PYTORCH_CONTAINER=/ceph/container/pytorch/pytorch_25.10.sif

# Adjust this if your repo is elsewhere
BASEFOLDER=/ceph/home/student.aau.dk/db56hw/revisitingkmers-p3

PYTHON="python"
# In the original AI-LAB scripts binning.py lives in evaluation/
SCRIPT_PATH=${BASEFOLDER}/evaluation/binning.py

# --- Data path on AI-LAB (where the datasets live) ---
DATA_DIR=/ceph/project/p3-kmer/dataset

# --- Results folder ---
RESULTS_FOLDER=${BASEFOLDER}/results
mkdir -p "${RESULTS_FOLDER}"

# --- Evaluation parameters (can be overridden via env vars) ---
SPECIES=${SPECIES:-"reference,marine,plant"}
SAMPLES=${SAMPLES:-"5,6"}
K=${K:-4}
METRIC=${METRIC:-""}
SCALABLE=${SCALABLE:-0}
SUFFIX=${SUFFIX:-""}

# --- Model paths (defaults match the training scripts' naming) ---
REVISIT_MODEL=${REVISIT_MODEL:-${BASEFOLDER}/models/revisit_k4_dim256_epoch=1000_LR=0.001_batch=10000.pt}
OUR_MODEL=${OUR_MODEL:-${BASEFOLDER}/models/our_k4_dim256_epoch=300_LR=0.001_batch=10000.pt}

# --- Output file ---
OUTPUT_PATH=${RESULTS_FOLDER}/evaluation_$(date +%Y%m%d_%H%M%S).txt

echo "==== Evaluation job starting ===="
echo "Host:               $(hostname)"
echo "Start time:         $(date)"
echo "Script path:        ${SCRIPT_PATH}"
echo "Data dir:           ${DATA_DIR}"
echo "Results file:       ${OUTPUT_PATH}"
echo "Revisit model:      ${REVISIT_MODEL}"
echo "Our model:          ${OUR_MODEL}"
echo

# ---------- Evaluate REVISIT model (nonlinear) ----------
if [ -f "${REVISIT_MODEL}" ]; then
    echo "Evaluating REVISIT model: ${REVISIT_MODEL}"

    CMD="$PYTHON ${SCRIPT_PATH} --species ${SPECIES} --samples ${SAMPLES} --output ${OUTPUT_PATH}"
    CMD="${CMD} --test_model_dir ${REVISIT_MODEL} --model_list nonlinear"
    CMD="${CMD} --data_dir ${DATA_DIR} --k ${K}"
    [ -n "${METRIC}" ] && CMD="${CMD} --metric ${METRIC}"
    [ "${SCALABLE}" == "1" ] && CMD="${CMD} --scalable 1"
    [ -n "${SUFFIX}" ] && CMD="${CMD} --suffix ${SUFFIX}"

    echo "CMD (REVISIT): ${CMD}"
    singularity exec "${PYTORCH_CONTAINER}" ${CMD}
    echo "Finished REVISIT evaluation."
else
    echo "Warning: REVISIT model not found at ${REVISIT_MODEL}"
fi

echo

# ---------- Evaluate OUR model ----------
if [ -f "${OUR_MODEL}" ]; then
    echo "Evaluating OUR model: ${OUR_MODEL}"

    CMD="$PYTHON ${SCRIPT_PATH} --species ${SPECIES} --samples ${SAMPLES} --output ${OUTPUT_PATH}"
    CMD="${CMD} --test_model_dir ${OUR_MODEL} --model_list our"
    CMD="${CMD} --data_dir ${DATA_DIR} --k ${K}"
    [ -n "${METRIC}" ] && CMD="${CMD} --metric ${METRIC}"
    [ "${SCALABLE}" == "1" ] && CMD="${CMD} --scalable 1"
    [ -n "${SUFFIX}" ] && CMD="${CMD} --suffix ${SUFFIX}"

    echo "CMD (OUR): ${CMD}"
    singularity exec "${PYTORCH_CONTAINER}" ${CMD}
    echo "Finished OUR evaluation."
else
    echo "Warning: OUR model not found at ${OUR_MODEL}"
fi

echo
echo "Evaluation completed. Results saved to: ${OUTPUT_PATH}"
echo "End time:           $(date)"
echo "==============================="
