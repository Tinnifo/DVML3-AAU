#!/bin/bash
#
# Slurm job configuration (AI-LAB)
#SBATCH --job-name=EVAL_ALL_MODELS
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# ---------------- AI-LAB container setup ----------------
# PyTorch container on AI-LAB (check available versions with: ls /ceph/container/pytorch)
PYTORCH_CONTAINER=${PYTORCH_CONTAINER:-/ceph/container/pytorch/pytorch_25.10.sif}

echo "==== AI-LAB container job ===="
echo "Container:         ${PYTORCH_CONTAINER}"
echo "GPU requested:     1"
echo "CPUs per task:     ${SLURM_CPUS_PER_TASK:-15}"
echo "Memory:            24G"
echo

# ---------------- Original script logic -----------------
# Script to evaluate all models in the models/ directory
# This script finds all .pt model files and runs evaluation on each one

# --- Setup paths ---
# BASEFOLDER = repo root (one level above this script)
BASEFOLDER=${BASEFOLDER:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python3}

# Try to locate binning.py either in repo root or in evaluation/
if [ -f "${BASEFOLDER}/binning.py" ]; then
    SCRIPT_PATH=${BASEFOLDER}/binning.py
elif [ -f "${BASEFOLDER}/evaluation/binning.py" ]; then
    SCRIPT_PATH=${BASEFOLDER}/evaluation/binning.py
else
    echo "Error: binning.py not found in ${BASEFOLDER} or ${BASEFOLDER}/evaluation"
    exit 1
fi

# --- Data path ---
# Default to the shared dataset on AI-LAB, can still be overridden via env DATA_DIR
DATA_DIR=${DATA_DIR:-/ceph/project/p3-kmer/dataset}

# --- Results folder ---
RESULTS_FOLDER=${BASEFOLDER}/results
mkdir -p "${RESULTS_FOLDER}"

# --- Evaluation parameters (can be overridden via env vars) ---
SPECIES=${SPECIES:-"reference,marine,plant"}
SAMPLES=${SAMPLES:-"5,6"}
METRIC=${METRIC:-""}
SCALABLE=${SCALABLE:-0}
SUFFIX=${SUFFIX:-""}

# --- Output file ---
OUTPUT_PATH=${RESULTS_FOLDER}/evaluation_all_models_$(date +%Y%m%d_%H%M%S).txt

echo "==== Evaluation job starting ===="
echo "Host (Slurm node): $(hostname)"
echo "Start time:         $(date)"
echo "Base folder:        ${BASEFOLDER}"
echo "Script path:        ${SCRIPT_PATH}"
echo "Data dir:           ${DATA_DIR}"
echo "Results dir:        ${RESULTS_FOLDER}"
echo "Results file:       ${OUTPUT_PATH}"
echo "Models directory:   ${BASEFOLDER}/models"
echo

# Find all .pt model files
MODEL_FILES=$(find "${BASEFOLDER}/models" -name "*.pt" -type f | sort)

if [ -z "$MODEL_FILES" ]; then
    echo "Error: No .pt model files found in ${BASEFOLDER}/models"
    exit 1
fi

MODEL_COUNT=$(echo "$MODEL_FILES" | wc -l | tr -d ' ')
echo "Found ${MODEL_COUNT} model(s) to evaluate"
echo

# Counter for tracking progress
CURRENT=0

# Loop through each model file
while IFS= read -r MODEL_PATH; do
    CURRENT=$((CURRENT + 1))
    MODEL_NAME=$(basename "$MODEL_PATH")
    
    echo "=========================================="
    echo "[${CURRENT}/${MODEL_COUNT}] Evaluating: ${MODEL_NAME}"
    echo "=========================================="
    
    # Extract k value from filename
    # Pattern: our_k3_dim128_... or our_k4_dim256_...
    if [[ $MODEL_NAME =~ our_k([0-9]+)_ ]]; then
        K_VALUE="${BASH_REMATCH[1]}"
    else
        echo "Warning: Could not extract k value from ${MODEL_NAME}, defaulting to k=4"
        K_VALUE=4
    fi
    
    echo "Model path: ${MODEL_PATH}"
    echo "K value:    ${K_VALUE}"
    echo

    # ---------------- Run inside Singularity container ----------------
    # Build python command (inside container)
    PY_CMD="${PYTHON} ${SCRIPT_PATH} --species ${SPECIES} --samples ${SAMPLES} --output ${OUTPUT_PATH}"
    PY_CMD="${PY_CMD} --test_model_dir ${MODEL_PATH} --model_list our"
    PY_CMD="${PY_CMD} --data_dir ${DATA_DIR} --k ${K_VALUE}"
    [ -n "${METRIC}" ] && PY_CMD="${PY_CMD} --metric ${METRIC}"
    [ "${SCALABLE}" == "1" ] && PY_CMD="${PY_CMD} --scalable 1"
    [ -n "${SUFFIX}" ] && PY_CMD="${PY_CMD} --suffix ${SUFFIX}"

    # Full command with Singularity
    CMD="singularity exec --nv ${PYTORCH_CONTAINER} ${PY_CMD}"
    
    echo "Running in container:"
    echo "  ${CMD}"
    echo
    
    # Run evaluation
    if eval "${CMD}"; then
        echo "✓ Successfully evaluated ${MODEL_NAME}"
    else
        echo "✗ Error evaluating ${MODEL_NAME}"
    fi
    
    echo
    echo
    
done <<< "$MODEL_FILES"

echo "=========================================="
echo "Evaluation completed!"
echo "Total models evaluated: ${MODEL_COUNT}"
echo "Results saved to: ${OUTPUT_PATH}"
echo "End time: $(date)"
echo "=========================================="
