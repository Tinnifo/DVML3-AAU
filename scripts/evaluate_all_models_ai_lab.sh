#!/bin/bash
#
# Slurm job configuration (AI-LAB)
# Optimized to use all 256GB RAM available
#SBATCH --job-name=EVAL_ALL_MODELS
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=15
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# ---------------- AI-LAB container setup ----------------
# PyTorch container on AI-LAB (check available versions with: ls /ceph/container/pytorch)
PYTORCH_CONTAINER=${PYTORCH_CONTAINER:-/ceph/container/pytorch/pytorch_25.10.sif}

echo "==== AI-LAB container job ===="
echo "Container:         ${PYTORCH_CONTAINER}"
echo "GPU requested:     1"
echo "CPUs per task:     ${SLURM_CPUS_PER_TASK:-15}"
echo "Memory:            256G"
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
SCALABLE=${SCALABLE:-1}  # Default to 1 for better performance with high memory
SUFFIX=${SUFFIX:-""}

# --- Parallel execution settings ---
# With 256GB RAM, we can run many evaluations in parallel
# Each evaluation needs ~2-5GB RAM depending on dataset size
# Conservative estimate: 40 parallel jobs (40 * 5GB = 200GB, leaving 56GB buffer)
# Can be increased if needed, but 40 should be safe for most datasets
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-40}

# --- Output directory (separate files per model for parallel execution) ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${RESULTS_FOLDER}/evaluation_${TIMESTAMP}
mkdir -p "${OUTPUT_DIR}"
SUMMARY_FILE="${RESULTS_FOLDER}/evaluation_summary_${TIMESTAMP}.txt"

echo "==== Evaluation job starting ===="
echo "Host (Slurm node): $(hostname)"
echo "Start time:         $(date)"
echo "Base folder:        ${BASEFOLDER}"
echo "Script path:        ${SCRIPT_PATH}"
echo "Data dir:           ${DATA_DIR}"
echo "Results dir:        ${RESULTS_FOLDER}"
echo "Results directory:  ${OUTPUT_DIR}"
echo "Summary file:       ${SUMMARY_FILE}"
echo "Models directory:   ${BASEFOLDER}/models"
echo "Max parallel jobs:  ${MAX_PARALLEL_JOBS}"
echo "Scalable mode:      ${SCALABLE}"
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

# Function to evaluate a single model (for parallel execution)
evaluate_model() {
    local MODEL_PATH="$1"
    local MODEL_NAME=$(basename "$MODEL_PATH")
    local JOB_ID="$2"
    
    # Extract k value from filename
    local K_VALUE=4
    if [[ $MODEL_NAME =~ our_k([0-9]+)_ ]]; then
        K_VALUE="${BASH_REMATCH[1]}"
    fi
    
    # Create unique output file for this model
    local MODEL_OUTPUT="${OUTPUT_DIR}/${MODEL_NAME//.pt/.txt}"
    
    echo "[Job ${JOB_ID}] Starting: ${MODEL_NAME} (k=${K_VALUE})"
    
    # Build python command (inside container)
    local PY_CMD="${PYTHON} ${SCRIPT_PATH} --species ${SPECIES} --samples ${SAMPLES} --output ${MODEL_OUTPUT}"
    PY_CMD="${PY_CMD} --test_model_dir ${MODEL_PATH} --model_list our"
    PY_CMD="${PY_CMD} --data_dir ${DATA_DIR} --k ${K_VALUE}"
    [ -n "${METRIC}" ] && PY_CMD="${PY_CMD} --metric ${METRIC}"
    [ "${SCALABLE}" == "1" ] && PY_CMD="${PY_CMD} --scalable 1"
    [ -n "${SUFFIX}" ] && PY_CMD="${PY_CMD} --suffix ${SUFFIX}"

    # Full command with Singularity
    local CMD="singularity exec --nv ${PYTORCH_CONTAINER} ${PY_CMD}"
    
    # Run evaluation and capture exit code
    if eval "${CMD}" > "${MODEL_OUTPUT}.log" 2>&1; then
        echo "[Job ${JOB_ID}] ✓ Completed: ${MODEL_NAME}"
        echo "${MODEL_NAME}: SUCCESS" >> "${SUMMARY_FILE}"
        return 0
    else
        echo "[Job ${JOB_ID}] ✗ Failed: ${MODEL_NAME}"
        echo "${MODEL_NAME}: FAILED (check ${MODEL_OUTPUT}.log)" >> "${SUMMARY_FILE}"
        return 1
    fi
}

# Export function and variables for parallel execution
export -f evaluate_model
export BASEFOLDER SCRIPT_PATH DATA_DIR SPECIES SAMPLES METRIC SCALABLE SUFFIX OUTPUT_DIR SUMMARY_FILE PYTHON PYTORCH_CONTAINER

# Initialize summary file
echo "Evaluation Summary - Started: $(date)" > "${SUMMARY_FILE}"
echo "Total models: ${MODEL_COUNT}" >> "${SUMMARY_FILE}"
echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}" >> "${SUMMARY_FILE}"
echo "Memory allocated: 256GB" >> "${SUMMARY_FILE}"
echo "========================================" >> "${SUMMARY_FILE}"
echo >> "${SUMMARY_FILE}"

# Check if GNU parallel is available (preferred for better output handling)
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job distribution..."
    echo "$MODEL_FILES" | parallel -j ${MAX_PARALLEL_JOBS} --tag evaluate_model {} {#}
    PARALLEL_EXIT=$?
elif command -v xargs &> /dev/null; then
    echo "Using xargs for parallel execution..."
    # Use xargs with parallel execution
    JOB_ID=0
    echo "$MODEL_FILES" | while IFS= read -r MODEL_PATH; do
        JOB_ID=$((JOB_ID + 1))
        echo "$MODEL_PATH $JOB_ID"
    done | xargs -n 2 -P ${MAX_PARALLEL_JOBS} -I {} bash -c 'evaluate_model $(echo {} | cut -d" " -f1) $(echo {} | cut -d" " -f2)'
    PARALLEL_EXIT=$?
else
    echo "Warning: Neither GNU parallel nor xargs found. Falling back to sequential execution."
    echo "Install GNU parallel for better performance: module load parallel (if available)"
    
    JOB_ID=0
    while IFS= read -r MODEL_PATH; do
        JOB_ID=$((JOB_ID + 1))
        evaluate_model "$MODEL_PATH" "$JOB_ID"
    done <<< "$MODEL_FILES"
    PARALLEL_EXIT=$?
fi

# Count successes and failures
SUCCESS_COUNT=$(grep -c "SUCCESS" "${SUMMARY_FILE}" 2>/dev/null || echo "0")
FAILED_COUNT=$(grep -c "FAILED" "${SUMMARY_FILE}" 2>/dev/null || echo "0")

echo
echo "=========================================="
echo "Evaluation completed!"
echo "Total models: ${MODEL_COUNT}"
echo "Successful: ${SUCCESS_COUNT}"
echo "Failed: ${FAILED_COUNT}"
echo "Results directory: ${OUTPUT_DIR}"
echo "Summary file: ${SUMMARY_FILE}"
echo "End time: $(date)"
echo "=========================================="

# Append summary to summary file
echo >> "${SUMMARY_FILE}"
echo "========================================" >> "${SUMMARY_FILE}"
echo "Completed: $(date)" >> "${SUMMARY_FILE}"
echo "Successful: ${SUCCESS_COUNT}" >> "${SUMMARY_FILE}"
echo "Failed: ${FAILED_COUNT}" >> "${SUMMARY_FILE}"

# Exit with error if any jobs failed
if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
else
    exit 0
fi
