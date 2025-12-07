#!/bin/bash

# Script to evaluate both REVISIT and OUR models

# Define the global variables
BASEFOLDER=$(pwd)
PYTHON="python"
SCRIPT_PATH=${BASEFOLDER}/binning.py

# Evaluation Parameters
DATA_DIR=${BASEFOLDER}
SPECIES="reference,marine,plant"
SAMPLES="5,6"
K=4
METRIC=""
SCALABLE=0
SUFFIX=""

# Model paths (update these to point to your trained models)
REVISIT_MODEL=${BASEFOLDER}/models/revisit_k4_dim256_epoch=1000_LR0.001_batch=10000.pt
OUR_MODEL=${BASEFOLDER}/models/our_k4_dim256_epoch=2_LR0.001_batch=10000.pt

# Output path
OUTPUT_PATH=${BASEFOLDER}/results/evaluation_$(date +%Y%m%d_%H%M%S).txt

# Create results directory if it doesn't exist
mkdir -p ${BASEFOLDER}/results

# Evaluate REVISIT model (nonlinear)
if [ -f "${REVISIT_MODEL}" ]; then
    echo "Evaluating REVISIT model: ${REVISIT_MODEL}"
    CMD="$PYTHON ${SCRIPT_PATH} --species ${SPECIES} --samples ${SAMPLES} --output ${OUTPUT_PATH}"
    CMD="${CMD} --test_model_dir ${REVISIT_MODEL} --model_list nonlinear"
    CMD="${CMD} --data_dir ${DATA_DIR} --k ${K}"
    [ -n "${METRIC}" ] && CMD="${CMD} --metric ${METRIC}"
    [ "${SCALABLE}" == "1" ] && CMD="${CMD} --scalable 1"
    [ -n "${SUFFIX}" ] && CMD="${CMD} --suffix ${SUFFIX}"
    echo ${OUTPUT_PATH}
    $CMD
else
    echo "Warning: REVISIT model not found at ${REVISIT_MODEL}"
fi

# Evaluate OUR model
if [ -f "${OUR_MODEL}" ]; then
    echo "Evaluating OUR model: ${OUR_MODEL}"
    CMD="$PYTHON ${SCRIPT_PATH} --species ${SPECIES} --samples ${SAMPLES} --output ${OUTPUT_PATH}"
    CMD="${CMD} --test_model_dir ${OUR_MODEL} --model_list our"
    CMD="${CMD} --data_dir ${DATA_DIR} --k ${K}"
    [ -n "${METRIC}" ] && CMD="${CMD} --metric ${METRIC}"
    [ "${SCALABLE}" == "1" ] && CMD="${CMD} --scalable 1"
    [ -n "${SUFFIX}" ] && CMD="${CMD} --suffix ${SUFFIX}"
    echo ${OUTPUT_PATH}
    $CMD
else
    echo "Warning: OUR model not found at ${OUR_MODEL}"
fi

echo "Evaluation completed. Results saved to: ${OUTPUT_PATH}"
