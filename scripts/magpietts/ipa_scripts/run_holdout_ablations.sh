#!/bin/bash
#
# Run ablation experiments: hold out one language at a time during training,
# then test on all languages.
#
# This script runs:
# 1. Baseline: train on all languages, test on all
# 2. For each language X: train on all languages EXCEPT X, test on all
#
# Results are stored in separate .txt files in the specified output directory.
#

set -e

# -------------------------
# CONFIGURATION
# -------------------------

# Base output directory for all experiments
BASE_OUTPUT_DIR="${1:-/home/shehzeenh/Code/DecoderNeMo/NeMo/scripts/magpietts/ipa_scripts/ablation_results}"

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/analyze_ipa_tokenization.py"

# All available languages
ALL_LANGS=("de" "es" "fr" "hi" "it" "vi" "zh" "en")

# Experiment parameters
SAMPLES_PER_LANG=1000
MAX_SAMPLES_PER_LANG=50000
SEED=42

# -------------------------
# HELPER FUNCTIONS
# -------------------------

# Get comma-separated list of all languages except the holdout
get_train_langs_except() {
    local holdout="$1"
    local result=""
    for lang in "${ALL_LANGS[@]}"; do
        if [[ "$lang" != "$holdout" ]]; then
            if [[ -n "$result" ]]; then
                result="${result},${lang}"
            else
                result="$lang"
            fi
        fi
    done
    echo "$result"
}

# Get comma-separated list of all languages
get_all_langs() {
    local IFS=','
    echo "${ALL_LANGS[*]}"
}

# Run a single experiment
run_experiment() {
    local exp_name="$1"
    local train_langs="$2"
    local test_langs="$3"
    local output_dir="$4"
    local log_file="$5"

    echo "============================================================"
    echo "EXPERIMENT: ${exp_name}"
    echo "  Train langs: ${train_langs}"
    echo "  Test langs:  ${test_langs}"
    echo "  Output dir:  ${output_dir}"
    echo "  Log file:    ${log_file}"
    echo "============================================================"

    # Create output directory
    mkdir -p "$output_dir"

    # Run the analysis script and capture output
    python "$ANALYSIS_SCRIPT" \
        --output_dir "$output_dir" \
        --train_langs "$train_langs" \
        --test_langs "$test_langs" \
        --samples_per_lang "$SAMPLES_PER_LANG" \
        --max_samples_per_lang "$MAX_SAMPLES_PER_LANG" \
        --seed "$SEED" \
        2>&1 | tee "$log_file"

    echo ""
    echo "Experiment ${exp_name} completed. Results saved to:"
    echo "  - Log: ${log_file}"
    echo "  - JSON: ${output_dir}/tokenization_comparison.json"
    echo ""
}

# -------------------------
# MAIN SCRIPT
# -------------------------

echo "============================================================"
echo "IPA TOKENIZER HOLDOUT ABLATION EXPERIMENTS"
echo "============================================================"
echo ""
echo "Base output directory: ${BASE_OUTPUT_DIR}"
echo "Languages: ${ALL_LANGS[*]}"
echo ""

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Create a summary file
SUMMARY_FILE="${BASE_OUTPUT_DIR}/experiment_summary.txt"
echo "IPA Tokenizer Holdout Ablation Experiments" > "$SUMMARY_FILE"
echo "===========================================" >> "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# -------------------------
# EXPERIMENT 0: BASELINE (train on all, test on all)
# -------------------------

EXP_NAME="baseline_all_langs"
TRAIN_LANGS=$(get_all_langs)
TEST_LANGS="all"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
LOG_FILE="${BASE_OUTPUT_DIR}/${EXP_NAME}.txt"

echo "[1/$(( ${#ALL_LANGS[@]} + 1 ))] Running baseline experiment (all languages)..."
run_experiment "$EXP_NAME" "$TRAIN_LANGS" "$TEST_LANGS" "$OUTPUT_DIR" "$LOG_FILE"

echo "Experiment: ${EXP_NAME}" >> "$SUMMARY_FILE"
echo "  Train: ${TRAIN_LANGS}" >> "$SUMMARY_FILE"
echo "  Test: ${TEST_LANGS}" >> "$SUMMARY_FILE"
echo "  Log: ${LOG_FILE}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# -------------------------
# EXPERIMENTS 1-N: HOLDOUT ABLATIONS
# -------------------------

exp_num=2
for holdout_lang in "${ALL_LANGS[@]}"; do
    EXP_NAME="holdout_${holdout_lang}"
    TRAIN_LANGS=$(get_train_langs_except "$holdout_lang")
    TEST_LANGS="all"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
    LOG_FILE="${BASE_OUTPUT_DIR}/${EXP_NAME}.txt"

    echo "[${exp_num}/$(( ${#ALL_LANGS[@]} + 1 ))] Running holdout experiment: excluding ${holdout_lang}..."
    run_experiment "$EXP_NAME" "$TRAIN_LANGS" "$TEST_LANGS" "$OUTPUT_DIR" "$LOG_FILE"

    echo "Experiment: ${EXP_NAME}" >> "$SUMMARY_FILE"
    echo "  Train: ${TRAIN_LANGS}" >> "$SUMMARY_FILE"
    echo "  Test: ${TEST_LANGS}" >> "$SUMMARY_FILE"
    echo "  Holdout: ${holdout_lang}" >> "$SUMMARY_FILE"
    echo "  Log: ${LOG_FILE}" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    exp_num=$((exp_num + 1))
done

# -------------------------
# FINISH
# -------------------------

echo "Completed: $(date)" >> "$SUMMARY_FILE"

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "============================================================"
echo ""
echo "Results directory: ${BASE_OUTPUT_DIR}"
echo ""
echo "Log files:"
echo "  - ${BASE_OUTPUT_DIR}/baseline_all_langs.txt"
for holdout_lang in "${ALL_LANGS[@]}"; do
    echo "  - ${BASE_OUTPUT_DIR}/holdout_${holdout_lang}.txt"
done
echo ""
echo "Summary: ${SUMMARY_FILE}"
echo ""
echo "To view a specific experiment's results:"
echo "  cat ${BASE_OUTPUT_DIR}/baseline_all_langs.txt"
echo "  cat ${BASE_OUTPUT_DIR}/holdout_en.txt"
echo "  # etc."
echo ""
