#!/bin/bash
#
# Run add_ipa_to_lhotse_shards.py one language at a time.
# Logs the status to /home/shehzeenh/Code/DecoderNeMo/LhotseLogs/
# This avoids hangs that occur when running all languages at once.
#
# Usage:
#   ./run_ipa_by_language.sh
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/add_ipa_to_lhotse_shards.py"
LOG_DIR="/home/shehzeenh/Code/DecoderNeMo/LhotseLogs"

# Languages to process (in order)
# LANGUAGES=("de" "es" "fr" "hi" "it" "vi" "zh" "en")
LANGUAGES=("vi" "zh" "en")

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_DIR}/ipa_run_${TIMESTAMP}.log"

echo "=============================================="
echo "Starting IPA processing run at $(date)"
echo "Master log: ${MASTER_LOG}"
echo "=============================================="
echo "" | tee -a "${MASTER_LOG}"

# Track overall status
FAILED_LANGS=()
COMPLETED_LANGS=()

for lang in "${LANGUAGES[@]}"; do
    LANG_LOG="${LOG_DIR}/${lang}_ipa_${TIMESTAMP}.log"
    STATUS_FILE="${LOG_DIR}/${lang}_status.txt"
    
    echo "----------------------------------------------" | tee -a "${MASTER_LOG}"
    echo "[$(date)] Starting language: ${lang}" | tee -a "${MASTER_LOG}"
    echo "Log file: ${LANG_LOG}" | tee -a "${MASTER_LOG}"
    echo "----------------------------------------------" | tee -a "${MASTER_LOG}"
    
    # Mark as in-progress
    echo "in_progress" > "${STATUS_FILE}"
    
    # Run the Python script for this language
    if python3 "${PYTHON_SCRIPT}" --lang "${lang}" 2>&1 | tee "${LANG_LOG}"; then
        # Success
        echo "done" > "${STATUS_FILE}"
        echo "[$(date)] DONE: ${lang}" | tee -a "${MASTER_LOG}"
        COMPLETED_LANGS+=("${lang}")
    else
        # Failure
        echo "failed" > "${STATUS_FILE}"
        echo "[$(date)] FAILED: ${lang}" | tee -a "${MASTER_LOG}"
        FAILED_LANGS+=("${lang}")
    fi
    
    echo "" | tee -a "${MASTER_LOG}"
done

echo "=============================================="  | tee -a "${MASTER_LOG}"
echo "IPA processing run completed at $(date)" | tee -a "${MASTER_LOG}"
echo "=============================================="  | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"
echo "Completed languages: ${COMPLETED_LANGS[*]}" | tee -a "${MASTER_LOG}"
echo "Failed languages: ${FAILED_LANGS[*]}" | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"

# Exit with error if any language failed
if [ ${#FAILED_LANGS[@]} -gt 0 ]; then
    echo "Some languages failed. Check individual logs in ${LOG_DIR}" | tee -a "${MASTER_LOG}"
    exit 1
fi

echo "All languages processed successfully!" | tee -a "${MASTER_LOG}"
