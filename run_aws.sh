#!/usr/bin/env bash
set -e

echo "========================================================================"
echo "    ML CHALLENGE 2026: AWS HIGH-SPEED INFERENCE & SUBMISSION BUILDER    "
echo "========================================================================"

# 0. Restore test dataset if needed
if [ ! -d "dataset/test" ] && ls dataset/test_chunk_* 1> /dev/null 2>&1; then
    echo ""
    echo "[Step 0/4] Restoring test dataset from GitHub chunks..."
    cat dataset/test_chunk_* > test_data.tar.gz
    tar -xzf test_data.tar.gz
    rm test_data.tar.gz
    echo "Test dataset successfully restored!"
fi

# 1. Install dependencies
echo ""
echo "[Step 1/4] Installing required dependencies..."
pip install -r code/business_entity_resolution/requirements.txt

# 2. Run High-Speed Multi-Core Inference
echo ""
echo "[Step 2/4] Executing multi-core test inference on AWS..."
python3 code/business_entity_resolution/src/inference.py --test-dir dataset/test --output-dir output

# 3. Validate Submission Outputs
echo ""
echo "[Step 3/4] Running official submission validator..."
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids

# 4. Build Final Submission Archive
echo ""
echo "[Step 4/4] Assembling final submission package..."
TEAM_NAME="Deciders"
ZIP_NAME="${TEAM_NAME}_submission.zip"
rm -f "$ZIP_NAME"

zip -r "$ZIP_NAME" \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/business_entity_resolution/ \
    Documentation_template.md

echo ""
echo "========================================================================"
echo "SUCCESS! SUBMISSION PACKAGE CREATED: $ZIP_NAME"
echo "You can now upload $ZIP_NAME or output/matching_results.tsv to the Portal!"
echo "========================================================================"
