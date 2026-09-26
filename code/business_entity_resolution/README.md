# Business Entity Resolution Solution

## ML Challenge 2026

This package contains the complete, self-contained, reproducible pipeline for the Business Entity Resolution Challenge.

### System Architecture

1. **Linguistic Normalization (`preprocess.py`)**:
   - Strips legal suffixes across US (`Inc`, `Corp`, `LLC`), India (`Pvt Ltd`), and France (`SARL`, `SAS`, `SA`).
   - Normalizes roadway abbreviations and extracts numeric building/postal PIN tokens.
   - Converts multilingual Indic scripts (Devanagari, Telugu, Malayalam, Tamil) to Roman phonetic skeletons.
2. **High-Recall Multi-Index Blocking (`blocking.py`)**:
   - Strict country partitioning (US, India, France).
   - Inverted hash tables over: Exact Clean Names, Compact (No-Space) Names, Distinctive Tokens, and Phonetic Skeletons.
   - Produces `output/candidate_pairs.tsv`.
3. **Pairwise Feature Engineering (`features.py`)**:
   - RapidFuzz C++ vectorized similarity metrics (Jaro-Winkler, Levenshtein, Token Sort, Token Set).
   - Address numeric overlap and token Jaccard scores.
4. **LightGBM Calibrated Classifier & Singleton Gating (`train_model.py`)**:
   - Trained on hard negative candidate pairs.
   - Optimal threshold $\theta^* = 0.50$ and Singleton Gate $\tau = 0.40$ tuned to maximize official Macro-$F_{0.5}$.
5. **Full Test Inference (`inference.py`)**:
   - Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

---

### End-to-End Reproduction Instructions

#### 1. Setup Environment

```bash
pip install -r requirements.txt
```

#### 2. Train Model and Optimize Thresholds

```bash
python3 code/business_entity_resolution/src/train_model.py
```

Outputs `model.joblib` and `model_config.json`.

#### 3. Run Inference on Test Set

```bash
python3 code/business_entity_resolution/src/inference.py --test-dir dataset/test --output-dir output
```

Generates:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

#### 4. Validate Submission

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

Prints `PASS — no blocking issues found. Safe to submit.`
