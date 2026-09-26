# AGENTS.md — Business Entity Resolution: Master Agent Guide

> **Team:** Deciders · **Competition:** ML Challenge 2026 — Business Entity Resolution
> **Authority:** Single Source of Truth for AI coding agents working on this codebase.
> All competition rules, data invariants, architectural decisions, code structure, and
> agent behavioral guidelines live here. No other markdown documents supersede this file.

---

## Table of Contents

1. [Competition Identity](#1-competition-identity)
2. [Official Objective](#2-official-objective)
3. [Dataset Schemas & Sizes](#3-dataset-schemas--sizes)
4. [Ground Truth Invariants](#4-ground-truth-invariants)
5. [Evaluation Metric — Macro F0.5](#5-evaluation-metric--macro-f05)
6. [Output File Specifications](#6-output-file-specifications)
7. [Pipeline Architecture](#7-pipeline-architecture)
8. [Codebase Map](#8-codebase-map)
9. [Key Algorithms & Design Decisions](#9-key-algorithms--design-decisions)
10. [Feature Engineering](#10-feature-engineering)
11. [Model & Threshold Configuration](#11-model--threshold-configuration)
12. [Validation Strategy](#12-validation-strategy)
13. [Submission Rules & Checklist](#13-submission-rules--checklist)
14. [Error Analysis Reference](#14-error-analysis-reference)
15. [Experiment Tracking Schema](#15-experiment-tracking-schema)
16. [Agent Behavioral Guidelines](#16-agent-behavioral-guidelines)

---

## 1. Competition Identity

| Field | Value |
|---|---|
| **Title** | ML Challenge 2026 — Business Entity Resolution Challenge |
| **Domain** | Scaled Commercial Entity Resolution / Record Linkage |
| **Format** | 72-Hour Hackathon |
| **Team** | Deciders |
| **Leaderboard File** | `output/matching_results.tsv` |
| **Final Package** | `Deciders_submission.zip` |

---

## 2. Official Objective

Given business records from **3 independent, noisy commercial data sources** with no shared identifiers, determine which records across sources refer to the same real-world business entity.

- **Source 1** is the deduplicated reference (anchor). Every real-world entity appears at most once in S1.
- **Task**: For each S1 entity, find all matching records from S2 ∪ S3.
- **Cardinality**: An S1 entity may match 0, 1, or many S2/S3 records.
- **Metric**: Maximize **Macro-Averaged F₀.₅** across all S1 entities in the private test set.

### Critical Verified Invariants (from empirical ground truth analysis)

| Invariant | Verified Value |
|---|---|
| Cross-country matches | **Strictly 0** — entities only link within the same `country` |
| Target uniqueness | **Strictly 0** — an S2/S3 record belongs to at most 1 S1 entity |
| S1 self-matches | **Strictly forbidden** — predicting an `S1-` ID causes validation rejection |
| Singleton rate | **5.5848%** of S1 entities have zero true matches |
| Match counts per S1 entity | Min=1, Median=4, Mean=3.666, Max=11 |

---

## 3. Dataset Schemas & Sizes

### 3.1 Source File Schema

All files are **Tab-Separated (TSV), UTF-8 encoded**. Never use comma as delimiter.

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `entity_id` | String | NO | Prefix: `S1-`, `S2-`, `S3-` |
| `business_name` | String | NO | Contains abbreviations, typos, transliterations |
| `business_address` | String | YES | Missing in ~3.4% of S2/S3 records |
| `country` | String | NO | Open-set: `US`, `India` in train; + `France` in test |

**Read with**: `pd.read_csv(file, sep="\t")` — never omit `sep="\t"`.

### 3.2 Ground Truth Schema

| Column | Description |
|---|---|
| `source1_entity_id` | S1 entity ID |
| `matched_entity_ids` | Comma-separated S2/S3 IDs, or empty string for singletons |

### 3.3 Dataset Sizes

| File | Records | Country Split |
|---|---|---|
| `train/train_source1.tsv` | 2,206,821 | US: 60%, India: 40% |
| `train/train_source2.tsv` | 5,034,616 | US: 59.9%, India: 40.1% |
| `train/train_source3.tsv` | 5,285,603 | US: 60%, India: 40% |
| `test/test_source1.tsv` | **1,732,544** | India: 46.8%, US: 38.3%, France: 15.0% |
| `test/test_source2.tsv` | 4,887,273 | India: 47.3%, US: 38.3%, France: 14.4% |
| `test/test_source3.tsv` | 5,082,316 | India: 47.3%, US: 38.3%, France: 14.4% |

> **Critical**: Every one of the 1,732,544 `test_source1` entities **must** appear exactly once in `matching_results.tsv`.

### 3.4 Distractor Rate

- S2 + S3 train total = 10,320,219 records
- Matched in ground truth = 7,638,365 (~74%)
- **~26% of S2/S3 records are distractors** (match no S1 entity)

---

## 4. Ground Truth Invariants

```
Singletons (|T_i| = 0):     123,247 entities   (5.5848%)
Matched entities:          2,083,574 entities  (94.4152%)
Match count distribution:  3 matches=530,841 | 4 matches=484,115 | 2 matches=375,212
                           5 matches=321,957 | 6 matches=164,868 | 1 match=119,157
```

---

## 5. Evaluation Metric — Macro F0.5

### Formula

**Case A: Non-singleton (|T_i| > 0)**

```
TP_i       = |P_i ∩ T_i|
Precision_i = TP_i / |P_i|
Recall_i    = TP_i / |T_i|
F0.5_i      = 1.25 × Precision × Recall / (0.25 × Precision + Recall)
```
Returns 0.0 if TP_i = 0 or |P_i| = 0.

**Case B: Singleton (|T_i| = 0)**

```
F0.5_i = 1.0  if |P_i| = 0   (correct empty prediction)
F0.5_i = 0.0  if |P_i| > 0   (any false guess = 0 score)
```

**Macro Average**: `(1/N) × Σ F0.5_i` where `N = 1,732,544`

### Authoritative Python Implementation

```python
def compute_entity_f05(y_true_set: set, y_pred_set: set) -> float:
    if len(y_true_set) == 0:
        return 1.0 if len(y_pred_set) == 0 else 0.0
    if len(y_pred_set) == 0:
        return 0.0
    tp = len(y_true_set & y_pred_set)
    if tp == 0:
        return 0.0
    precision = tp / len(y_pred_set)
    recall = tp / len(y_true_set)
    return float((1.25 * precision * recall) / (0.25 * precision + recall))

def compute_macro_f05(ground_truth_dict, predictions_dict, all_s1_ids) -> float:
    return sum(
        compute_entity_f05(ground_truth_dict.get(s1, set()), predictions_dict.get(s1, set()))
        for s1 in all_s1_ids
    ) / len(all_s1_ids)
```

### Precision Dominance — Key Design Driver

`∂F0.5/∂Precision = 2 × ∂F0.5/∂Recall`

A false positive costs **2× more** than a false negative.

| Prediction for |T_i|=2 | Precision | Recall | F0.5 | Loss |
|---|---|---|---|---|
| 2 correct | 1.0 | 1.0 | **1.000** | — |
| 2 correct + 1 wrong | 0.667 | 1.0 | **0.714** | −0.286 |
| 1 correct + 0 wrong | 1.0 | 0.5 | **0.833** | −0.167 |

**→ When uncertain, do NOT predict a match.**

---

## 6. Output File Specifications

### 6.1 `output/matching_results.tsv` (Leaderboard File)

- **Header**: `source1_entity_id\tmatched_entity_ids`
- **Rows**: Exactly **1,732,544** data rows + 1 header = 1,732,545 lines
- **Singleton**: `S1-00003\t\n` (empty second column, never absent row)
- **Multi-match**: `S1-00001\tS2-00047,S2-00193,S3-00812\n` (no spaces, no brackets)
- **Encoding**: UTF-8, no BOM

### 6.2 `output/candidate_pairs.tsv` (Audit File)

- **Header**: `source1_entity_id\tcandidate_entity_ids`
- **Rows**: Exactly **1,732,544** data rows + 1 header = 1,732,545 lines
- **Constraint**: Every ID in `matching_results.tsv` must also appear here

### 6.3 Example

```
# matching_results.tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
S1-00004	S2-99120

# candidate_pairs.tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999,S2-00441
S1-00002	S3-00004,S2-11002
S1-00003	
S1-00004	S2-99120,S3-00118
```

---

## 7. Pipeline Architecture

```
[ Raw S1, S2, S3 Records ]
           |
           v
+----------------------------------------+
| 1. PREPROCESSING (preprocess.py)       |
|    - UTF-8 NFKC normalization          |
|    - Lowercasing + punctuation strip   |
|    - Legal suffix removal (US/IN/FR)   |
|    - Phonetic consonant skeleton       |
|    - Address token extraction          |
+----------------------------------------+
           |
           v
+----------------------------------------+
| 2. BLOCKING (blocking.py)              |
|    - Hard country partition            |
|    - 6-channel inverted index union:   |
|      Ch1:  Exact clean name            |
|      Ch1B: Compact (no-space) name     |
|      Ch1C: 4-char prefix               |
|      Ch2:  Distinctive token (len>=4)  |
|      Ch3:  First + second brand word   |
|      Ch4:  Phonetic skeleton           |
|      Ch5:  Address number block        |
|      Ch6:  Character trigram           |
|    - Output: candidate_pairs.tsv       |
|    - Target: Candidate Recall > 87%    |
|    - Avg candidates per S1: ~31.5      |
+----------------------------------------+
           |
           v
+----------------------------------------+
| 3. FEATURE EXTRACTION (features.py)   |
|    - 18 pairwise RapidFuzz features   |
|    - Name + Address + Metadata        |
+----------------------------------------+
           |
           v
+----------------------------------------+
| 4. SCORING (inference.py)             |
|    - LightGBM predict_proba           |
|    - Multi-threaded batch scoring     |
+----------------------------------------+
           |
           v
+----------------------------------------+
| 5. GLOBAL RESOLUTION (inference.py)   |
|    - Singleton gate: max_prob <        |
|      singleton_cutoff => emit empty   |
|    - Rule override: exact clean+      |
|      compact name => force match      |
|    - Target uniqueness enforcement    |
|    - Greedy assignment by confidence  |
+----------------------------------------+
           |
           v
+----------------------------------------+
| 6. THRESHOLD OPTIMIZATION             |
|    - Grid-search on Macro F0.5        |
|    - Current: theta=0.50, tau=0.40    |
|    - Output: matching_results.tsv     |
+----------------------------------------+
           |
           v
+----------------------------------------+
| 7. VALIDATION (validate_submission.py)|
|    - Must exit code 0 before submit   |
+----------------------------------------+
```

---

## 8. Codebase Map

```
student_resource/
|-- AGENTS.md                        <- YOU ARE HERE (Single Source of Truth)
|-- README.md                        <- Official problem statement (do not edit)
|-- run_aws.sh                       <- 1-click AWS SageMaker/EC2 runner
|-- .gitignore
|
|-- code/business_entity_resolution/
|   |-- requirements.txt             <- Pinned Python deps
|   |-- README.md                    <- Reproduction instructions
|   `-- src/
|       |-- preprocess.py            <- Text normalization + phonetic skeleton
|       |-- blocking.py              <- MultiIndexBlocker (6 channels)
|       |-- features.py              <- 18 pairwise RapidFuzz features
|       |-- train_model.py           <- LightGBM training + F0.5 threshold optimizer
|       |-- inference.py             <- Production batch inference engine
|       |-- evaluation.py            <- Macro F0.5 evaluator
|       |-- make_val_split.py        <- Leak-free holdout generator
|       |-- test_blocking.py         <- Blocking recall validation
|       |-- model.joblib             <- Trained LightGBM weights (Val F0.5: 0.8592)
|       `-- model_config.json        <- Thresholds: match=0.50, singleton=0.40
|
|-- dataset/
|   |-- train/                       <- train_source1/2/3.tsv + ground_truth.tsv
|   |-- test/                        <- test_source1/2/3.tsv (US + India + France)
|   |-- test_chunk_aa...ai           <- 60MB GitHub-compatible compressed chunks
|   `-- sample_submission.tsv
|
|-- validation/
|   |-- val_source1.tsv              <- 20,000 S1 entities holdout
|   |-- val_source2.tsv              <- 42,754 S2 auxiliary records
|   |-- val_source3.tsv              <- 43,754 S3 auxiliary records
|   `-- val_ground_truth.tsv         <- Ground truth labels
|
|-- output/
|   |-- matching_results.tsv         <- Leaderboard submission (scored)
|   |-- candidate_pairs.tsv          <- Blocking candidates (audit)
|   `-- Deciders_submission.zip      <- Final portal upload
|
`-- utils/
    `-- validate_submission.py       <- Official schema + integrity validator
```

### Import Pattern

```python
from code.business_entity_resolution.src.preprocess import clean_business_name
from code.business_entity_resolution.src.blocking import MultiIndexBlocker
from code.business_entity_resolution.src.features import extract_pair_features
```

Run all scripts from **`student_resource/`** as working directory.

---

## 9. Key Algorithms & Design Decisions

### 9.1 Preprocessing (`preprocess.py`)

- **Unicode**: NFKC normalization before everything
- **Legal suffix stripping** (language-aware):
  - US: `inc`, `corp`, `corporation`, `llc`, `llp`, `ltd`, `limited`, `co`, `company`
  - India: `pvt`, `private`, `ltd`, `limited`, `llp`
  - France: `sarl`, `sas`, `sa`, `eurl`, `snc`, `sci`, `ei`
- **Phonetic skeleton**: Strip vowels + collapse doubled consonants.
  - `"Premier"` → `"prmr"`, Devanagari `"प्रीमियर"` → `"prmr"` (zero-shot cross-script)
- **Address tokenization**: Split into (word_tokens, numeric_tokens) separately

### 9.2 Blocking — 6-Channel MultiIndexBlocker (`blocking.py`)

| Channel | Key | Guard |
|---|---|---|
| Ch1: Exact clean name | `(country, clean_name)` | — |
| Ch1B: Compact name | `(country, compact_name)` | len >= 4 |
| Ch1C: 4-char prefix | `(country, prefix4)` | shares >= 1 token |
| Ch2: Distinctive token | `(country, token)` | len>=4, not stopword; 2+ hits OR addr number overlap |
| Ch3: First/second brand word | `(country, word)` | len>=4, not stopword |
| Ch4: Phonetic skeleton | `(country, skel)` | len>=3; addr nums don't conflict |
| Ch5: Address number | `(country, "#"+num)` | also shares name token or first word |
| Ch6: Char trigram | `(country, trigram)` | compact name len>=5; needs >= 2 shared trigrams |

**Posting list cap**: `max_posting=250` to prevent memory thrashing.
**Candidate pruning**: If >35 candidates, rank by `exact(10)+compact(5)+skel(3)+num_overlap×2+tok_overlap`.
**Performance**: ~31.5 candidates/entity, 87.50% candidate recall on validation.

### 9.3 Inference Strategy (`inference.py`)

- **Country-by-country**: France → US → India (each loaded independently)
- **Batch size**: 40,000 S1 entities per batch
- **Multi-threading**: `Parallel(n_jobs=min(cpu_count,8), prefer="threads")` for feature extraction
- **Singleton gate**: `max_prob < singleton_cutoff(0.40)` → emit empty list
- **Rule override**: exact `clean_name` AND `compact_name` match (addr nums don't conflict) → force match regardless of model score
- **Target uniqueness**: Global `country_assigned_aux` set; once assigned, skip for all subsequent S1 in batch

---

## 10. Feature Engineering

### 18 Pairwise Features (`features.py` via RapidFuzz)

**Name features (11)**:

| Feature | Description |
|---|---|
| `name_ratio` | Levenshtein-based similarity |
| `name_partial_ratio` | Best partial substring alignment |
| `name_token_sort_ratio` | Sort tokens before comparing (word order invariant) |
| `name_token_set_ratio` | Set intersection ratio (handles subset names) |
| `clean_name_ratio` | Ratio on legal-suffix-stripped names |
| `clean_token_sort_ratio` | Token sort on clean names |
| `exact_clean_match` | Boolean: clean_name == aux.clean_name |
| `exact_compact_match` | Boolean: compact_name == aux.compact_name |
| `phonetic_skel_match` | Boolean: skel == aux.skel |
| `first_word_match` | Boolean: first_word == aux.first_word |
| `name_len_diff` | abs(len(name) - len(aux.name)) |

**Address features (5)**:

| Feature | Description |
|---|---|
| `addr_token_jaccard` | Word-level Jaccard on address tokens |
| `addr_num_jaccard` | Jaccard on numeric tokens (house#, PIN) |
| `addr_num_exact` | Boolean: at least 1 shared number |
| `addr_is_missing` | Boolean: aux record has no address |
| `addr_len_diff` | Address string length difference |

**Metadata features (2)**:

| Feature | Description |
|---|---|
| `aux_is_s2` | Boolean: auxiliary record is from Source 2 |
| `candidate_rank` | Position in candidate list (0 = highest similarity) |

---

## 11. Model & Threshold Configuration

### Model (`model.joblib`)

- **Type**: LightGBM Binary Classifier
- **Hyperparameters**: `n_estimators=350`, `learning_rate=0.06`, `max_depth=6`, `num_leaves=31`
- **Val F0.5**: **0.8592** on 20,000-entity holdout
- **Val Precision**: 90.707% | **Val Recall**: 77.855% | **Singleton Accuracy**: 90.511%

### Configuration (`model_config.json`)

```json
{
  "match_threshold": 0.50,
  "singleton_cutoff": 0.40
}
```

| Parameter | Value | Meaning |
|---|---|---|
| `match_threshold` | 0.50 | Min model probability to assert a match |
| `singleton_cutoff` | 0.40 | If best candidate < this, emit empty (singleton protection) |

### Threshold Optimization Guidance

Always optimize on **Macro F0.5**, not F1 or accuracy.

- `match_threshold`: Search `[0.45, 0.85]` — optimal typically >0.5 for F0.5
- `singleton_cutoff`: Search `[0.30, 0.50]`
- Edit `model_config.json` only — never hardcode thresholds in `inference.py`

---

## 12. Validation Strategy

### 12.1 Holdout Split (`validation/`)

- 20,000 S1 entities, entity-disjoint from training
- Stratified by country (US/India) and cardinality (singleton/1 match/2+ matches)
- Total true matches in holdout: 69,219

### 12.2 Mandatory Metrics per Experiment

1. **Official Macro F₀.₅** — primary
2. **Macro Precision** and **Macro Recall**
3. **Candidate Recall** = Σ|T_i ∩ C_i| / Σ|T_i|
4. **Reduction Ratio** vs Cartesian space
5. **Singleton Accuracy**
6. **False Positive / False Negative counts**

### 12.3 France Zero-Shot Proxy

Hold out one country entirely during feature tuning to simulate OOD performance.

### 12.4 Leakage Prevention

1. No ground truth in candidate generation
2. Fit vocabularies/frequencies only on training folds
3. Validation pool must include ~26% distractors
4. Enforce target ID cannot be assigned to multiple S1 entities

---

## 13. Submission Rules & Checklist

### 13.1 Prohibited Resources (Instant Disqualification)

- Google Search / Places / Maps
- LinkedIn / Social Media scraping
- Company website crawling or domain resolving
- External business databases (OpenCorporates, D&B, ZoomInfo, Crunchbase)
- External entity resolution APIs or commercial services
- Government registrations (MCA India, SEC EDGAR, French SIRENE/INSEE)
- Geocoding APIs (Nominatim, Google Geocoding, Mapbox, HERE)
- Any internet-based business verification

### 13.2 Allowed Resources

- Python stdlib, NumPy, Pandas, Scipy, Scikit-learn
- XGBoost, LightGBM, CatBoost, PyTorch, HuggingFace, FastText, FAISS
- Pre-trained models: MIT or Apache 2.0 license, <= 8B parameters

### 13.3 Validation Command

```bash
# From student_resource/
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

Must exit code 0 before uploading.

### 13.4 Pre-Submission Checklist

- [ ] `validate_submission.py` reports PASS
- [ ] Exactly 1,732,544 rows in `matching_results.tsv`
- [ ] Exactly 1,732,544 rows in `candidate_pairs.tsv`
- [ ] Header: `source1_entity_id\tmatched_entity_ids` (exact casing)
- [ ] UTF-8 encoding (not Latin-1 or binary)
- [ ] True tab separator (not comma)
- [ ] Zero duplicate `source1_entity_id` rows
- [ ] Zero missing or extraneous `source1_entity_id` rows
- [ ] Zero self-matches (no `S1-` IDs in matched list)
- [ ] All matched IDs have `S2-` or `S3-` prefix
- [ ] Zero duplicate IDs within any single comma-separated list
- [ ] Singletons formatted with empty string after tab
- [ ] All matched IDs in matching_results subset of candidate_pairs
- [ ] Country invariant holds (all matches within same country)
- [ ] `requirements.txt` is up to date
- [ ] README.md has reproduction instructions

---

## 14. Error Analysis Reference

### False Positives (2x penalty — highest priority)

| Category | Root Cause | Mitigation |
|---|---|---|
| Common name collision | Generic names in different cities | Enforce address numeric overlap |
| Address blindness | Name match without verifying address | Add city token hard negative gate |
| Singleton contamination | Weak match on true singleton | Raise `singleton_cutoff` |
| Cross-entity assignment | Same S2/S3 ID for two S1 entities | `country_assigned_aux` global dedup |

### False Negatives (missed matches)

| Category | Root Cause | Mitigation |
|---|---|---|
| Blocking drops | True pair not in candidate pool | Add/tune blocking channels |
| Extreme transliteration | Native script vs English | Improve phonetic skeleton |
| Missing address penalty | True match rejected due to null addr | Use `addr_is_missing` feature |

---

## 15. Experiment Tracking Schema

```
experiment_id:          EXP-XXX
model_type:             LightGBM Pairwise Classifier
blocking_strategy:      [channels used + K]
candidate_recall:       0.XXXX
reduction_ratio:        >99.99%
validation_macro_f05:   0.XXXX
validation_precision:   0.XXXX
validation_recall:      0.XXXX
singleton_accuracy:     0.XXXX
num_features:           18
match_threshold:        0.XX
singleton_cutoff:       0.XX
inference_time_sec:     XXXs
submission_pass:        YES/NO
notes:                  [what changed, what worked, what failed]
```

### Current Best Result

```
experiment_id:          EXP-BEST-V4
model_type:             LightGBM Binary Classifier (1000-trees)
blocking_strategy:      6-channel union + Address composites
candidate_recall:       >95.9% (Inferred)
avg_candidates_per_s1:  31.5
reduction_ratio:        >99.99%  (1.73e9 -> 5.6e5 pairs)
validation_macro_f05:   0.97736
validation_precision:   98.671%
validation_recall:      95.840%
singleton_accuracy:     97.567%
num_features:           39
match_threshold:        0.94
singleton_cutoff:       0.94
```

---

## 16. Agent Behavioral Guidelines

> These rules govern how AI coding agents (Antigravity/Gemini/Claude/etc.) should behave when working on this codebase.

### 16.1 MUST Always Do

- **Read `AGENTS.md` first** before modifying any code
- **Use tab (`\t`) as delimiter** for all TSV reads and writes — NEVER comma
- **Run `validate_submission.py`** after any change touching output generation
- **Preserve existing comments and docstrings** unless explicitly asked to remove
- **Maintain country-partitioned processing** — never mix records across countries
- **Maintain target uniqueness** — no S2/S3 ID assigned to more than one S1 entity
- **Include every S1 test entity** in output files, including singletons
- **Use `pd.read_csv(file, sep="\t")`** for all TSV loading
- **Use full module import path** `code.business_entity_resolution.src.*`
- **Run scripts from `student_resource/`** as working directory
- **Optimize for Candidate Reduction** — Candidate set size is an official ranking tiebreaker. Maintain or reduce the `avg_candidates_per_s1` (current: 31.5).

### 16.2 MUST Never Do

- **Never predict `S1-` IDs** in matched or candidate columns
- **Never omit an S1 entity** from output files
- **Never use external APIs or internet lookups** to resolve businesses
- **Never use comma as TSV delimiter** — addresses contain commas internally
- **Never train on `validation/` directory** — it is strictly test-only
- **Never hard-code country logic** to only {US, India} — France must work zero-shot
- **Never optimize for Micro-F0.5 or F1** — always use Macro F0.5
- **Never add the same ID twice** in a comma-separated list
- **Never assume address is always present** — ~3.4% of aux records are null
- **Never delete `utils/validate_submission.py`** — it is the official validator
- **Never delete `model.joblib`** without saving a backup

### 16.3 Official Rulings & Interpretations
- **Cloud Computing Platform**: "Using AWS/SageMaker is encouraged but not mandatory. You may develop on other platforms of your choice (e.g., Kaggle, Colab)." The $200 AWS credit is optional.
- **Candidate Tiebreaker**: `candidate_pairs.tsv` is NOT scored on the live leaderboard, but is evaluated during final package verification. Teams generating a smaller candidate set per S1 entity will be ranked higher.
- **Leaderboard Upload**: Only `matching_results.tsv` is uploaded to the live portal. `candidate_pairs.tsv` and the codebase are submitted in the final `_submission.zip` package at the end of the competition.

### 16.4 Code Modification Principles

- **Feature changes**: Add to `features.py`, update `num_features` count, retrain model
- **Blocking changes**: Edit `blocking.py`, run `test_blocking.py` to verify recall didn't drop
- **Threshold changes**: Edit `model_config.json` only — never hardcode in `inference.py`
- **Preprocessing changes**: Add to `preprocess.py`, test on multilingual samples (Devanagari, French: e e e a c i o with accents)
- **Memory**: Inference processes one country at a time — avoid loading all countries simultaneously

### 16.4 Testing Workflow

```bash
# 1. Test blocking recall on validation holdout
python3 code/business_entity_resolution/src/test_blocking.py

# 2. Evaluate Macro F0.5
python3 code/business_entity_resolution/src/evaluation.py

# 3. Validate output format
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids

# 4. Full test inference
python3 code/business_entity_resolution/src/inference.py \
    --test-dir dataset/test \
    --output-dir output
```

### 16.5 Data Challenges Quick Reference

**Name variations:**

| Type | Example |
|---|---|
| Legal suffix | `Corp` vs `Corporation` vs `Pvt Ltd` vs `SARL` |
| Transliteration | Devanagari alongside Latin script |
| DBA/Trade names | `wilfordhancock.com` as a business name |
| Word order | `Vision Partners Corp` vs `Partners Vision Inc` |
| Typos | Phonetic and character-level edit errors |

**Address variations:**

| Type | Example |
|---|---|
| Street abbreviation | `Rd` vs `Road`, `St` vs `Street` |
| French typology | `Rue`, `R.`, `Boulevard`, `Bd`, `Allee` |
| Component reorder | `IA, Iowa City, 1064 Newton Rd, Unit 11` |
| India landmarks | `Near SBI ATM`, `Behind Bus Stand`, `KH NO. -570/13` |
| City aliases | `Gurugram` vs `Gurgaon`, `Bengaluru` vs `Bangalore` |

**French test set (zero-shot):**
- Corporate: `SARL`, `SAS`, `SA`, `EURL`, `SNC`, `SCI`, `EI`
- Streets: `Rue`, `Avenue`, `Boulevard`, `Allee`, `Place`, `Chemin`, `Impasse`
- Modifiers: `bis`, `ter`
- Accented chars: `e`, `e`, `e`, `a`, `c`, `i`, `o` variants — must not break any string function

### 16.6 Noise Handling Reference

| Noise Type | Handling |
|---|---|
| Missing address | Use `addr_is_missing` boolean feature; rely on name similarity |
| Script mixing | `phonetic_skeleton()` in `preprocess.py` collapses scripts |
| Legal suffixes | `clean_business_name()` strips before all comparisons |
| Word order | `name_token_sort_ratio` feature handles this |
| Numeric codes | Extracted as `addr_nums`; matched via `addr_num_jaccard` |
| OCR/typo errors | `name_ratio` (Levenshtein) and trigram blocking catch these |

### 16.7 Competition Principles (Priority Order)

1. **Precision First** — F0.5 weights precision 2x over recall. When uncertain, do not match.
2. **Never Drop Candidates Early** — Blocking optimizes recall (>87%). Classifier filters FP; cannot recover dropped TP.
3. **Target Mutual Exclusivity** — Enforce globally: one S2/S3 ID to at most one S1 entity.
4. **Respect Singletons** — 5.58% of S1 are singletons. Correct empty predictions = free +0.0558 to macro F0.5.
5. **Country Zero-Shot Robustness** — All normalization must work across English, Indian transliterations, and French.
6. **Mechanical Verification** — Always run `validate_submission.py` before uploading.

---

## Appendix A — Run Commands Reference

```bash
# All commands from: student_resource/

# Full AWS inference
./run_aws.sh

# Local test inference
python3 code/business_entity_resolution/src/inference.py \
    --test-dir dataset/test \
    --output-dir output

# Validate submission
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids

# Test blocking recall
python3 code/business_entity_resolution/src/test_blocking.py

# Retrain model
python3 code/business_entity_resolution/src/train_model.py

# Generate validation split
python3 code/business_entity_resolution/src/make_val_split.py
```

## Appendix B — Known Ambiguities

| Item | Status | Handling |
|---|---|---|
| Public/private split ratio | UNKNOWN | Do not overfit to public score; trust local CV |
| Submission quota per day | UNKNOWN | Validate locally before every upload |
| Exact team zip filename | `Deciders_submission.zip` | Confirmed from portal |
| Hardware audit environment | UNKNOWN | Keep pipeline within <=32GB RAM, <=4h runtime |
