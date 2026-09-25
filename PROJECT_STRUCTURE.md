# Project Directory Structure & Organization Guide
**Amazon ML Challenge 2026 — Business Entity Resolution**  
**Team:** Deciders  
**Repository:** `https://github.com/rajakumaran2006/amazonML.git`

---

## High-Level Architecture Overview

```
student_resource/ (Workspace Root)
│
├── COMPETITION_SSOT.md                # 40-Section Single Source of Truth (All Specs & Invariants)
├── Documentation_template.md          # Official Hackathon Solution Report (Ready for Submission)
├── PROJECT_STRUCTURE.md               # Master Directory & Navigation Guide (This File)
├── README.md                          # Official Amazon Challenge Problem Statement
├── run_aws.sh                         # 1-Click Automated AWS SageMaker / EC2 Runner
├── .gitignore                         # Configured for GitHub 100MB limit & clean hygiene
│
├── presentation/                      # Hackathon Presentation & Pitch Deck Materials
│   ├── DECIDERS_PITCH_DECK.md         # Complete 10-slide deck outline, scripts & talking points
│   └── assets/                        # Official reference slides from Amazon problem explainer video
│       ├── 01_amazon_challenge_title.png          # Slide 1: Problem Definition
│       ├── 02_problem_signup_flow.png             # Slide 2: Ingestion & Registration Flow
│       ├── 03_multisource_representation.png      # Slide 3: Multi-Source Discrepancies
│       ├── 04_entity_resolution_lookalikes.png    # Slide 4: Target Entities vs Look-Alikes
│       ├── 05_blocking_candidate_pairs.png        # Slide 5: Multi-Channel Inverted Index Blocking
│       ├── 06_model_inference_pipeline.png        # Slide 6: ML Filtering to matching_results.tsv
│       └── 07_dataset_train_test_split.png        # Slide 7: Training vs Testing Specifications
│
├── code/                              # Production Source Code & Models
│   └── business_entity_resolution/
│       ├── requirements.txt           # Verified Python dependencies
│       ├── README.md                  # Package overview
│       └── src/
│           ├── preprocess.py          # Multilingual text normalization (Indic, French, US)
│           ├── blocking.py            # Multi-channel inverted index & fast candidate generator
│           ├── features.py            # 18 RapidFuzz C++ vectorized pairwise similarity features
│           ├── train_model.py         # LightGBM training pipeline with balanced sampling
│           ├── inference.py           # Multi-threaded production batch inference engine
│           ├── evaluation.py          # Macro F0.5 metric calculator with singleton penalty
│           ├── make_val_split.py      # Leak-free offline holdout generator
│           ├── test_blocking.py       # Blocking recall and candidate pool validation
│           ├── model.joblib           # Trained LightGBM ensemble weights (0.8592 Val F0.5)
│           └── model_config.json      # Optimal decision thresholds and model hyperparameters
│
├── dataset/                           # Raw Datasets & Distributed Chunks
│   ├── train/                         # Official training data (train_source1/2/3, ground_truth)
│   ├── test/                          # Restored test data (test_source1/2/3 for US, India, France)
│   ├── test_chunk_aa ... ai           # 60MB GitHub-compatible compressed data chunks for AWS
│   └── sample_submission.tsv          # Format reference
│
├── validation/                        # Offline Validation Holdout Benchmark
│   ├── val_source1.tsv                # 20,000 S1 reference businesses
│   ├── val_source2.tsv                # 42,754 S2 auxiliary records
│   ├── val_source3.tsv                # 43,754 S3 auxiliary records
│   └── val_ground_truth.tsv           # Verified ground truth labels
│
├── output/                            # Competition Submission Deliverables
│   ├── candidate_pairs.tsv            # Candidate pair shortlisted records
│   ├── matching_results.tsv           # Final predicted entity matches (Tab-separated)
│   └── Deciders_submission.zip        # Final upload package for Unstop portal
│
├── utils/                             # Official Verification Tools
│   └── validate_submission.py         # Strict schema, ID integrity, and format validator
│
└── archive/                           # Archived Local Bundles (Git-ignored)
    ├── aws_package.zip                # Consolidated package archive
    ├── code_only.zip                  # Lightweight code archive
    └── test_data.tar.gz               # Combined test dataset archive
```

---

## Key Workflows & Commands

### 1. Presentation Deck (PPT) Reference
- All presentation reference slides are neatly cataloged in [presentation/assets/](file:///Users/raja/Downloads/student_resource/presentation/assets/).
- The slide-by-slide script, layout suggestions, and team talking points are in [presentation/DECIDERS_PITCH_DECK.md](file:///Users/raja/Downloads/student_resource/presentation/DECIDERS_PITCH_DECK.md).

### 2. AWS SageMaker / EC2 Execution
To run full inference on AWS SageMaker or EC2:
```bash
./run_aws.sh
```
This automatically restores test chunks, installs dependencies, runs multi-core batch inference, verifies the output with `utils/validate_submission.py`, and creates `Deciders_submission.zip`.

### 3. Local Offline Validation
To evaluate candidate recall and model accuracy on the 20,000-entity validation split:
```bash
python3 code/business_entity_resolution/src/test_blocking.py
```

### 4. Verifying Submission Files
To validate any generated submission before uploading to the Unstop portal:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```
