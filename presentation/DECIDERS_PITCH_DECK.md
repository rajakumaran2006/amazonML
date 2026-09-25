# Team Deciders: Official Hackathon Presentation & Pitch Deck Guide
**Competition:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Team Name:** Deciders  
**Members:** Rajakumaran P + Team  
**Evaluation Metric:** Official Macro-$F_{0.5}$ with Singleton Penalties  
**Leaderboard Target:** >0.9804 (Surpassing Current #1 "Grinders")

---

## Presentation Executive Overview

This presentation blueprint translates the official Amazon problem statement and our production-grade ML engineering pipeline into a winning 10-slide deck. Each section links directly to the official visual reference assets captured from Amazon's problem explainer video.

| Slide # | Slide Title | Reference Visual Asset | Core Innovation / Key Message |
|---|---|---|---|
| **Slide 1** | **Title & Executive Summary** | `assets/01_amazon_challenge_title.png` | Team Deciders: Scaling Entity Resolution across 24M records |
| **Slide 2** | **The Industry Problem: Fragmented Records** | `assets/02_problem_signup_flow.png` | Vendor silos & noisy registration data with zero shared IDs |
| **Slide 3** | **Multi-Source Discrepancies & Variations** | `assets/03_multisource_representation.png` | 1 real business, 3 distinct vendor representations |
| **Slide 4** | **Resolution Targets & Hard Invariants** | `assets/04_entity_resolution_lookalikes.png` | 1-to-many matches, look-alike rejection & country isolation |
| **Slide 5** | **Ultra-Fast 4-Channel Inverted Index Blocking** | `assets/05_blocking_candidate_pairs.png` | 99.999% Cartesian reduction ($17.2 \times 10^{12} \to 31$ cands/entity) |
| **Slide 6** | **Multilingual & Cross-Script Feature Engineering** | `assets/06_model_inference_pipeline.png` | RapidFuzz C++ + Phonetic consonant skeletons + French suffixes |
| **Slide 7** | **Classifier & Metric-Aligned Optimization** | `assets/06_model_inference_pipeline.png` | Calibrated LightGBM tuned directly for Macro-$F_{0.5}$ |
| **Slide 8** | **Dataset & Offline Leak-Free Validation** | `assets/07_dataset_train_test_split.png` | 20k validation benchmark reproducing leaderboard rules |
| **Slide 9** | **Scalability & AWS Cloud Deployment** | N/A | Multi-core batch streaming on AWS SageMaker / EC2 |
| **Slide 10** | **Conclusion, Business Impact & Next Steps** | N/A | Production readiness, zero external APIs, enterprise ROI |

---

## Detailed Slide-by-Slide Deck Outline & Speaker Script

### Slide 1: Title & Executive Summary
- **Visual Reference:** `presentation/assets/01_amazon_challenge_title.png`
- **Headline:** Business Entity Resolution at Enterprise Scale
- **Sub-headline:** High-Precision, Cross-Lingual Multi-Source Matching across 24 Million Records
- **Key Talking Points:**
  - Introduce Team Deciders.
  - Frame the challenge: Unifying business identities across three disparate commercial feeds (Source 1 reference, Source 2 vendor, Source 3 vendor) with zero shared keys.
  - Highlight the core metric: **Macro-$F_{0.5}$**, demanding high precision ($\beta=0.5$ penalizes false positives twice as heavily as false negatives) while properly handling 5.58% singletons.

---

### Slide 2: The Problem — From Sign-Up to Matched Records
- **Visual Reference:** `presentation/assets/02_problem_signup_flow.png`
- **Headline:** The Real-World Ingestion Bottleneck
- **Key Talking Points:**
  - When businesses onboard onto Amazon Business, they supply basic registration attributes: `business_name` and `business_address`.
  - In practice, phone numbers and tax IDs are missing, leaving only raw, unstandardized text strings.
  - Explain why naive string matching fails: typos, word order transpositions, and partial addresses.

---

### Slide 3: Multi-Source Representations
- **Visual Reference:** `presentation/assets/03_multisource_representation.png`
- **Headline:** One Business, Three Contradictory Views
- **Visual Demonstration:**
  - **Source 1 (Reference):** `Acme Robotics Inc.`, `500 Market St, San Jose`
  - **Source 2 (Vendor A):** `Acme Robotics Incorporated`, `500 Market Street, San Jose CA`
  - **Source 3 (Vendor B):** `Acme Robotics`, `Nr. City Hall, San Jose`
- **Key Talking Points:**
  - Vendor A includes state codes and spells out legal forms (`Incorporated`).
  - Vendor B uses colloquial landmarks (`Nr. City Hall`) rather than street addresses.
  - Our pipeline must bridge these format discrepancies without human intervention.

---

### Slide 4: Entity Resolution Targets & Look-Alikes
- **Visual Reference:** `presentation/assets/04_entity_resolution_lookalikes.png`
- **Headline:** Disambiguating True Matches from Dangerous Look-Alikes
- **Key Talking Points:**
  - A Source 1 entity may map to zero, one, or multiple records across Source 2 and Source 3.
  - Look-alike hazards: A company like `Acme Bakery LLC` at a neighboring address shares the brand token `Acme` but is completely distinct.
  - False positives destroy Macro-$F_{0.5}$; our model enforces strict address verification to eliminate look-alikes.

---

### Slide 5: Candidate Generation (Blocking Pipeline)
- **Visual Reference:** `presentation/assets/05_blocking_candidate_pairs.png`
- **Headline:** Sub-Linear Search Space Reduction: From $17.2 \times 10^{12}$ to 31 Pairs
- **Key Talking Points:**
  - Full Cartesian product between $S_1$ and $S_2 \cup S_3$ is computationally intractable ($17.2$ Trillion pairs).
  - Our 4-Channel Inverted Hash Index reduces candidates to an average of **31.5 candidates per entity**:
    1. *Channel 1 (Canonical Name)*: Normalized corporate suffix removal across US, India, and France.
    2. *Channel 1B (Domain / Compact Name)*: Whitespace-stripped brand roots.
    3. *Channel 2 (Distinctive Token Index)*: Inverted index on non-stopwords with length $\ge 4$.
    4. *Channel 3 (Core Brand Anchor)*: High-frequency brand anchor matching.
    5. *Channel 4 (Phonetic Consonant Skeletons)*: Zero-shot cross-script alignment.
  - Achieves **>87.5% candidate recall** while pruning **99.999%** of irrelevant comparisons.

---

### Slide 6: Multilingual & Cross-Script Engineering
- **Visual Reference:** `presentation/assets/06_model_inference_pipeline.png`
- **Headline:** Cross-Script Phonetics & Zero-Shot French Handling
- **Key Talking Points:**
  - **Indic Cross-Script Transliteration**: Devanagari, Telugu, Malayalam, and Tamil scripts are transliterated and mapped to phonetic consonant skeletons (`Premier` $\to$ `prmr`, `ప్రీమియర్` $\to$ `prmr`).
  - **Zero-Shot French Adaptation**: Handled French corporate acronyms (`SARL`, `SAS`, `SA`, `EURL`) and street types (`Rue`, `Boulevard`, `Avenue`).
  - **Vectorized RapidFuzz**: 18 pairwise features computed in C++ at >100,000 pairs/sec.

---

### Slide 7: Matching Model & Invariant Enforcement
- **Visual Reference:** `presentation/assets/06_model_inference_pipeline.png`
- **Headline:** Calibrated LightGBM & Post-Processing Guardrails
- **Key Talking Points:**
  - Gradient boosted decision trees (LightGBM) trained on 200,000 candidate pairs with balanced positive/negative sampling.
  - Threshold $\theta^*$ chosen directly to maximize Macro-$F_{0.5}$.
  - **Singleton Protection Gate**: Dedicated classifier for singletons ($5.58\%$ of entities) to capture 1.0 perfect scores.
  - **Target Uniqueness**: Enforces that auxiliary records link to at most one reference entity.

---

### Slide 8: Dataset & Leak-Free Validation Benchmark
- **Visual Reference:** `presentation/assets/07_dataset_train_test_split.png`
- **Headline:** Rigorous Validation Replicating Official Leaderboard Rules
- **Key Talking Points:**
  - Built an offline leak-free holdout benchmark of 20,000 $S_1$ entities and 86,508 auxiliary records.
  - Faithful Python implementation of the official Unstop evaluation script.
  - Validation metrics: **Macro-$F_{0.5} = 0.8592$**, **Macro Precision = $90.71\%$**, **Macro Recall = $77.86\%$**.

---

### Slide 9: Scalability & Cloud Architecture
- **Headline:** High-Throughput AWS Distributed Inference
- **Key Talking Points:**
  - Automated deployment via `run_aws.sh` on AWS SageMaker / EC2 instances.
  - Chunked dataset transfer (under 100MB per GitHub chunk) enables fast replication and continuous integration.
  - Multi-threaded batch streaming processes 1.43M records in minutes with constant memory usage.

---

### Slide 10: Conclusion & Business Impact
- **Headline:** Enterprise-Ready Entity Resolution for Amazon Business
- **Key Talking Points:**
  - **Strictly Compliant**: Zero external API dependencies, <8B parameters, 100% open-source (MIT/Apache 2.0).
  - **High Business Value**: Solves vendor catalog deduplication, automated vendor onboarding, and fraud detection.
  - **Team Deciders**: Setting the benchmark for high-precision entity resolution.

---

## Presentation Design & Delivery Tips for the Team

1. **Keep Visuals Front and Center**: Use the reference screenshots in `presentation/assets/` to directly anchor your explanations to the official problem diagrams.
2. **Emphasize Precision**: When asked why our precision is so high ($90.71\%$), emphasize that the hackathon evaluates on $F_{0.5}$ which penalizes false positives twice as much as false negatives.
3. **Highlight the Singleton Strategy**: Emphasize how our singleton detector protects the 5.58% unlinked entities to secure perfect 1.0 scores on those entities.
4. **Demonstrate Cross-Script Matching**: Show a live example of Indic transliteration (e.g., Telugu `ప్రీమియర్` matching English `Premier`).
