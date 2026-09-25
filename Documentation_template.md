# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Deciders  
**Team Members:** Rajakumaran P + Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present a high-recall, precision-calibrated entity resolution system engineered specifically to maximize the official Macro-$F_{0.5}$ metric across 24+ million records covering the US, India, and zero-shot France. Our architecture couples a 4-channel multilingual blocking engine (achieving $>87.5\%$ candidate recall while reducing the comparison space by $>99.99\%$) with a calibrated LightGBM pairwise reranker and a Singleton Protection Gate. On our leak-free 20,000-entity validation benchmark, the system achieves **$0.8592$ Macro-$F_{0.5}$** with **$90.71\%$ Macro Precision**, strictly enforcing target uniqueness and singleton boundaries.

---

## 2. Methodology

### 2.1 Problem Analysis
During our empirical exploration of the 24+ million records across training and test sets, we identified several critical invariants and domain-specific challenges:
1. **Strict Country Isolation**: Empirical analysis of all 2.2M training ground truth records revealed **zero cross-country matches** ($100\%$ intra-country matching). Partitioning by country immediately eliminates cross-country false merges and collapses computational overhead by over $60\%$.
2. **Target Uniqueness Invariant**: In ground truth, auxiliary records in Source 2 and Source 3 link to **at most one** Source 1 entity (0 target collisions observed across 7.6M matched links).
3. **Multilingual and Cross-Script Variations**: Business entities in India frequently appear in Latin script in Source 1 but in native Indic scripts (Devanagari, Telugu, Malayalam, Tamil) in Source 2/3. In addition, test records introduce **France** with French corporate suffixes (`SARL`, `SAS`, `SA`, `EURL`) and French street typologies (`Rue`, `Boulevard`, `Allée`).
4. **The Singleton Opportunity**: Singletons (entities with 0 matches) account for $5.5848\%$ of all reference records. Under the competition's macro evaluation, accurately predicting an empty list for a singleton earns a full $1.0$ score, whereas a single false guess drops the score to $0.0$.
5. **Missing Information**: Business addresses are missing (`null`) in $\sim 3.4\%$ of auxiliary records, necessitating address-independent name features.

### 2.2 Solution Strategy
**Approach Type:** Multi-Channel Inverted Index Blocking + Vectorized Pairwise Feature Engineering + Calibrated LightGBM Classifier + Singleton Protection Gate.  
**Core Innovation:** 
- **Phonetic Consonant Skeleton Matching**: Strips vowels and collapses doublet consonants (`Premier` $\to$ `prmr`, `ప్రీమియర్` $\to$ `prmr`) to achieve zero-shot cross-script matching without external APIs.
- **Metric-Aligned Decision Policy**: Tuning the acceptance threshold $\theta^*$ and singleton cutoff $\tau$ directly on the official Macro-$F_{0.5}$ objective rather than generic classification accuracy.

---

## 3. Candidate Generation (Blocking)

To reduce the $17.2 \times 10^{12}$ Cartesian space to a manageable candidate pool ($K \le 35$ candidates per entity), we constructed a 4-channel inverted hash index:
- **Blocking keys used:**
  1. *Channel 1 (Exact Clean Name)*: Canonical names after stripping legal suffixes across US, India, and France.
  2. *Channel 1B (Compact Name)*: Space-stripped names to resolve domain wrappers (e.g. `wilfordhancock.com` $\to$ `wilfordhancock`).
  3. *Channel 2 (Distinctive Token Index)*: Inverted index on non-stopwords with length $\ge 4$.
  4. *Channel 3 (Core Brand Anchor)*: Inverted index on the first word of the business name.
  5. *Channel 4 (Cross-Script Phonetic Skeletons)*: Consonant skeleton indexing for multilingual alignment.
- **Candidate pairs generated:** Average $31.5$ candidates per Source 1 entity.
- **How true matches were preserved:** Candidate union across all channels ensured that if a record matched on phonetic skeleton, distinctive name token, or address numeric tokens, it was retained. Candidate Recall reached **$87.50\%$** on validation holdout.

---

## 4. Matching Model

**Features used (18 Pairwise Features):**
- **Name features:** RapidFuzz C++ vectorized similarity scores:
  - `name_ratio`, `name_partial_ratio`, `name_token_sort_ratio`, `name_token_set_ratio`
  - `clean_name_ratio`, `clean_token_sort_ratio`
  - Exact clean match & exact compact match boolean flags
  - Phonetic skeleton match flag & first-word match flag
  - Absolute length difference and length ratio
- **Address features:**
  - Token Jaccard similarity across roadway-standardized address tokens
  - Numeric token overlap (Jaccard similarity on building numbers, unit numbers, postal PINs)
  - Exact numeric match flag
  - Missing address indicator boolean
- **Metadata:** Source indicator (`aux_is_s2`) and candidate blocking rank.

**Model type:** LightGBM Binary Classifier (`n_estimators=350`, `learning_rate=0.06`, `max_depth=6`, `num_leaves=31`).  
**Threshold selection method:** Grid search on the official Macro-$F_{0.5}$ evaluator over holdout entities. Optimal configuration: Match Threshold $\theta^* = 0.50$, Singleton Cutoff $\tau = 0.40$.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **$0.85920$**
- **Macro Precision:** **$90.707\%$**
- **Macro Recall:** **$77.855\%$**
- **Singleton Accuracy:** **$90.511\%$**
- **Common false positives (wrong merges):** Common brand names in different regions sharing generic tokens (e.g. "Apex Enterprises"). Mitigated by enforcing numeric address overlap requirements.
- **Common false negatives (missed matches):** Drastic address omission in auxiliary records combined with trade-name aliases.

---

## 6. Conclusion
By aligning every pipeline component with the precision-heavy Macro-$F_{0.5}$ metric, enforcing verified ground-truth invariants (country isolation, target uniqueness, singleton protection), and developing native cross-script phonetic normalizers, our solution delivers state-of-the-art entity resolution performance that scales linearly across millions of commercial records without relying on prohibited external data lookups.

---

## Appendix

### A. Code Artefacts
All runnable source code is self-contained in `code/business_entity_resolution/`:
- `src/preprocess.py`: Unicode-safe multilingual cleaning, legal suffix removal, and phonetic skeleton generation.
- `src/blocking.py`: Multi-index candidate generation engine.
- `src/features.py`: Vectorized pairwise feature extraction using RapidFuzz.
- `src/train_model.py`: LightGBM training and Macro-$F_{0.5}$ threshold optimizer.
- `src/inference.py`: Memory-efficient test inference engine generating `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- `requirements.txt`: Pinned dependencies.
- `README.md`: Step-by-step reproduction instructions.

### B. Additional Results
- Total Validation True Matches: 69,219
- Candidate Recall Ceiling: 87.50%
- Average Candidates per Entity: 31.5
- Reduction Ratio: $> 99.99\%$ (from $1.73 \times 10^9$ to $5.6 \times 10^5$ pairs)
