# BUSINESS ENTITY RESOLUTION HACKATHON

# SINGLE SOURCE OF TRUTH (SSOT)

> **Document Status**: Definitive / Competition-Grade Specification
> **Target Competition**: ML Challenge 2026 — Business Entity Resolution Challenge
> **Authority**: Formulated strictly from official hackathon guidelines, dataset schemas, validation scripts, and ground truth analysis.
> **Enforcement**: Mandatory governance document for all downstream data processing, candidate generation, modeling, validation, threshold selection, and submission packaging.

---

## 1. Competition Identity

- **Competition Title**: ML Challenge 2026 — Business Entity Resolution Challenge
- **Domain**: Scaled Commercial Entity Resolution (ER) / Record Linkage / Deduplication
- **Evaluation Duration**: 72-Hour Hackathon
- **Primary Deliverables**:
  1. Live Leaderboard Submission: `matching_results.tsv`
  2. Final Submission Archive: `<team_name>_submission.zip` containing code, outputs, and documentation template.

---

## 2. Official Objective

Given business records arriving from three (3) independent, noisy commercial data sources with no shared common identifiers, construct a machine learning system that determines which records across sources refer to the exact same real-world business entity.

- **Anchor / Reference Source**: **Source 1** is the deduplicated reference source. Every real-world entity in Source 1 appears at most once.
- **Linkage Direction**: Link **Source 2** and **Source 3** records to each **Source 1** entity.
- **Cardinality per Source 1 Entity**: A Source 1 entity may match zero (0), one (1), or multiple records from Source 2 and Source 3.
- **Primary Scoring Objective**: Maximize the official **Macro-Averaged F_0.5 Score** across all Source 1 entities in the private test evaluation set.

---

## 3. Problem Definition

Entity Resolution across heterogeneous business listings:

- Let $\mathcal{S}_1$ be the set of deduplicated reference records.
- Let $\mathcal{S}_2$ and $\mathcal{S}_3$ be auxiliary, un-deduplicated, noisy collections of business records.
- For each record $e_1 \in \mathcal{S}_1$, identify the exact subset of records $\mathcal{M}(e_1) \subseteq (\mathcal{S}_2 \cup \mathcal{S}_3)$ such that every $m \in \mathcal{M}(e_1)$ denotes the identical physical business establishment as $e_1$.
- If no record in $\mathcal{S}_2 \cup \mathcal{S}_3$ refers to entity $e_1$, then $\mathcal{M}(e_1) = \emptyset$ (singleton entity).
- **Global Uniqueness Semantics (Verified from Ground Truth)**: Each auxiliary record $s \in \mathcal{S}_2 \cup \mathcal{S}_3$ belongs to at most one reference entity $e_1 \in \mathcal{S}_1$. Overlapping clusters where the same $S_2$ or $S_3$ ID is assigned to multiple $S_1$ entities are strictly non-existent in ground truth.

---

## 4. Source Data Structure

All data files are Tab-Separated Values (`.tsv`), UTF-8 encoded.

### 4.1 Schema for Source Files (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)

| Column Name          | Data Type | Nullable | Description & Constraints                                                                                           |
| :------------------- | :-------- | :------- | :------------------------------------------------------------------------------------------------------------------ |
| `entity_id`        | String    | NO       | Unique record identifier. Prefix denotes source:`S1-` (Source 1), `S2-` (Source 2), `S3-` (Source 3).         |
| `business_name`    | String    | NO       | Name of the business entity. Contains variations, abbreviations, suffixes, typos, multilingual transliterations.    |
| `business_address` | String    | YES      | Address of the business. Contains missing components, variations, landmarks. Missing in ~3.4% of S2 and S3 records. |
| `country`          | String    | NO       | Open-set string label.`US` and `India` in train; `US`, `India`, and `France` in test.                     |

### 4.2 Delimiter Rule

- **Mandatory Delimiter**: `\t` (Tab, ASCII `0x09`).
- **Reason**: `business_address` and comma-separated ID lists frequently contain commas (`,`). Reading or writing with commas will corrupt column boundaries.

---

## 5. Training Data

The training set contains verified matching pairs across 3 sources covering `US` and `India`.

| File Path                                | Total Line Count | Total Records | Country Breakdown                                     |
| :--------------------------------------- | :--------------- | :------------ | :---------------------------------------------------- |
| `dataset/train/train_source1.tsv`      | 2,206,822        | 2,206,821     | `US`: 1,323,633 (60.0%)`India`: 883,188 (40.0%)   |
| `dataset/train/train_source2.tsv`      | 5,034,617        | 5,034,616     | `US`: 3,016,817 (59.9%)`India`: 2,017,799 (40.1%) |
| `dataset/train/train_source3.tsv`      | 5,285,604        | 5,285,603     | `US`: 3,170,056 (60.0%)`India`: 2,115,547 (40.0%) |
| `dataset/train/train_ground_truth.tsv` | 2,206,822        | 2,206,821     | Source 1 Ground Truth Links                           |

**Total Records in Train**: 12,527,040 records.

---

## 6. Test Data

The test set contains unlabelled records across 3 sources covering `US`, `India`, and a completely new country: `France`.

| File Path                         | Total Line Count | Total Records | Country Breakdown                                                                |
| :-------------------------------- | :--------------- | :------------ | :------------------------------------------------------------------------------- |
| `dataset/test/test_source1.tsv` | 1,732,545        | 1,732,544     | `India`: 809,986 (46.8%)`US`: 663,106 (38.3%)`France`: 259,452 (15.0%)     |
| `dataset/test/test_source2.tsv` | 4,887,274        | 4,887,273     | `India`: 2,312,565 (47.3%)`US`: 1,871,330 (38.3%)`France`: 703,378 (14.4%) |
| `dataset/test/test_source3.tsv` | 5,082,317        | 5,082,316     | `India`: 2,405,000 (47.3%)`US`: 1,945,701 (38.3%)`France`: 731,615 (14.4%) |

**Total Records in Test**: 11,702,133 records.
**Critical Constraint**: Every one of the 1,732,544 `test_source1.tsv` entities must appear exactly once in the submission files.

---

## 7. Ground Truth

File: `dataset/train/train_ground_truth.tsv`
Columns: `source1_entity_id\tmatched_entity_ids`

### Verified Empirical Properties of Ground Truth:

1. **Row Count**: Exactly 2,206,821 rows matching `train_source1.tsv` 1-to-1.
2. **Singletons (Zero Matches)**: Exactly 123,247 entities (5.5848% of Source 1).
3. **Matched Entities**: Exactly 2,083,574 entities (94.4152% of Source 1).
4. **Distribution of Matches per Matched S1 Entity**:
   - Min: 1
   - 25th percentile: 3
   - Median (50th percentile): 4
   - Mean: 3.666
   - 75th percentile: 5
   - Max: 11
   - Common match counts: 3 matches (530,841), 4 matches (484,115), 2 matches (375,212), 5 matches (321,957), 6 matches (164,868), 1 match (119,157).
5. **Cross-Country Matches**: **Strictly 0**. Verified across training ground truth. Entities only link to entities within the exact same `country`.
6. **Target Entity Uniqueness**: **Strictly 0** target entities ($S_2$ or $S_3$) match more than one $S_1$ entity. Total unique targets = 7,638,365 across 2,083,574 non-singleton S1 entities.
7. **Target Source Composition**: Matches contain a blend of Source 2 (`S2-`) and Source 3 (`S3-`) IDs.
8. **Distractor / Unmatched S2/S3 Records**:
   - Total S2 + S3 records in train = 10,320,219.
   - Matched S2 + S3 records in GT = 7,638,365 (~74.01%).
   - Approximately 25.99% (2,681,854) of S2 and S3 records in the pool are distractors that do not match any Source 1 record.

---

## 8. Entity Matching Semantics

1. **Reference-Anchor Semantics**: All matching is oriented from Source 1 outwards: $\mathcal{S}_1 \rightarrow \mathcal{P}(\mathcal{S}_2 \cup \mathcal{S}_3)$.
2. **No Self-Matches**: Source 1 entities can NEVER match another Source 1 entity. Predicting an `S1-` ID in `matched_entity_ids` causes immediate validation rejection.
3. **Partitioning**: Each real-world entity is a distinct cluster anchored by at most one $S_1$ entity. A single $S_1$ entity matches a set $\{S_{2,a}, S_{2,b}, \dots, S_{3,x}, S_{3,y}, \dots\}$.
4. **Target Mutual Exclusivity**: Because $S_1$ is deduplicated, an $S_2$ or $S_3$ ID assigned to entity $S_1^A$ cannot simultaneously belong to $S_1^B$.

---

## 9. Noise / Data Challenges

### 9.1 Business Name Variations

- **Legal Form Variations**: `Corp` vs `Corporation`, `Pvt` vs `Private`, `Ltd` vs `Limited`, `Inc` vs `Incorporated`, `LLC`, `LLP`, `Co`, `Company`.
- **Transliteration / Native Scripts**: Devanagari script (e.g. `राम मार्केटिंग प्राइवेट लिमिटेड`, `मॉडर्न फाइनेंस`) co-occurring with Latin/English names.
- **French Corporate Forms in Test**: `SARL`, `SAS`, `SA`, `EURL`, `SNC`, `SCI`, `EI`.
- **DBA & Trade Names**: Prefixes like `-- Holloway Peak Inc Seafood`, brand names vs legal entities, website domain names used as business names (e.g. `wilfordhancock.com`).
- **Formatting / Punctuation**: Ampersands (`&` vs `and`), slashes, hyphens, parentheses, quotes.
- **Word Order Transposition**: `Vision Partners Corp` vs `Partners Vision Inc`.
- **Typos & Misspellings**: Phonetic and character-level edit distance errors.

### 9.2 Address Variations

- **Street / Way Typology Abbreviations**: `Rd` vs `Road`, `St` vs `Street`, `Ave` vs `Avenue`, `Blvd` vs `Boulevard`, `Hwy` vs `Highway`.
- **French Address Typologies**: `Rue`, `R.`, `Avenue`, `Av`, `Boulevard`, `Bd`, `Allée`, `Place`, `Chemin`, `Impasse`, along with modifier terms `bis`, `ter`.
- **Component Permutation / Reordering**:
  - Standard US: `Street, City, State ZIP`
  - Reordered: `State, City, Street, Unit` (e.g. `IA, Iowa City, 1064 Newton Rd, Unit 11`)
- **Landmark-Based References (Indian Context)**: `Near SBI ATM`, `Opposite Police Station`, `Behind Bus Stand`, `Plot No`, `Khasra No` (`KH NO. -570/13`).
- **Transliteration & Geographic Discrepancies**: Alternate city names (e.g. `Gurugram` vs `Gurgaon`, `Bengaluru` vs `Bangalore`, `Mumbai` vs `Bombay`).
- **Missing Address Components**: Missing postal PIN code, missing state, or completely missing address (`null`/empty in ~3.4% of records).

### 9.3 Country Generalization Challenge (Zero-Shot France)

- The model must generalize to `France` without any training ground truth for French entities.
- Rule: Do NOT train models with features or pipelines that fail or break on French vocabulary, French address patterns, French characters (`é`, `è`, `ê`, `à`, `ç`, `î`, `ô`), or hardcoded US/India assumptions.

---

## 10. Official Evaluation Metric

Submissions are evaluated using the **Macro-Averaged $F_\beta$ Score with $\beta = 0.5$ ($F_{0.5}$)**.

- **Metric Philosophy**: Precision-heavy metric. Merging two different businesses (False Positive) is penalised twice as severely as missing a link (False Negative).
- **Evaluation Unit**: Per Source 1 entity in the evaluation set.
- **Aggregation**: Unweighted arithmetic mean across all Source 1 entities in the evaluation split (Macro-Average). Singletons (entities with 0 true matches) are included directly in this average.

---

## 11. Exact F0.5 Definition

For any single Source 1 entity $i$, let:

- $T_i$ be the set of true matching IDs from ground truth ($T_i \subset \mathcal{S}_2 \cup \mathcal{S}_3$).
- $P_i$ be the set of predicted matching IDs ($P_i \subset \mathcal{S}_2 \cup \mathcal{S}_3$).

### Case A: Non-Singleton Ground Truth ($|T_i| > 0$)

If $|P_i| = 0$:

$$
\text{Precision}_i = 0, \quad \text{Recall}_i = 0 \implies F_{0.5, i} = 0.0
$$

If $|P_i| > 0$:

$$
\text{TP}_i = |P_i \cap T_i|
$$

$$
\text{Precision}_i = \frac{\text{TP}_i}{|P_i|}
$$

$$
\text{Recall}_i = \frac{\text{TP}_i}{|T_i|}
$$

If $\text{TP}_i = 0$:

$$
F_{0.5, i} = 0.0
$$

If $\text{TP}_i > 0$:

$$
F_{0.5, i} = \frac{(1 + 0.5^2) \times \text{Precision}_i \times \text{Recall}_i}{(0.5^2 \times \text{Precision}_i) + \text{Recall}_i} = \frac{1.25 \times \text{Precision}_i \times \text{Recall}_i}{0.25 \times \text{Precision}_i + \text{Recall}_i}
$$

### Case B: Singleton Ground Truth ($|T_i| = 0$)

- **Correct Singleton Prediction**: If $|P_i| = 0$ (predicted empty list):
  $$
  F_{0.5, i} = 1.0
  $$
- **False Merge on Singleton**: If $|P_i| > 0$ (predicted any matches):
  $$
  F_{0.5, i} = 0.0
  $$

---

## 12. Macro-Averaging Definition

Let $N$ be the total number of Source 1 entities in the evaluation set ($N = 1,732,544$ for full test; or $N_{\text{val}}$ for a validation fold).

$$
\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^{N} F_{0.5, i}
$$

### Critical Distinctions:

1. **NOT Micro-Averaged**: Micro-averaging aggregates total global TPs, FPs, and FNs across all pairs before calculating a single $F_{0.5}$. The official competition metric is **NOT** micro-averaged.
2. **Equal Weight per Entity**: An entity with 1 match contributes exactly the same weight ($1/N$) to the final score as an entity with 10 matches or a singleton entity with 0 matches.
3. **Singleton Contribution**: In the training set, 5.585% of entities are singletons. Correctly predicting empty lists for singletons provides a direct score floor of up to $+0.05585$. Conversely, making a single false guess on a singleton instantly drops its score from $1.0 \rightarrow 0.0$.

---

## 13. TP / FP / FN Semantics

For an individual Source 1 entity $i$:

| Metric Component                           | Mathematical Set Definition | Practical Meaning                                                                         |
| :----------------------------------------- | :-------------------------- | :---------------------------------------------------------------------------------------- |
| **True Positive ($\text{TP}_i$)**  | $|P_i \cap T_i|$          | Correctly predicted auxiliary records that belong to entity$i$.                         |
| **False Positive ($\text{FP}_i$)** | $|P_i \setminus T_i|$     | Incorrectly predicted records (false merges; linking a different business or distractor). |
| **False Negative ($\text{FN}_i$)** | $|T_i \setminus P_i|$     | Missed records (failing to link a genuine business match).                                |

### Precision Weighting Impact:

$$
\frac{\partial F_{0.5}}{\partial \text{Precision}} = 2 \times \frac{\partial F_{0.5}}{\partial \text{Recall}}
$$

A false positive hurts the score substantially more than a false negative.

- Example: If $|T_i| = 2$:
  - Predict 2 correct: $P=1.0, R=1.0 \implies F_{0.5} = 1.000$
  - Predict 2 correct + 1 wrong: $P=0.667, R=1.0 \implies F_{0.5} = 0.714$ (Loss of 0.286)
  - Predict 1 correct + 0 wrong: $P=1.0, R=0.5 \implies F_{0.5} = 0.833$ (Loss of only 0.167)
- **Conclusion**: Missing a match is far preferable to asserting an uncertain match.

---

## 14. Evaluation Pseudocode

This exact logic matches the official evaluation guidelines and must be used for all local validation scoring:

```python
def compute_entity_f05(y_true_set: set, y_pred_set: set) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    """
    # Case 1: Ground truth is singleton (no matches)
    if len(y_true_set) == 0:
        return 1.0 if len(y_pred_set) == 0 else 0.0

    # Case 2: Ground truth has matches, but prediction is empty
    if len(y_pred_set) == 0:
        return 0.0

    # Case 3: Both have entries
    tp = len(y_true_set & y_pred_set)
    if tp == 0:
        return 0.0

    precision = tp / len(y_pred_set)
    recall = tp / len(y_true_set)

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return float(f05)


def compute_macro_f05(
    ground_truth_dict: dict[str, set],
    predictions_dict: dict[str, set],
    all_s1_ids: list[str],
) -> float:
    """
    Computes official macro-averaged F_0.5 across all Source 1 entities.
    ground_truth_dict: {s1_id: set(target_ids)}
    predictions_dict:  {s1_id: set(target_ids)}
    all_s1_ids:        list of all S1 IDs in evaluation set
    """
    total_score = 0.0
    n = len(all_s1_ids)
    assert n > 0, "Evaluation set cannot be empty"

    for s1_id in all_s1_ids:
        y_true = ground_truth_dict.get(s1_id, set())
        y_pred = predictions_dict.get(s1_id, set())
        total_score += compute_entity_f05(y_true, y_pred)

    return total_score / n
```

---

## 15. Evaluation Optimization Implications

1. **Precision Dominance**: Speculative candidate emission destroys $F_{0.5}$. The decision threshold $\theta_{\text{match}}$ must be tuned strictly to maximize Macro $F_{0.5}$, yielding higher thresholds than standard $F_1$ models ($\theta > 0.5$ typically).
2. **Singleton Protection**: Because singletons account for ~5.6% of entities, if an entity has low maximum candidate match probability (e.g. $\max P(\text{match}) < \tau_{\text{singleton}}$), predicting an empty list immediately guarantees a perfect $1.0$ score on that entity.
3. **No Target Contamination**: If model predicts candidate $c$ for $S_1^A$ with confidence $0.92$ and for $S_1^B$ with confidence $0.51$, candidate $c$ must be assigned exclusively to $S_1^A$ (or filtered), because true targets never link to multiple S1 records.

---

## 16. Candidate Generation Requirements

Candidate Generation (Blocking) is the **High-Recall Gatekeeper**:

- **Recall Ceiling**: Downstream matching models can ONLY predict matches that survive candidate generation. Any true pair missed during blocking is permanently lost ($\text{Recall} = 0$).
- **Metric**: Must measure **Candidate Recall** on training/validation:
  $$
  \text{Candidate Recall} = \frac{\sum_{i} |T_i \cap C_i|}{\sum_{i} |T_i|}
  $$
- **Reduction Ratio**: Must reduce the comparison space from $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.72 \times 10^{13}$ pairs down to a computationally feasible candidate pool ($K \le 50$ candidates per S1 entity on average).
- **Candidate Output**: Candidate generation must output the exact candidate set into `candidate_pairs.tsv`.
- **Subset Invariance**: Every ID predicted in `matching_results.tsv` must exist in `candidate_pairs.tsv`.

---

## 17. Final Matching Requirements

Final Matching is the **Precision-Calibrated Scorer**:

- Takes the candidates produced by blocking ($C_i$ for each $S_1$).
- Computes pairwise similarity and alignment features across names, addresses, and geographic tokens.
- Produces a calibrated probability or match score $s(e_1, e_{\text{aux}})$.
- Employs optimal decision logic (thresholding, ranking, cluster resolution) to output final matches $\mathcal{M}(e_1) \subseteq C_i$.

---

## 18. No-Match Rules

- An entity with no true matches is a **Singleton**.
- **Representation in Output**: In both `matching_results.tsv` and `candidate_pairs.tsv`, an entity with no matches must be represented by a line with the entity ID, followed by a TAB, followed immediately by a newline (`\n`), i.e., an empty string for the second column:
  ```
  S1-00003\t\n
  ```
- **Scoring Behavior**:
  - Predicted empty for true singleton $\implies F_{0.5} = 1.0$.
  - Predicted non-empty for true singleton $\implies F_{0.5} = 0.0$.
- Any S1 record without sufficiently strong candidates must be emitted as empty.

---

## 19. Multiple-Match Rules

- A single Source 1 entity CAN match multiple auxiliary entities (1 to 11 matches observed in ground truth).
- Multiple matches occur when multiple records in Source 2 and/or Source 3 refer to the same physical establishment (e.g. variations of trade name, different regional directories).
- **Representation**: Comma-separated list of IDs with **no spaces**, no quotes, no brackets:
  ```
  S1-00001\tS2-00047,S2-00193,S3-00812\n
  ```
- **No Duplicate IDs**: A single ID must never appear twice in the list for the same $S_1$ entity (e.g. `S2-00047,S2-00047` is strictly rejected by the validator).
- **No S1 IDs**: The matched ID list must contain ONLY `S2-` and `S3-` IDs.

---

## 20. Required Output Files

Every complete submission requires two (2) distinct output files located in the `output/` folder:

1. **`matching_results.tsv`**
   - **Role**: Final predicted matches.
   - **Leaderboard Impact**: **THIS IS THE ONLY FILE SCORED ON THE LEADERBOARD**.
   - **Contents**: The pruned, high-confidence entity matches produced by the final matching model.
2. **`candidate_pairs.tsv`**
   - **Role**: The candidate set from the blocking/candidate-generation stage.
   - **Leaderboard Impact**: Not scored on the leaderboard; used by organizers to audit blocking quality, recall ceiling, and verify pipeline consistency.
   - **Constraint**: Must represent the candidate set fed into the final matching model. Final matches must be a subset of these candidates.

---

## 21. Exact Output Schemas

### 21.1 `output/matching_results.tsv`

- **File Format**: Plain UTF-8 text, Tab-Separated Values (`.tsv`).
- **Header Line**: `source1_entity_id\tmatched_entity_ids` (exact casing).
- **Row Count**: Exactly 1,732,544 data rows + 1 header line = 1,732,545 lines.
- **Columns**:
  1. `source1_entity_id`: Exact string ID from `test_source1.tsv`. Every S1 test entity must appear exactly once.
  2. `matched_entity_ids`: Comma-separated string of `S2-` and `S3-` IDs, or empty string.

### 21.2 `output/candidate_pairs.tsv`

- **File Format**: Plain UTF-8 text, Tab-Separated Values (`.tsv`).
- **Header Line**: `source1_entity_id\tcandidate_entity_ids` (exact casing).
- **Row Count**: Exactly 1,732,544 data rows + 1 header line = 1,732,545 lines.
- **Columns**:
  1. `source1_entity_id`: Exact string ID from `test_source1.tsv`. Every S1 test entity must appear exactly once.
  2. `candidate_entity_ids`: Comma-separated string of candidate `S2-` and `S3-` IDs, or empty string.

---

## 22. Output Examples

### 22.1 `matching_results.tsv` Example:

```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003
S1-00004	S2-99120
```

### 22.2 `candidate_pairs.tsv` Example:

```tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999,S2-00441
S1-00002	S3-00004,S2-11002
S1-00003
S1-00004	S2-99120,S3-00118
```

*(Notice: For `S1-00001`, the final matches `{S2-00047, S2-00193, S3-00812}` are a strict subset of the candidate set).*

---

## 23. Submission Package Structure

For final evaluation, the top teams submit a single zip archive: `<team_name>_submission.zip`.

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # Scored leaderboard file
│   └── candidate_pairs.tsv         # Candidate pairs from blocking
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # All source code (.py scripts, modules)
│       ├── README.md               # End-to-end reproduction instructions
│       └── requirements.txt        # Pinned dependencies
└── Documentation_template.md       # Filled-in methodology write-up (or .pdf)
```

---

## 24. Submission Validation Checklist

The local validation script `utils/validate_submission.py` must run with exit code 0 (`PASS`) prior to submitting.

Verification commands:

```bash
# Standard fast check
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# Full existence diagnostic check
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

### Mandatory Verification Gates:

- [X] File exists and is named `matching_results.tsv`
- [X] Plain UTF-8 encoding (not Latin-1, cp1252, or binary)
- [X] True Tab separator (`\t`), NOT comma (`,`)
- [X] Header is strictly `source1_entity_id\tmatched_entity_ids`
- [X] Exactly 1,732,544 data rows matching `test_source1.tsv`
- [X] Zero duplicate `source1_entity_id` rows
- [X] Zero missing `source1_entity_id` rows
- [X] Zero extraneous / invented `source1_entity_id` rows
- [X] Zero self-matches (no `S1-` IDs in matched list)
- [X] All IDs in matched list have valid `S2-` or `S3-` prefixes
- [X] Zero duplicate IDs inside any single comma-separated list
- [X] Singletons formatted with empty string after tab
- [X] `candidate_pairs.tsv` exists with header `source1_entity_id\tcandidate_entity_ids`
- [X] All matched IDs in `matching_results.tsv` are a subset of candidate IDs in `candidate_pairs.tsv`

---

## 25. Allowed Resources

- **Training Dataset Provided**: `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`.
- **Test Dataset Provided**: `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`.
- **Open-Source ML Frameworks**: Python standard library, NumPy, Pandas, Scipy, Scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, Hugging Face Transformers, FastText, FAISS.
- **Model Licensing & Size Constraint**: Any pre-trained model must have an **MIT or Apache 2.0 License** and have a size of **up to 8 Billion parameters**.
- **Local Preprocessing / Encoders**: Multilingual string distance algorithms, phonetic encodings (Soundex, Metaphone), n-gram tokenizers, character embeddings.

---

## 26. Prohibited Resources

### ⚠️ STRICTLY PROHIBITED: External Data Lookup

Participants are **STRICTLY NOT ALLOWED** to use external databases, APIs, or services to look up business identities or resolve entities.

Explicitly Prohibited:

- ❌ **NO Google Search / Google Places / Google Maps Lookups**
- ❌ **NO LinkedIn / Social Media Scraping or Lookups**
- ❌ **NO Company Website Crawling or Domain Resolving**
- ❌ **NO External Business Databases** (e.g. OpenCorporates, Dun & Bradstreet, ZoomInfo, Crunchbase)
- ❌ **NO External Entity Resolution APIs or Commercial Services**
- ❌ **NO Government Corporate Registrations Lookups** (e.g. MCA India, SEC EDGAR, French SIRENE/INSEE registry)
- ❌ **NO Geocoding APIs** (e.g. Nominatim, Google Geocoding, Mapbox, HERE) to normalize addresses
- ❌ **NO Internet-based Business Verification of any kind**

**Penalty**: Automatic and immediate disqualification. Top solutions will undergo manual code and network audit.

---

## 27. Academic Integrity / Fair Play

1. **Closed-World Entity Resolution**: The problem must be solved using only the statistical, linguistic, and phonetic patterns present in the provided training and test corpora.
2. **Deterministic Reproducibility**: Organizers will execute `code/business_entity_resolution/` from scratch to verify that `matching_results.tsv` and `candidate_pairs.tsv` are reproduced exactly.
3. **No Target Peeking / Manual Annotation**: Manual labeling of test entities is strictly forbidden.

---

## 28. Validation Strategy

To reliably maximize the Private Leaderboard score without leaking or overfitting:

### 28.1 Country-Stratified, Entity-Disjoint Cross Validation

- Ground truth contains 2,206,821 Source 1 entities.
- Hold out a clean, representative validation split (e.g. 10% = ~220,682 S1 entities, or 5-fold CV).
- Split must be **Entity-Disjoint**: All links associated with a validation $S_1$ entity must be strictly isolated to the validation set.
- Stratify across:
  1. Country (`US` vs `India`)
  2. Entity Cardinality (Singleton vs 1 match vs 2+ matches)

### 28.2 Zero-Shot Country Transfer Simulation (The France Proxy)

Because the test set introduces `France` (which has 0 training labels):

- Build a dedicated validation split where one country (or language/region) is held out completely during feature tuning and threshold selection to measure how well the pipeline transfers out-of-distribution without country-specific memorization.

### 28.3 Mandatory Experiment Metrics to Report:

Every experiment must log:

1. **Official Macro $F_{0.5}$** (Primary Metric)
2. **Macro Precision** and **Macro Recall**
3. **Candidate Recall** (% of true matches captured by blocking)
4. **Reduction Ratio** (Candidate pool size vs Cartesian space)
5. **Singleton Accuracy** (Precision on zero-match S1 entities)
6. **Error Counts**: False Positives vs False Negatives

---

## 29. Leakage Prevention

1. **No Target Leakage in Candidate Generation**: Candidate generation index must never use ground truth labels.
2. **No Fit-on-Validation**: TF-IDF vocabularies, frequency tables, and embeddings must be fit either strictly on training folds or unsupervised across raw text without ground truth link supervision.
3. **Independent Auxiliary Evaluation**: Auxiliary records in the validation pool must include both genuine matches and realistic distractor records (~26% distractors as observed in ground truth).
4. **Cross-Entity Assignment Safety**: In post-processing, enforce the invariant that target IDs cannot be assigned to multiple S1 entities.

---

## 30. Recommended ML Architecture

```
[ Raw S1, S2, S3 Records ]
           │
           ▼
┌────────────────────────────────────────┐
│ 1. DATA AUDIT & CANONICALIZATION       │
│    - UTF-8 normalization (NFKC)        │
│    - Lowercasing, punctuation handling │
│    - Legal suffix normalization        │
│    - Transliteration preservation      │
│    - Street/Address token standard     │
└────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│ 2. BLOCKING & CANDIDATE GENERATION     │
│    - Hard Country Partition (US/IN/FR) │
│    - Multi-Index Union Blocking:       │
│      * Exact normalized name match     │
│      * MinHash LSH / TF-IDF Char N-gram│
│      * High-frequency token inversion  │
│      * Address postal/city token index │
│    - Output: candidate_pairs.tsv       │
│    - Target: Candidate Recall > 98%    │
└────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│ 3. PAIRWISE FEATURE ENGINEERING        │
│    - Name Similarities:                │
│      * Jaro-Winkler, Levenshtein ratio │
│      * Token Sort / Set Ratio          │
│      * N-gram Jaccard (char 2,3,4)     │
│      * Legal entity match indicator    │
│    - Address Similarities:             │
│      * Address token intersection      │
│      * Numeric token match (house/pin) │
│      * City / State / Region alignment │
│    - Cross-Source Prior & Metadata:    │
│      * S2 vs S3 indicator              │
│      * Missing address indicator       │
└────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│ 4. MATCH SCORING & PROBABILITY         │
│    - Gradient Boosted Decision Trees   │
│      (LightGBM / XGBoost / CatBoost)   │
│    - High throughput pairwise scorer   │
│    - Calibrated match probabilities    │
└────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│ 5. GLOBAL RESOLUTION & CLUSTERING      │
│    - Enforce Target Uniqueness         │
│      (Max-weight bipartite matching or │
│       greedy 1-to-1 target assignment) │
│    - Singleton Decision Gate           │
└────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│ 6. THRESHOLD OPTIMIZATION (F_0.5)      │
│    - Optimize threshold θ on Macro F0.5│
│    - Set high-precision margin filter  │
│    - Prune uncertain matches           │
│    - Output: matching_results.tsv      │
└────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│ 7. MECHANICAL PRE-SUBMISSION VALIDATION│
│    - Run validate_submission.py        │
│    - Check exit code == 0              │
└────────────────────────────────────────┘
```

---

## 31. Feature Engineering Strategy

Features must be computationally fast, robust across US, India, and France, and operate without external lookups.

### 31.1 Business Name Alignment

- `name_exact_match`: Boolean flag indicating exact normalized string equality.
- `name_clean_exact`: Match after stripping legal suffixes (`Inc`, `Corp`, `Pvt Ltd`, `SARL`, etc.).
- `name_jaro_winkler`: Captures prefix agreements and typographical slips.
- `name_token_sort_ratio`: Robust against word order permutations (`Vision Partners` vs `Partners Vision`).
- `name_token_set_ratio`: Handles subset names (`Zephay Labs` vs `Zephay Labs International Inc`).
- `name_char_ngram_jaccard_3`: Character 3-gram overlap, highly effective for transliteration variants and slight phonetic spelling shifts.
- `name_length_diff`: Absolute and relative string length differences.
- `name_initial_acronym_match`: Detects acronym expansions (`IBM` vs `International Business Machines`).

### 31.2 Address Alignment

- `addr_exact_match`: Boolean flag.
- `addr_token_jaccard`: Word token overlap.
- `addr_numeric_overlap`: Jaccard similarity of extracted numbers (critical for house numbers, street numbers, PIN codes).
- `addr_is_empty`: Boolean flag for missing address in auxiliary record.
- `addr_char_ngram_3`: Substring overlap for partial addresses and landmark descriptions.

### 31.3 Country & Geographic Consistency

- `country_exact`: Hard invariant ($= 1.0$).
- `geo_token_overlap`: Overlap of city/state/region tokens.

---

## 32. Threshold Optimization

Because the metric is **Macro $F_{0.5}$**, default decision thresholds ($\theta = 0.5$) are suboptimal.

### Parameter Space to Optimize on Validation Split:

1. **$\theta_{\text{match}}$ (Match Confidence Threshold)**: Minimum probability required to assert a match between $S_1$ and auxiliary candidate. Due to $\beta = 0.5$ precision penalization, optimal $\theta$ is typically in $[0.65, 0.85]$.
2. **$\tau_{\text{singleton}}$ (Singleton Cutoff)**: If an entity's top candidate has score $< \tau_{\text{singleton}}$, emit an empty list immediately to secure a guaranteed $1.0$ score.
3. **$\Delta_{\text{margin}}$ (Competitive Margin)**: If target $t$ is scored for multiple $S_1$ entities, assign to $\arg\max_{S_1} s(S_1, t)$ provided score exceeds runner-up by margin $\Delta$.

---

## 33. Error Analysis

Systematic error tracking must categorize all mistakes into actionable buckets:

1. **False Positives (Precision Killers — 2x penalty)**:
   - *Common Name Collisions*: Different businesses sharing generic names (e.g. "Apex Enterprises" in different cities).
   - *Address Blindness*: Matched on name without verifying conflicting city/address.
   - *Singleton Contamination*: Asserting a weak match on a true singleton record.
2. **False Negatives (Recall Losses)**:
   - *Blocking Drops*: True match never entered candidate pool (candidate recall failure).
   - *Extreme Script / Transliteration Variance*: Native script vs English name without transliteration alignment.
   - *Missing Address Penalty*: True matches rejected because address was missing in Source 2/3.

---

## 34. Experiment Tracking

Every iteration must record the following tabular schema:

```
experiment_id:          EXP-001
model_type:             LightGBM Pairwise Classifier
blocking_strategy:      Country + Name 3-Gram MinHash (K=30)
candidate_recall:       0.9842
validation_macro_f05:   0.8415
validation_precision:   0.8920
validation_recall:      0.7130
singleton_accuracy:     0.9410
num_features:           18
best_threshold:         0.74
inference_time_sec:     420s
submission_pass:        YES
notes:                  Added French legal suffixes; tuned singleton threshold.
```

---

## 35. Public Leaderboard

- **Subset Evaluation**: Public leaderboard scores are computed on a subset of the test records.
- **Role**: Diagnostic sanity check and submission format confirmation (`SCORED` status).
- **Risk**: Overfitting to the public split by submitting repeatedly and tweaking thresholds to match public feedback.

---

## 36. Private Leaderboard

- **Authoritative Ranking**: **The final competition winner is determined exclusively by the Private Leaderboard**.
- Evaluated on the remaining portion of the test set after the competition closes.
- Generalization to unseen records and to `France` governs the final outcome.
- **Rule**: Trust robust, leakage-free local cross-validation over minor public leaderboard fluctuations.

---

## 37. Competition Optimization Principles

1. **Precision First**: In $F_{0.5}$, Precision is weighted twice as heavily as Recall. When in doubt, do not match.
2. **Never Drop Candidates Early**: In blocking, optimize for Recall ($> 98\%$). The classifier can filter false positives, but cannot resurrect dropped true matches.
3. **Target Mutual Exclusivity**: Enforce 1-to-at-most-1 target assignment. Auxiliary records do not belong to multiple Source 1 entities.
4. **Respect Singletons**: 5.58% of Source 1 entities have zero true matches. Accurately predicting empty lists earns a free $+0.0558$ to macro $F_{0.5}$.
5. **Country Zero-Shot Robustness**: Design all name and address normalization to work natively across English, Indian transliterations, and French language patterns.
6. **Mechanical Verification**: Always execute `validate_submission.py` locally before uploading.

---

## 38. Implementation Order

All subsequent development must proceed in this exact sequence:

1. **Phase 1: Exploratory Data Audit & Ground Truth Analysis** (Completed in SSOT creation).
2. **Phase 2: Local Macro $F_{0.5}$ Evaluation Harness** (Exact script matching competition metric).
3. **Phase 3: High-Recall Multi-Index Candidate Generation** (Measure Candidate Recall on train).
4. **Phase 4: Pairwise Feature Engineering Pipeline** (Vectorized, scalable across millions of pairs).
5. **Phase 5: Model Training & Probability Calibration** (LightGBM/XGBoost).
6. **Phase 6: Threshold & Singleton Optimization** (Grid-search $\theta$ for maximum Macro $F_{0.5}$).
7. **Phase 7: Full Test Inference & Output Generation** (`matching_results.tsv` and `candidate_pairs.tsv`).
8. **Phase 8: Pre-Submission Verification & Package Assembly** (`validate_submission.py` pass and zip archive creation).

---

## 39. Final Pre-Submission Quality Gate

Before submitting `matching_results.tsv` or packaging `<team_name>_submission.zip`:

- [ ] `validate_submission.py` reports `PASS — no blocking issues found. Safe to submit.`
- [ ] Exactly 1,732,544 rows in `matching_results.tsv` matching test S1 entity IDs.
- [ ] Exactly 1,732,544 rows in `candidate_pairs.tsv` matching test S1 entity IDs.
- [ ] Every predicted match in `matching_results.tsv` is present in `candidate_pairs.tsv`.
- [ ] No `S1-` IDs present in matched or candidate lists.
- [ ] No duplicate IDs in any list.
- [ ] Country invariant verified (all matches belong to the same country).
- [ ] Dependencies documented in `requirements.txt`.
- [ ] Reproducibility instructions documented in `code/business_entity_resolution/README.md`.
- [ ] Methodology documented in `Documentation_template.md`.

---

## 40. UNKNOWN / AMBIGUOUS ITEMS

The following items are not explicitly specified in the official documentation and require verification or deliberate conservative handling:l

1. **Public/Private Split Ratio**:
   - *Status*: UNKNOWN — VERIFY FROM OFFICIAL SOURCE
   - *Handling*: Assume standard 30%/70% or 50%/50% split. Do not overfit to public score.
2. **Submission Quota per Day**:
   - *Status*: UNKNOWN — VERIFY FROM OFFICIAL SOURCE
   - *Handling*: Conserve submissions. Verify all outputs locally with `validate_submission.py` before submitting.
3. **Exact Team Zip Filename Prefix**:
   - *Status*: Template specifies `<team_name>_submission.zip`. Replace `<team_name>` with registered team name on portal.
4. **Hardware Execution Environment for Auditing**:
   - *Status*: UNKNOWN — VERIFY FROM OFFICIAL SOURCE
   - *Handling*: Ensure pipeline runs within standard memory (<= 32GB RAM / 16GB VRAM) and finishes within reasonable runtime (< 4 hours end-to-end).
