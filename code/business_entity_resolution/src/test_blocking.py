"""
ML Challenge 2026: Business Entity Resolution
Benchmark Candidate Generation (Blocking) Recall on Validation Set
"""

import os
import sys
import time
import pandas as pd

sys.path.insert(0, ".")
from code.business_entity_resolution.src.blocking import MultiIndexBlocker
from code.business_entity_resolution.src.evaluation import compute_macro_f05

def run_blocking_benchmark(val_dir: str = "validation"):
    print("=" * 60)
    print("RUNNING CANDIDATE GENERATION (BLOCKING) BENCHMARK")
    print("=" * 60)

    # 1. Load Ground Truth
    gt_df = pd.read_csv(os.path.join(val_dir, "val_ground_truth.tsv"), sep="\t")
    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")
    gt_dict = {
        row["source1_entity_id"]: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"].strip() else set()
        for _, row in gt_df.iterrows()
    }
    all_s1_ids = list(gt_df["source1_entity_id"])

    # 2. Load Auxiliary Records
    s2_df = pd.read_csv(os.path.join(val_dir, "val_source2.tsv"), sep="\t")
    s3_df = pd.read_csv(os.path.join(val_dir, "val_source3.tsv"), sep="\t")
    aux_records = s2_df.to_dict("records") + s3_df.to_dict("records")
    print(f"Loaded {len(aux_records)} total auxiliary records.")

    # 3. Build Index
    t0 = time.time()
    blocker = MultiIndexBlocker(max_candidates_per_entity=150)
    blocker.build_auxiliary_index(aux_records)
    t_index = time.time() - t0
    print(f"Index built in {t_index:.2f} seconds.")

    # 4. Generate Candidates for S1
    s1_df = pd.read_csv(os.path.join(val_dir, "val_source1.tsv"), sep="\t")
    t1 = time.time()
    cand_dict = blocker.batch_generate_candidates(s1_df)
    t_gen = time.time() - t1
    print(f"Generated candidates for {len(s1_df)} S1 entities in {t_gen:.2f} seconds.")

    # 5. Measure Candidate Recall
    total_true_matches = 0
    captured_matches = 0
    candidate_counts = []

    for s1_id in all_s1_ids:
        y_true = gt_dict.get(s1_id, set())
        cands = cand_dict.get(s1_id, set())
        candidate_counts.append(len(cands))
        if len(y_true) > 0:
            total_true_matches += len(y_true)
            captured_matches += len(y_true & cands)

    cand_recall = captured_matches / total_true_matches if total_true_matches > 0 else 0.0
    avg_cands = sum(candidate_counts) / len(candidate_counts)
    max_cands = max(candidate_counts)
    zero_cands = sum(1 for c in candidate_counts if c == 0)

    print("\n" + "=" * 60)
    print("BLOCKING RESULTS SUMMARY:")
    print(f"  Total True Matches:        {total_true_matches}")
    print(f"  Captured in Candidates:    {captured_matches}")
    print(f"  CANDIDATE RECALL CEILING:  {cand_recall:.4%}")
    print(f"  Average Candidates per S1: {avg_cands:.2f}")
    print(f"  Max Candidates per S1:     {max_cands}")
    print(f"  Entities with 0 Candidates:{zero_cands} ({zero_cands / len(all_s1_ids):.2%})")
    print("=" * 60)

    return blocker, cand_dict, gt_dict

if __name__ == "__main__":
    run_blocking_benchmark()
