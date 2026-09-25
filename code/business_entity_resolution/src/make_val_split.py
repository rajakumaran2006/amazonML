"""
ML Challenge 2026: Business Entity Resolution
Script to generate a clean, entity-disjoint, stratified validation benchmark.
"""

import os
import sys
import pandas as pd
import numpy as np

def create_validation_split(
    data_dir: str = "dataset/train",
    val_dir: str = "validation",
    val_size: int = 20000,
    random_seed: int = 42,
    distractor_ratio: float = 0.25,
):
    os.makedirs(val_dir, exist_ok=True)
    print(f"Creating validation split with {val_size} S1 entities...")

    # 1. Read S1 and Ground Truth
    print("Loading train_source1.tsv and train_ground_truth.tsv...")
    s1_df = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t")
    gt_df = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t")

    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")
    merged = pd.merge(s1_df, gt_df, left_on="entity_id", right_on="source1_entity_id")

    # Stratify by country and singleton status
    merged["is_singleton"] = merged["matched_entity_ids"].str.strip() == ""
    merged["strata"] = merged["country"] + "_" + merged["is_singleton"].astype(str)

    # Sample stratified validation S1
    val_s1 = merged.groupby("strata", group_keys=False).apply(
        lambda x: x.sample(n=int(round(val_size * len(x) / len(merged))), random_state=random_seed)
    )
    print(f"Validation S1 entities sampled: {len(val_s1)}")
    print(val_s1["country"].value_counts())
    print(f"Singletons: {val_s1['is_singleton'].sum()}")

    val_s1_ids = set(val_s1["entity_id"])

    # Save val_source1.tsv and val_ground_truth.tsv
    val_s1[["entity_id", "business_name", "business_address", "country"]].to_csv(
        os.path.join(val_dir, "val_source1.tsv"), sep="\t", index=False
    )
    val_s1[["source1_entity_id", "matched_entity_ids"]].to_csv(
        os.path.join(val_dir, "val_ground_truth.tsv"), sep="\t", index=False
    )

    # Collect all true matched S2 and S3 IDs for validation S1
    val_matched_ids = set()
    for mids in val_s1["matched_entity_ids"]:
        if mids.strip():
            val_matched_ids.update(mids.split(","))

    print(f"Total true auxiliary matches in validation: {len(val_matched_ids)}")

    # 2. Extract matching records from train_source2 and train_source3 + distractors
    for src in [2, 3]:
        src_file = os.path.join(data_dir, f"train_source{src}.tsv")
        print(f"Filtering {src_file} for validation...")
        chunks = []
        # Stream read in chunks to keep memory footprint under 500MB
        for chunk in pd.read_csv(src_file, sep="\t", chunksize=250000):
            # Keep records that match validation S1
            matched_chunk = chunk[chunk["entity_id"].isin(val_matched_ids)]
            if len(matched_chunk) > 0:
                chunks.append(matched_chunk)
            
            # Sample distractors (records not matching validation S1)
            distractor_cand = chunk[~chunk["entity_id"].isin(val_matched_ids)]
            n_dist = int(len(matched_chunk) * distractor_ratio)
            if n_dist > 0 and len(distractor_cand) > 0:
                chunks.append(distractor_cand.sample(n=min(n_dist, len(distractor_cand)), random_state=random_seed))

        val_src_df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["entity_id"])
        out_path = os.path.join(val_dir, f"val_source{src}.tsv")
        val_src_df.to_csv(out_path, sep="\t", index=False)
        print(f"Saved {out_path} with {len(val_src_df)} records.")

    print(f"Validation split created successfully in '{val_dir}'!")

if __name__ == "__main__":
    create_validation_split()
