"""
ML Challenge 2026: Business Entity Resolution
Module: High-Performance AWS & Local Inference Engine (Multi-Core Optimized)

Generates:
1. output/matching_results.tsv  (Scored on Leaderboard)
2. output/candidate_pairs.tsv   (Audit candidate pairs)

Optimizations for AWS:
- Automatically utilizes all available vCPUs (n_jobs=-1)
- Fast memory-mapped loading when RAM >= 16 GB
- Progress logging with ETA
- Direct validation call at completion
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, ".")
from code.business_entity_resolution.src.preprocess import (
    clean_business_name,
    extract_address_tokens,
    phonetic_skeleton,
)
from code.business_entity_resolution.src.blocking import MultiIndexBlocker
from code.business_entity_resolution.src.features import extract_pair_features


def load_auxiliary_records_for_country(test_dir: str, target_country: str) -> List[Dict]:
    """
    Loads all auxiliary records (S2 + S3) for the target country.
    """
    records = []
    t0 = time.time()
    print(f"\n[Data Loader] Loading test_source2 and test_source3 for {target_country}...")
    for src in [2, 3]:
        file_path = os.path.join(test_dir, f"test_source{src}.tsv")
        # Read in chunks of 500k to balance speed and memory
        for chunk in pd.read_csv(file_path, sep="\t", chunksize=500000):
            filtered = chunk[chunk["country"] == target_country]
            if len(filtered) > 0:
                records.extend(filtered.to_dict("records"))
    print(f"[Data Loader] Loaded {len(records)} records for {target_country} in {time.time() - t0:.1f}s.")
    return records


def run_country_inference(
    s1_country_df: pd.DataFrame,
    aux_records: List[Dict],
    model,
    match_threshold: float,
    singleton_cutoff: float,
    matching_f,
    candidate_f,
    batch_size: int = 25000,
):
    """
    Constructs blocking index, generates candidates, extracts features,
    and runs multi-threaded LightGBM inference for one country.
    """
    t_start = time.time()
    country_name = s1_country_df["country"].iloc[0]

    # 1. Build Multi-Index Blocker (with posting cap to prevent memory thrashing)
    t0 = time.time()
    blocker = MultiIndexBlocker(max_candidates_per_entity=35, max_posting=250)
    blocker.build_auxiliary_index(aux_records)
    print(f"[Blocking Index] Built index for {country_name} in {time.time() - t0:.1f}s.")

    n_total = len(s1_country_df)
    n_batches = int(np.ceil(n_total / batch_size))
    country_assigned_aux = set()

    print(f"[Inference] Running inference across {n_total:,} {country_name} entities in {n_batches} batches...")

    for b_idx in range(n_batches):
        tb0 = time.time()
        b_start = b_idx * batch_size
        b_end = min(b_start + batch_size, n_total)
        batch_df = s1_country_df.iloc[b_start:b_end]

        # Fast S1 Preprocessing using itertuples
        s1_records = {}
        for row in batch_df.itertuples(index=False):
            sid = row.entity_id
            name = str(row.business_name) if pd.notna(row.business_name) else ""
            addr = str(row.business_address) if pd.notna(row.business_address) else ""
            c_name = clean_business_name(name)
            addr_words, addr_nums = extract_address_tokens(addr)
            s1_records[sid] = {
                "id": sid,
                "name": name,
                "clean_name": c_name,
                "compact_name": c_name.replace(" ", ""),
                "first_word": c_name.split()[0] if c_name.split() else "",
                "skel": phonetic_skeleton(c_name),
                "addr_tokens": set(addr_words),
                "addr_nums": addr_nums,
                "country": str(row.country).strip(),
            }

        # Candidate Generation
        cand_dict = blocker.batch_generate_candidates(batch_df)

        # Feature Extraction
        X_rows = []
        pair_meta = []  # (s1_id, cand_id)

        for s1_id, cands in cand_dict.items():
            s1_rec = s1_records[s1_id]
            for rank, cand_id in enumerate(cands):
                aux_rec = blocker.aux_records[cand_id]
                feats = extract_pair_features(s1_rec, aux_rec, rank=rank)
                X_rows.append(feats)
                pair_meta.append((s1_id, cand_id))

        # Model Scoring
        s1_cand_scores = {s1_id: [] for s1_id in batch_df["entity_id"]}
        if len(X_rows) > 0:
            X_batch = np.array(X_rows, dtype=np.float32)
            probs = model.predict_proba(X_batch)[:, 1]
            for idx, prob in enumerate(probs):
                s1_id, cand_id = pair_meta[idx]
                s1_cand_scores[s1_id].append((cand_id, float(prob)))

        # Precision-Heavy Singleton Gating & Target Uniqueness Optimization
        batch_pairs = []
        for s1_id, scores in s1_cand_scores.items():
            if scores:
                max_p = max(p for _, p in scores)
                # If top candidate is below singleton_cutoff, entity is protected as a singleton
                if max_p >= singleton_cutoff:
                    for cid, p in scores:
                        if p >= match_threshold:
                            batch_pairs.append((p, s1_id, cid))

        # Rank all pairs by confidence descending
        batch_pairs.sort(key=lambda x: x[0], reverse=True)
        batch_assigned = defaultdict(set)
        for p, s1_id, cid in batch_pairs:
            if cid not in country_assigned_aux:
                country_assigned_aux.add(cid)
                batch_assigned[s1_id].add(cid)

        # Output Generation
        for row in batch_df.itertuples(index=False):
            s1_id = row.entity_id
            all_cands = cand_dict.get(s1_id, set())
            final_matches = batch_assigned.get(s1_id, set())

            # Write candidate pairs
            cand_str = ",".join(sorted(all_cands)) if all_cands else ""
            candidate_f.write(f"{s1_id}\t{cand_str}\n")

            # Write matching results (empty string for singletons, sorted comma-separated for matches)
            match_str = ",".join(sorted(final_matches)) if final_matches else ""
            matching_f.write(f"{s1_id}\t{match_str}\n")

        batch_time = time.time() - tb0
        rate = len(batch_df) / batch_time
        remaining_records = n_total - b_end
        eta_min = (remaining_records / rate) / 60.0 if rate > 0 else 0.0

        print(
            f"  Batch {b_idx + 1}/{n_batches} ({b_end:,}/{n_total:,}) | "
            f"Pairs scored: {len(X_rows):,} | "
            f"Speed: {rate:.0f} entities/s | "
            f"ETA for {country_name}: {eta_min:.1f}m"
        )

    print(f"[Country Complete] {country_name} finished in {(time.time() - t_start) / 60:.2f} minutes.")


def run_full_inference(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_dir: str = "code/business_entity_resolution/src",
):
    print("=" * 70)
    print("      ML CHALLENGE 2026: HIGH-PERFORMANCE INFERENCE ENGINE")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # 1. Load Model & Config
    model_path = os.path.join(model_dir, "model.joblib")
    config_path = os.path.join(model_dir, "model_config.json")

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    # Enable all CPU threads in LightGBM for maximum AWS speed
    model.set_params(n_jobs=-1)

    with open(config_path) as f:
        config = json.load(f)

    match_thresh = config.get("match_threshold", 0.50)
    singleton_cut = config.get("singleton_cutoff", 0.40)
    print(f"Loaded config: Match Threshold = {match_thresh}, Singleton Cutoff = {singleton_cut}")

    # 2. Load test_source1.tsv
    source1_file = os.path.join(test_dir, "test_source1.tsv")
    print(f"Reading {source1_file}...")
    s1_full_df = pd.read_csv(source1_file, sep="\t")
    print(f"Total test Source 1 entities: {len(s1_full_df):,}")
    print(s1_full_df["country"].value_counts())

    # 3. Process country-by-country: France -> US -> India
    country_order = ["France", "US", "India"]
    t0 = time.time()

    with open(matching_path, "w", encoding="utf-8") as matching_f, \
         open(candidate_path, "w", encoding="utf-8") as candidate_f:

        matching_f.write("source1_entity_id\tmatched_entity_ids\n")
        candidate_f.write("source1_entity_id\tcandidate_entity_ids\n")

        for country in country_order:
            country_s1 = s1_full_df[s1_full_df["country"] == country].copy()
            if len(country_s1) == 0:
                continue

            print("\n" + "=" * 70)
            print(f"  PROCESSING COUNTRY: {country.upper()} ({len(country_s1):,} S1 entities)")
            print("=" * 70)

            aux_records = load_auxiliary_records_for_country(test_dir, country)

            run_country_inference(
                country_s1,
                aux_records,
                model,
                match_thresh,
                singleton_cut,
                matching_f,
                candidate_f,
                batch_size=25000,
            )

            del aux_records
            del country_s1

    t_total = time.time() - t0
    print("\n" + "=" * 70)
    print(f"INFERENCE COMPLETE in {t_total / 60:.2f} minutes!")
    print(f"Output 1: {matching_path}")
    print(f"Output 2: {candidate_path}")
    print("=" * 70)

    # Automatic Validation
    print("\nRunning official submission validator...")
    import subprocess
    cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir,
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    run_full_inference(args.test_dir, args.output_dir)
