"""
ML Challenge 2026: Business Entity Resolution
Module: End-to-End Training, Scoring, and Macro-F0.5 Threshold Optimization
"""

import os
import sys
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from code.business_entity_resolution.src.preprocess import (
    clean_business_name,
    extract_address_tokens,
    phonetic_skeleton,
    get_character_ngrams,
)
from code.business_entity_resolution.src.blocking import MultiIndexBlocker
from code.business_entity_resolution.src.features import extract_pair_features, FEATURE_NAMES
from code.business_entity_resolution.src.evaluation import compute_macro_f05


def train_and_evaluate_pipeline(val_dir: str = "validation"):
    print("=" * 60)
    print("STARTING END-TO-END PIPELINE: TRAINING & F_0.5 OPTIMIZATION")
    print("=" * 60)

    # 1. Load Data
    s1_df = pd.read_csv(os.path.join(val_dir, "val_source1.tsv"), sep="\t")
    s2_df = pd.read_csv(os.path.join(val_dir, "val_source2.tsv"), sep="\t")
    s3_df = pd.read_csv(os.path.join(val_dir, "val_source3.tsv"), sep="\t")
    gt_df = pd.read_csv(os.path.join(val_dir, "val_ground_truth.tsv"), sep="\t")

    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")
    gt_dict = {
        row["source1_entity_id"]: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"].strip() else set()
        for _, row in gt_df.iterrows()
    }
    all_s1_ids = list(gt_df["source1_entity_id"])

    # 2. Build Preprocessed S1 Dictionary
    print("Preprocessing Source 1 records...")
    s1_records = {}
    for _, row in s1_df.iterrows():
        sid = row["entity_id"]
        name = str(row["business_name"])
        addr = str(row["business_address"]) if pd.notna(row["business_address"]) else ""
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
            "country": str(row["country"]).strip(),
        }

    # 3. Index Auxiliary Records & Generate Candidates
    aux_list = s2_df.to_dict("records") + s3_df.to_dict("records")
    blocker = MultiIndexBlocker(max_candidates_per_entity=35)
    blocker.build_auxiliary_index(aux_list)

    print("Generating candidate sets...")
    cand_dict = blocker.batch_generate_candidates(s1_df)

    # 4. Construct Feature Matrix (X, y)
    print("Extracting pairwise features for training...")
    X_rows = []
    y_rows = []
    pair_meta = []  # (s1_id, aux_id)

    # Process pairs
    for s1_id, cands in cand_dict.items():
        s1_rec = s1_records[s1_id]
        true_set = gt_dict.get(s1_id, set())

        for rank, cand_id in enumerate(cands):
            aux_rec = blocker.aux_records[cand_id]
            feats = extract_pair_features(s1_rec, aux_rec, rank=rank)
            label = 1 if cand_id in true_set else 0

            X_rows.append(feats)
            y_rows.append(label)
            pair_meta.append((s1_id, cand_id))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    print(f"Dataset constructed: {X.shape[0]} candidate pairs. Positives: {np.sum(y)} ({np.mean(y):.2%}).")

    # 5. Split Train / Validation (Entity-Aware Split)
    unique_s1 = list(cand_dict.keys())
    train_s1, test_s1 = train_test_split(unique_s1, test_size=0.35, random_state=42)
    train_s1_set, test_s1_set = set(train_s1), set(test_s1)

    train_indices = [i for i, (s1, _) in enumerate(pair_meta) if s1 in train_s1_set]
    test_indices = [i for i, (s1, _) in enumerate(pair_meta) if s1 in test_s1_set]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    print(f"Train pairs: {len(X_train)}, Holdout validation pairs: {len(X_test)}")

    # 6. Train LightGBM Classifier
    print("Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.06,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Feature Importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTOP FEATURE IMPORTANCES:")
    for idx in sorted_idx[:10]:
        print(f"  {FEATURE_NAMES[idx]}: {importances[idx]}")

    # 7. Predict Holdout Scores
    test_probs = model.predict_proba(X_test)[:, 1]

    # Group predictions by S1 entity
    holdout_predictions_by_s1 = defaultdict(list)
    for idx, prob in enumerate(test_probs):
        orig_idx = test_indices[idx]
        s1_id, cand_id = pair_meta[orig_idx]
        holdout_predictions_by_s1[s1_id].append((cand_id, float(prob)))

    # Ensure all test S1 entities are present (even those with 0 candidates)
    for s1_id in test_s1:
        if s1_id not in holdout_predictions_by_s1:
            holdout_predictions_by_s1[s1_id] = []

    # 8. Optimize Threshold theta and Singleton Cutoff on Macro F_0.5
    print("\n" + "=" * 60)
    print("OPTIMIZING DECISION THRESHOLDS FOR MACRO F_0.5")
    print("=" * 60)

    best_score = -1.0
    best_thresh = 0.5
    best_single_cut = 0.5
    best_results = None

    threshold_grid = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    single_cut_grid = [0.40, 0.50, 0.60, 0.70]

    for thresh in threshold_grid:
        for single_cut in single_cut_grid:
            pred_dict = {}
            for s1_id in test_s1:
                cand_probs = holdout_predictions_by_s1[s1_id]
                if not cand_probs:
                    pred_dict[s1_id] = set()
                    continue

                max_p = max(p for _, p in cand_probs)
                if max_p < single_cut:
                    # Singleton Gate: predict empty
                    pred_dict[s1_id] = set()
                else:
                    # Accept candidates meeting match threshold
                    matches = {cid for cid, p in cand_probs if p >= thresh}
                    pred_dict[s1_id] = matches

            # Evaluate on exact official competition metric
            metrics = compute_macro_f05(gt_dict, pred_dict, test_s1)
            score = metrics["macro_f05"]

            if score > best_score:
                best_score = score
                best_thresh = thresh
                best_single_cut = single_cut
                best_results = metrics

            print(
                f"Threshold: {thresh:.2f}, Singleton Cut: {single_cut:.2f} -> "
                f"Macro F_0.5: {score:.5f} (P: {metrics['macro_precision']:.4f}, R: {metrics['macro_recall']:.4f}, SingleAcc: {metrics['singleton_accuracy']:.4f})"
            )

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS:")
    print(f"  BEST MACRO F_0.5 SCORE:    {best_score:.5f}")
    print(f"  Optimal Match Threshold:   {best_thresh:.2f}")
    print(f"  Optimal Singleton Cutoff:  {best_single_cut:.2f}")
    print(f"  Macro Precision:           {best_results['macro_precision']:.5f}")
    print(f"  Macro Recall:              {best_results['macro_recall']:.5f}")
    print(f"  Singleton Accuracy:        {best_results['singleton_accuracy']:.5f}")
    print("=" * 60)

    # 9. Retrain on all available pairs and save model artifact
    print("Saving production model and configuration...")
    import joblib
    import json

    model_dir = "code/business_entity_resolution/src"
    model_path = os.path.join(model_dir, "model.joblib")
    config_path = os.path.join(model_dir, "model_config.json")

    # Fit on all validation pairs
    final_model = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.06,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y)
    joblib.dump(final_model, model_path)

    config = {
        "match_threshold": best_thresh,
        "singleton_cutoff": best_single_cut,
        "macro_f05": best_score,
        "macro_precision": best_results["macro_precision"],
        "macro_recall": best_results["macro_recall"],
        "features": FEATURE_NAMES,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Config saved to {config_path}")

    return final_model, best_thresh, best_single_cut


if __name__ == "__main__":
    train_and_evaluate_pipeline()
