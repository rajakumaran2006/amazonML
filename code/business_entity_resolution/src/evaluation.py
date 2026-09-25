"""
ML Challenge 2026: Business Entity Resolution
Evaluation Module: Faithful Official Macro-Averaged F_0.5 Evaluator

Evaluation Logic:
- Metric: F_beta with beta = 0.5 (Precision-heavy: precision weighted 2x over recall)
  Formula: F_0.5 = (1.25 * P * R) / (0.25 * P + R)
- Macro-Averaged across ALL Source 1 entities in the evaluation set.
- Singletons (entities with 0 true matches):
  * Correctly predicting empty list -> F_0.5 = 1.0
  * Predicting any match -> F_0.5 = 0.0
"""

from typing import Dict, List, Optional, Set, Tuple


def compute_entity_f05(y_true_set: Set[str], y_pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (F_0.5, precision, recall) for a single Source 1 entity.
    """
    # Case 1: Ground truth is singleton (no matches)
    if len(y_true_set) == 0:
        if len(y_pred_set) == 0:
            return 1.0, 1.0, 1.0  # Correct singleton
        else:
            return 0.0, 0.0, 0.0  # False merge on singleton

    # Case 2: Ground truth has matches, but prediction is empty
    if len(y_pred_set) == 0:
        return 0.0, 0.0, 0.0

    # Case 3: Both have entries
    tp = len(y_true_set & y_pred_set)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len(y_pred_set)
    recall = tp / len(y_true_set)

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return float(f05), float(precision), float(recall)


def compute_macro_f05(
    ground_truth_dict: Dict[str, Set[str]],
    predictions_dict: Dict[str, Set[str]],
    all_s1_ids: List[str],
    candidate_dict: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, float]:
    """
    Computes official macro-averaged F_0.5 across all Source 1 entities.
    
    Args:
        ground_truth_dict: {s1_id: set(target_s2_s3_ids)}
        predictions_dict:  {s1_id: set(target_s2_s3_ids)}
        all_s1_ids:        list of all S1 IDs in evaluation set
        candidate_dict:    optional {s1_id: set(candidate_s2_s3_ids)} to measure candidate recall ceiling
        
    Returns:
        dict with macro_f05, macro_precision, macro_recall, singleton_accuracy, candidate_recall, etc.
    """
    n_total = len(all_s1_ids)
    if n_total == 0:
        raise ValueError("Evaluation set cannot be empty.")

    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0

    singleton_count = 0
    singleton_correct = 0

    non_singleton_count = 0
    non_singleton_f05 = 0.0

    # Candidate recall tracking (if candidate set provided)
    total_true_matches = 0
    captured_candidate_matches = 0

    empty_preds_count = 0

    for s1_id in all_s1_ids:
        y_true = ground_truth_dict.get(s1_id, set())
        y_pred = predictions_dict.get(s1_id, set())

        if len(y_pred) == 0:
            empty_preds_count += 1

        f05, prec, rec = compute_entity_f05(y_true, y_pred)
        total_f05 += f05
        total_prec += prec
        total_rec += rec

        if len(y_true) == 0:
            singleton_count += 1
            if len(y_pred) == 0:
                singleton_correct += 1
        else:
            non_singleton_count += 1
            non_singleton_f05 += f05

        if candidate_dict is not None and len(y_true) > 0:
            cands = candidate_dict.get(s1_id, set())
            total_true_matches += len(y_true)
            captured_candidate_matches += len(y_true & cands)

    macro_f05 = total_f05 / n_total
    macro_prec = total_prec / n_total
    macro_rec = total_rec / n_total
    singleton_acc = (singleton_correct / singleton_count) if singleton_count > 0 else 1.0
    non_singleton_macro = (non_singleton_f05 / non_singleton_count) if non_singleton_count > 0 else 0.0

    cand_recall = (
        (captured_candidate_matches / total_true_matches)
        if (candidate_dict is not None and total_true_matches > 0)
        else None
    )

    metrics = {
        "macro_f05": round(macro_f05, 5),
        "macro_precision": round(macro_prec, 5),
        "macro_recall": round(macro_rec, 5),
        "singleton_accuracy": round(singleton_acc, 5),
        "non_singleton_macro_f05": round(non_singleton_macro, 5),
        "num_total_s1": n_total,
        "num_singletons": singleton_count,
        "num_non_singletons": non_singleton_count,
        "empty_predictions_rate": round(empty_preds_count / n_total, 5),
    }
    if cand_recall is not None:
        metrics["candidate_recall"] = round(cand_recall, 5)

    return metrics
