"""
ML Challenge 2026: Business Entity Resolution
Module: Fast Vectorized Pairwise Feature Engineering using RapidFuzz
"""

from typing import Dict, List, Tuple
import numpy as np
from rapidfuzz import fuzz

FEATURE_NAMES = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "clean_name_ratio",
    "clean_token_sort_ratio",
    "exact_clean_match",
    "exact_compact_match",
    "skel_match",
    "first_word_match",
    "name_len_diff",
    "name_len_ratio",
    "addr_token_jaccard",
    "addr_numeric_overlap",
    "addr_numeric_exact",
    "is_addr_missing",
    "aux_is_s2",
    "blocking_rank",
]


def extract_pair_features(
    s1_record: Dict, aux_record: Dict, rank: int = 0
) -> List[float]:
    """
    Extracts numerical feature vector for one (S1, Auxiliary) candidate pair.
    """
    s1_clean = s1_record["clean_name"]
    aux_clean = aux_record["clean_name"]
    s1_compact = s1_record["compact_name"]
    aux_compact = aux_record["compact_name"]
    s1_skel = s1_record["skel"]
    aux_skel = aux_record["skel"]
    s1_first = s1_record["first_word"]
    aux_first = aux_record["first_word"]

    # 1. Name similarities
    name_ratio = fuzz.ratio(s1_record["name"], aux_record["name"]) / 100.0
    name_partial = fuzz.partial_ratio(s1_record["name"], aux_record["name"]) / 100.0
    name_tok_sort = fuzz.token_sort_ratio(s1_record["name"], aux_record["name"]) / 100.0
    name_tok_set = fuzz.token_set_ratio(s1_record["name"], aux_record["name"]) / 100.0

    clean_ratio = fuzz.ratio(s1_clean, aux_clean) / 100.0
    clean_tok_sort = fuzz.token_sort_ratio(s1_clean, aux_clean) / 100.0

    exact_clean = 1.0 if (s1_clean and s1_clean == aux_clean) else 0.0
    exact_compact = 1.0 if (s1_compact and s1_compact == aux_compact) else 0.0
    skel_match = 1.0 if (s1_skel and s1_skel == aux_skel) else 0.0
    first_match = 1.0 if (s1_first and s1_first == aux_first) else 0.0

    l1, l2 = len(s1_clean), len(aux_clean)
    len_diff = float(abs(l1 - l2))
    len_ratio = float(min(l1, l2) / max(l1, l2, 1))

    # 2. Address similarities
    s1_tokens = s1_record.get("addr_tokens", set())
    aux_tokens = aux_record.get("tokens", set())
    tok_intersect = len(s1_tokens & aux_tokens)
    tok_union = len(s1_tokens | aux_tokens)
    addr_tok_jaccard = (tok_intersect / tok_union) if tok_union > 0 else 0.0

    s1_nums = s1_record["addr_nums"]
    aux_nums = aux_record["addr_nums"]
    num_intersect = len(s1_nums & aux_nums)
    num_union = len(s1_nums | aux_nums)
    addr_num_overlap = (num_intersect / num_union) if num_union > 0 else 0.0
    addr_num_exact = 1.0 if (s1_nums and s1_nums == aux_nums) else 0.0
    is_addr_missing = 1.0 if not aux_record.get("has_address", True) else 0.0

    # 3. Metadata & ranking
    aux_is_s2 = 1.0 if aux_record["id"].startswith("S2-") else 0.0
    blocking_rank = float(rank)

    return [
        name_ratio,
        name_partial,
        name_tok_sort,
        name_tok_set,
        clean_ratio,
        clean_tok_sort,
        exact_clean,
        exact_compact,
        skel_match,
        first_match,
        len_diff,
        len_ratio,
        addr_tok_jaccard,
        addr_num_overlap,
        addr_num_exact,
        is_addr_missing,
        aux_is_s2,
        blocking_rank,
    ]
