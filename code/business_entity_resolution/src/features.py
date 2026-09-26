"""
ML Challenge 2026: Business Entity Resolution
Module: Ultra-High-Precision Pairwise Feature Engineering (V4 - 35 Features)
 
Key additions over V3:
- Jaro-Winkler similarity
- Character n-gram Jaccard (3-gram and 4-gram)
- Token precision/recall separately
- Prefix exact match signals
- Compact name ratio
- Rule-override composite signals
"""

from typing import Dict, List, Set, Tuple
import numpy as np
from rapidfuzz import fuzz, distance


FEATURE_NAMES = [
    # Name String Similarities
    "name_ratio",               # 0
    "name_partial_ratio",       # 1
    "name_token_sort_ratio",    # 2
    "name_token_set_ratio",     # 3
    "clean_name_ratio",         # 4
    "clean_token_sort_ratio",   # 5
    "jaro_winkler",             # 6
    "compact_ratio",            # 7
    # Exact / Hard Match Flags
    "exact_clean_match",        # 8
    "exact_compact_match",      # 9
    "skel_match",               # 10
    "first_word_match",         # 11
    "prefix3_match",            # 12
    "prefix5_match",            # 13
    # Token-Level Signals
    "token_jaccard",            # 14
    "token_precision",          # 15
    "token_recall",             # 16
    "shared_token_count",       # 17
    # n-gram Signals
    "trigram_jaccard",          # 18
    "fourgram_jaccard",         # 19
    # Name Length Signals
    "name_len_diff",            # 20
    "name_len_ratio",           # 21
    "compact_len_diff",         # 22
    # Address Signals
    "addr_ratio",               # 23
    "addr_partial_ratio",       # 24
    "addr_token_jaccard",       # 25
    "addr_numeric_overlap",     # 26
    "addr_numeric_exact",       # 27
    "addr_num_shared",          # 28
    "is_addr_missing",          # 29
    # Source / Rank Metadata
    "aux_is_s2",                # 30
    # Rule-Override / Composite Signals
    "rule_exact_override",      # 31
    "rule_high_conf",           # 32
    "rule_addr_conflict",       # 33
    "name_starts_with_s1",      # 34
    "s1_starts_with_aux",       # 35
    "skel_ratio",               # 36
    "s1_is_addr_missing",       # 37
    "is_hard_rule_match",       # 38
    "is_hard_rule_conflict",    # 39
]


def _ngram_jaccard(a: str, b: str, n: int) -> float:
    """Character n-gram Jaccard similarity between two strings."""
    if not a or not b:
        return 0.0
    a_s = a.replace(" ", "")
    b_s = b.replace(" ", "")
    if len(a_s) < n or len(b_s) < n:
        return fuzz.ratio(a_s, b_s) / 100.0
    a_grams = {a_s[i:i+n] for i in range(len(a_s) - n + 1)}
    b_grams = {b_s[i:i+n] for i in range(len(b_s) - n + 1)}
    inter = len(a_grams & b_grams)
    union = len(a_grams | b_grams)
    return inter / union if union > 0 else 0.0


def extract_pair_features(
    s1_record: Dict, aux_record: Dict, rank: int = 0
) -> List[float]:
    """
    Extracts 35-dimensional numerical feature vector for one (S1, Auxiliary) candidate pair.
    """
    s1_name = s1_record["name"]
    aux_name = aux_record["name"]
    s1_clean = s1_record["clean_name"]
    aux_clean = aux_record["clean_name"]
    s1_compact = s1_record["compact_name"]
    aux_compact = aux_record["compact_name"]
    s1_skel = s1_record["skel"]
    aux_skel = aux_record["skel"]
    s1_first = s1_record["first_word"]
    aux_first = aux_record["first_word"]

    # 0-7: Name String Similarities
    name_ratio = fuzz.ratio(s1_name, aux_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, aux_name) / 100.0
    name_tok_sort = fuzz.token_sort_ratio(s1_name, aux_name) / 100.0
    name_tok_set = fuzz.token_set_ratio(s1_name, aux_name) / 100.0
    clean_ratio = fuzz.ratio(s1_clean, aux_clean) / 100.0
    clean_tok_sort = fuzz.token_sort_ratio(s1_clean, aux_clean) / 100.0
    jaro_winkler = distance.JaroWinkler.normalized_similarity(s1_clean, aux_clean)
    compact_ratio = fuzz.ratio(s1_compact, aux_compact) / 100.0

    # 8-13: Exact / Hard Match Flags
    exact_clean = 1.0 if (s1_clean and s1_clean == aux_clean) else 0.0
    exact_compact = 1.0 if (s1_compact and s1_compact == aux_compact) else 0.0
    skel_match = 1.0 if (s1_skel and s1_skel == aux_skel) else 0.0
    skel_ratio = (fuzz.ratio(s1_skel, aux_skel) / 100.0) if (s1_skel and aux_skel) else 0.0
    first_match = 1.0 if (s1_first and s1_first == aux_first) else 0.0
    prefix3 = 1.0 if (len(s1_clean) >= 3 and len(aux_clean) >= 3 and s1_clean[:3] == aux_clean[:3]) else 0.0
    prefix5 = 1.0 if (len(s1_clean) >= 5 and len(aux_clean) >= 5 and s1_clean[:5] == aux_clean[:5]) else 0.0
    
    # 14-17: Token-Level Signals
    s1_toks = set(s1_clean.split())
    aux_toks = set(aux_clean.split())
    tp_toks = len(s1_toks & aux_toks)
    union_toks = len(s1_toks | aux_toks)
    token_jaccard = (tp_toks / union_toks) if union_toks > 0 else 0.0
    token_precision = (tp_toks / len(aux_toks)) if aux_toks else 0.0
    token_recall = (tp_toks / len(s1_toks)) if s1_toks else 0.0
    shared_token_count = float(tp_toks)

    # 18-19: n-gram Signals
    trigram_jac = _ngram_jaccard(s1_clean, aux_clean, 3)
    fourgram_jac = _ngram_jaccard(s1_clean, aux_clean, 4)

    # 20-22: Name Length Signals
    l1, l2 = len(s1_clean), len(aux_clean)
    len_diff = float(abs(l1 - l2))
    len_ratio = float(min(l1, l2) / max(l1, l2, 1))
    compact_len_diff = float(abs(len(s1_compact) - len(aux_compact)))

    # 23-29: Address Signals
    s1_addr = s1_record.get("business_address", "")
    aux_addr = aux_record.get("business_address", "")
    addr_ratio = (fuzz.ratio(s1_addr, aux_addr) / 100.0) if s1_addr and aux_addr else 0.0
    addr_partial_ratio = (fuzz.partial_ratio(s1_addr, aux_addr) / 100.0) if s1_addr and aux_addr else 0.0

    s1_addr_toks = s1_record.get("addr_tokens", set())
    # aux_record stores its name tokens under "tokens" key, address tokens under "addr_tokens"
    aux_addr_toks = aux_record.get("addr_tokens", set())
    addr_inter = len(s1_addr_toks & aux_addr_toks)
    addr_union = len(s1_addr_toks | aux_addr_toks)
    addr_tok_jaccard = (addr_inter / addr_union) if addr_union > 0 else 0.0

    s1_nums = s1_record["addr_nums"]
    aux_nums = aux_record["addr_nums"]
    num_inter = len(s1_nums & aux_nums)
    num_union = len(s1_nums | aux_nums)
    addr_num_overlap = (num_inter / num_union) if num_union > 0 else 0.0
    addr_num_exact = 1.0 if (s1_nums and s1_nums == aux_nums) else 0.0
    addr_num_shared = float(num_inter)
    is_addr_missing = aux_record.get("is_addr_missing", 0.0)
    s1_is_addr_missing = 1.0 if not s1_record.get("has_address") else 0.0

    # 28: Source Metadata
    aux_is_s2 = 1.0 if aux_record["id"].startswith("S2-") else 0.0

    # 30-34: Rule-Override / Composite Signals
    rule_exact = 1.0 if (exact_clean == 1.0 and skel_match == 1.0) else 0.0
    
    is_hard_rule_match = 1.0 if (exact_clean == 1.0 and exact_compact == 1.0 and addr_num_overlap > 0) else 0.0
    is_hard_rule_conflict = 1.0 if (exact_clean == 1.0 and num_inter == 0 and num_union > 0) else 0.0

    strong_signals = sum([
        1 if exact_clean == 1.0 else 0,
        1 if exact_compact == 1.0 else 0,
        1 if skel_match == 1.0 else 0,
        1 if clean_ratio >= 0.95 else 0,
        1 if jaro_winkler >= 0.97 else 0,
        1 if compact_ratio >= 0.95 else 0,
    ])
    rule_high_conf = float(strong_signals) / 6.0

    if s1_nums and aux_nums and not (s1_nums & aux_nums) and (clean_ratio >= 0.85 or exact_clean == 1.0):
        rule_addr_conflict = 1.0
    else:
        rule_addr_conflict = 0.0

    name_starts = 1.0 if (s1_clean and aux_clean and aux_clean.startswith(s1_clean)) else 0.0
    s1_starts = 1.0 if (s1_clean and aux_clean and s1_clean.startswith(aux_clean)) else 0.0

    return [
        name_ratio,         # 0
        name_partial,       # 1
        name_tok_sort,      # 2
        name_tok_set,       # 3
        clean_ratio,        # 4
        clean_tok_sort,     # 5
        jaro_winkler,       # 6
        compact_ratio,      # 7
        exact_clean,        # 8
        exact_compact,      # 9
        skel_match,         # 10
        first_match,        # 11
        prefix3,            # 12
        prefix5,            # 13
        token_jaccard,      # 14
        token_precision,    # 15
        token_recall,       # 16
        shared_token_count, # 17
        trigram_jac,        # 18
        fourgram_jac,       # 19
        len_diff,           # 20
        len_ratio,          # 21
        compact_len_diff,   # 22
        addr_ratio,         # 23
        addr_partial_ratio, # 24
        addr_tok_jaccard,   # 25
        addr_num_overlap,   # 26
        addr_num_exact,     # 27
        addr_num_shared,    # 28
        is_addr_missing,    # 29
        aux_is_s2,          # 30
        rule_exact,         # 31
        rule_high_conf,     # 32
        rule_addr_conflict, # 33
        name_starts,        # 34
        s1_starts,          # 35
        skel_ratio,         # 36
        s1_is_addr_missing, # 37
        is_hard_rule_match, # 38
        is_hard_rule_conflict, # 39
    ]
