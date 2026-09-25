"""
ML Challenge 2026: Business Entity Resolution
Module: High-Recall Multi-Index Blocking Engine (V3 - Phonetic & Multilingual)

Generates candidate pairs for each Source 1 entity across auxiliary sources (S2, S3).
Enforces:
1. Strict Country Partitioning (US, India, France)
2. Channel 1: Exact Clean Name & Compact (No-Space) Name
3. Channel 2: Distinctive Token Inverted Index
4. Channel 3: First-Word / Core Brand Anchor Index
5. Channel 4: Cross-Script Phonetic Skeleton Index
"""

import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

from code.business_entity_resolution.src.preprocess import (
    clean_business_name,
    normalize_text,
    extract_address_tokens,
    get_character_ngrams,
    phonetic_skeleton,
)

COMMON_STOPWORDS = {
    "and", "the", "of", "in", "for", "on", "at", "to", "a", "an",
    "de", "du", "des", "le", "la", "les", "et", "en", "sur",
    "services", "solutions", "enterprises", "trading", "group", "holdings",
    "international", "consultants", "india", "usa", "america", "france",
}


class MultiIndexBlocker:
    def __init__(self, max_candidates_per_entity: int = 40):
        self.max_candidates = max_candidates_per_entity
        self.exact_name_index = defaultdict(list)
        self.compact_name_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.first_word_index = defaultdict(list)
        self.phonetic_index = defaultdict(list)
        self.aux_records = {}
        self.country_aux_ids = defaultdict(list)

    def build_auxiliary_index(self, aux_records: List[Dict]):
        """
        Indexes all auxiliary (S2 and S3) records into multi-channel hash tables.
        """
        print(f"Building multi-channel inverted index over {len(aux_records)} auxiliary records...")
        for r in aux_records:
            eid = r["entity_id"]
            country = str(r.get("country", "")).strip()
            name = str(r.get("business_name", ""))
            addr = str(r.get("business_address", "")) if pd.notna(r.get("business_address")) else ""

            clean_name = clean_business_name(name)
            compact_name = clean_name.replace(" ", "")
            addr_words, addr_nums = extract_address_tokens(addr)
            tokens = set(clean_name.split())
            first_word = clean_name.split()[0] if clean_name.split() else ""
            skel = phonetic_skeleton(clean_name)
            first_skel = phonetic_skeleton(first_word) if first_word else ""

            record_info = {
                "id": eid,
                "country": country,
                "name": name,
                "clean_name": clean_name,
                "compact_name": compact_name,
                "addr_nums": addr_nums,
                "tokens": tokens,
                "first_word": first_word,
                "skel": skel,
                "first_skel": first_skel,
                "ngrams": get_character_ngrams(clean_name, n=3),
            }
            self.aux_records[eid] = record_info
            self.country_aux_ids[country].append(eid)

            # Channel 1: Exact Clean Name
            if clean_name:
                self.exact_name_index[(country, clean_name)].append(eid)

            # Channel 1B: Compact Name
            if len(compact_name) >= 4:
                self.compact_name_index[(country, compact_name)].append(eid)

            # Channel 2: Distinctive Token Index
            for tok in tokens:
                if len(tok) >= 4 and tok not in COMMON_STOPWORDS:
                    self.token_index[(country, tok)].append(eid)

            # Channel 3: First Word Brand Index
            if len(first_word) >= 4 and first_word not in COMMON_STOPWORDS:
                self.first_word_index[(country, first_word)].append(eid)

            # Channel 4: Phonetic Skeleton Index
            if len(skel) >= 3:
                self.phonetic_index[(country, skel)].append(eid)
            if len(first_skel) >= 3 and first_skel != skel:
                self.phonetic_index[(country, first_skel)].append(eid)

        print("Multi-channel inverted index successfully constructed.")

    def generate_candidates_for_s1(
        self, s1_id: str, s1_name: str, s1_addr: str, s1_country: str
    ) -> List[str]:
        """
        Retrieves candidates for a single Source 1 entity using union of channels.
        """
        clean_name = clean_business_name(s1_name)
        compact_name = clean_name.replace(" ", "")
        s1_words, s1_nums = extract_address_tokens(s1_addr)
        s1_tokens = set(clean_name.split())
        s1_first_word = clean_name.split()[0] if clean_name.split() else ""
        s1_skel = phonetic_skeleton(clean_name)
        s1_first_skel = phonetic_skeleton(s1_first_word) if s1_first_word else ""
        s1_ngrams = get_character_ngrams(clean_name, n=3)

        candidates = set()

        # Channel 1: Exact Clean Name
        if clean_name:
            for match_id in self.exact_name_index.get((s1_country, clean_name), []):
                candidates.add(match_id)

        # Channel 1B: Compact Name
        if len(compact_name) >= 4:
            for match_id in self.compact_name_index.get((s1_country, compact_name), []):
                candidates.add(match_id)

        # Channel 2: Distinctive Token Matches
        meaningful_tokens = [t for t in s1_tokens if len(t) >= 4 and t not in COMMON_STOPWORDS]
        token_candidate_counts = defaultdict(int)

        for tok in meaningful_tokens:
            for match_id in self.token_index.get((s1_country, tok), []):
                token_candidate_counts[match_id] += 1

        for cand_id, count in token_candidate_counts.items():
            if count >= 2:
                candidates.add(cand_id)
            elif count >= 1:
                cand_info = self.aux_records[cand_id]
                if len(meaningful_tokens) <= 2:
                    candidates.add(cand_id)
                elif s1_nums and (s1_nums & cand_info["addr_nums"]):
                    candidates.add(cand_id)
                else:
                    intersection_len = len(s1_ngrams & cand_info["ngrams"])
                    union_len = len(s1_ngrams | cand_info["ngrams"])
                    if union_len > 0 and (intersection_len / union_len) >= 0.20:
                        candidates.add(cand_id)

        # Channel 3: First Word Brand Anchor
        if len(s1_first_word) >= 4 and s1_first_word not in COMMON_STOPWORDS:
            for match_id in self.first_word_index.get((s1_country, s1_first_word), []):
                cand_info = self.aux_records[match_id]
                if s1_nums and (s1_nums & cand_info["addr_nums"]):
                    candidates.add(match_id)
                else:
                    intersection_len = len(s1_ngrams & cand_info["ngrams"])
                    union_len = len(s1_ngrams | cand_info["ngrams"])
                    if union_len > 0 and (intersection_len / union_len) >= 0.20:
                        candidates.add(match_id)

        # Channel 4: Phonetic Skeleton Hits
        for sk in [s1_skel, s1_first_skel]:
            if len(sk) >= 3:
                for match_id in self.phonetic_index.get((s1_country, sk), []):
                    cand_info = self.aux_records[match_id]
                    if s1_nums and (s1_nums & cand_info["addr_nums"]):
                        candidates.add(match_id)
                    else:
                        intersection_len = len(s1_ngrams & cand_info["ngrams"])
                        union_len = len(s1_ngrams | cand_info["ngrams"])
                        if union_len > 0 and (intersection_len / union_len) >= 0.20:
                            candidates.add(match_id)

        # Cap candidates to max_candidates using ranking
        cand_list = list(candidates)
        if len(cand_list) > self.max_candidates:
            def cand_rank_key(cid):
                c_info = self.aux_records[cid]
                tok_overlap = len(s1_tokens & c_info["tokens"])
                num_overlap = len(s1_nums & c_info["addr_nums"])
                exact_flag = 1 if c_info["clean_name"] == clean_name else 0
                compact_flag = 1 if c_info["compact_name"] == compact_name else 0
                skel_flag = 1 if c_info["skel"] == s1_skel else 0
                ngram_overlap = len(s1_ngrams & c_info["ngrams"])
                return (exact_flag, compact_flag, skel_flag, num_overlap, tok_overlap, ngram_overlap)

            cand_list.sort(key=cand_rank_key, reverse=True)
            cand_list = cand_list[: self.max_candidates]

        return cand_list

    def batch_generate_candidates(
        self, s1_df: pd.DataFrame
    ) -> Dict[str, Set[str]]:
        """
        Generates candidate sets for all S1 entities in dataframe.
        """
        candidate_dict = {}
        for _, row in s1_df.iterrows():
            s1_id = row["entity_id"]
            name = str(row["business_name"])
            addr = str(row["business_address"]) if pd.notna(row["business_address"]) else ""
            country = str(row["country"]).strip()

            cands = self.generate_candidates_for_s1(s1_id, name, addr, country)
            candidate_dict[s1_id] = set(cands)

        return candidate_dict
