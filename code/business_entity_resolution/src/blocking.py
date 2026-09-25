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
import time
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
    def __init__(self, max_candidates_per_entity: int = 40, max_posting: int = 250):
        self.max_candidates = max_candidates_per_entity
        self.max_posting = max_posting
        self.exact_name_index = defaultdict(list)
        self.compact_name_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.first_word_index = defaultdict(list)
        self.phonetic_index = defaultdict(list)
        self.aux_records = {}
        self.country_aux_ids = defaultdict(list)

    def build_auxiliary_index(self, aux_records: List[Dict]):
        """
        Indexes all auxiliary (S2 and S3) records into lightweight multi-channel hash tables.
        Memory-efficient: avoids heavy n-gram sets and caps high-frequency posting lists.
        """
        total = len(aux_records)
        print(f"Building optimized multi-channel inverted index over {total} auxiliary records...")
        t0 = time.time()
        
        for idx, r in enumerate(aux_records):
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

            # Store only fields strictly required by feature extraction
            self.aux_records[eid] = {
                "id": eid,
                "country": country,
                "name": name,
                "clean_name": clean_name,
                "compact_name": compact_name,
                "first_word": first_word,
                "skel": skel,
                "tokens": tokens,
                "addr_tokens": set(addr_words),
                "addr_nums": addr_nums,
                "has_address": bool(addr),
                "is_addr_missing": 1.0 if not addr else 0.0,
            }
            self.country_aux_ids[country].append(eid)

            # Channel 1: Exact Clean Name
            if clean_name and len(self.exact_name_index[(country, clean_name)]) < self.max_posting:
                self.exact_name_index[(country, clean_name)].append(eid)

            # Channel 1B: Compact Name
            if len(compact_name) >= 4 and len(self.compact_name_index[(country, compact_name)]) < self.max_posting:
                self.compact_name_index[(country, compact_name)].append(eid)

            # Channel 2: Distinctive Token Index (capped to discard stop-words)
            for tok in tokens:
                if len(tok) >= 4 and tok not in COMMON_STOPWORDS and len(self.token_index[(country, tok)]) < self.max_posting:
                    self.token_index[(country, tok)].append(eid)

            # Channel 3: First Word Brand Index
            if len(first_word) >= 4 and first_word not in COMMON_STOPWORDS and len(self.first_word_index[(country, first_word)]) < self.max_posting:
                self.first_word_index[(country, first_word)].append(eid)

            # Channel 4: Phonetic Skeleton Index
            if len(skel) >= 3 and len(self.phonetic_index[(country, skel)]) < self.max_posting:
                self.phonetic_index[(country, skel)].append(eid)

            if (idx + 1) % 200000 == 0 or (idx + 1) == total:
                elapsed = time.time() - t0
                pct = ((idx + 1) / total) * 100.0
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                print(f"  [Index Progress] {idx + 1:,} / {total:,} records ({pct:.1f}%) in {elapsed:.1f}s ({rate:.0f} rec/s)...")

        print(f"Multi-channel inverted index successfully constructed in {time.time() - t0:.1f}s.")

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
                elif len(s1_tokens & cand_info["tokens"]) >= 1:
                    candidates.add(cand_id)

        # Channel 3: First Word Brand Anchor
        if len(s1_first_word) >= 4 and s1_first_word not in COMMON_STOPWORDS:
            for match_id in self.first_word_index.get((s1_country, s1_first_word), []):
                candidates.add(match_id)

        # Channel 4: Phonetic Skeleton Hits
        if len(s1_skel) >= 3:
            for match_id in self.phonetic_index.get((s1_country, s1_skel), []):
                cand_info = self.aux_records[match_id]
                if not s1_nums or not cand_info["addr_nums"] or (s1_nums & cand_info["addr_nums"]):
                    candidates.add(match_id)

        # Fast candidate ranking and pruning to max_candidates
        cand_list = list(candidates)
        if len(cand_list) > self.max_candidates:
            def cand_rank_key(cid):
                c_info = self.aux_records[cid]
                tok_overlap = len(s1_tokens & c_info["tokens"])
                num_overlap = len(s1_nums & c_info["addr_nums"])
                exact_flag = 10 if c_info["clean_name"] == clean_name else 0
                compact_flag = 5 if c_info["compact_name"] == compact_name else 0
                skel_flag = 3 if c_info["skel"] == s1_skel else 0
                return (exact_flag + compact_flag + skel_flag + num_overlap * 2 + tok_overlap)

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
