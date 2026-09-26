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
        # NEW high-recall channels
        self.prefix4_index = defaultdict(list)      # 4-char prefix of compact name
        self.addr_num_index = defaultdict(list)     # address number tokens
        self.second_word_index = defaultdict(list)  # second brand word
        self.trigram_index = defaultdict(list)      # 3-char sliding window
        self.addr_word_index = defaultdict(list)    # address words
        self.addr_num_word_index = defaultdict(list)
        self.addr_word_pair_index = defaultdict(list)
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

            # Channel 1C: Prefix-4 of compact name (catches truncation/abbrev variants)
            prefix4 = compact_name[:4] if len(compact_name) >= 4 else ""
            if prefix4 and len(self.prefix4_index[(country, prefix4)]) < self.max_posting:
                self.prefix4_index[(country, prefix4)].append(eid)

            # Channel 2: Distinctive Token Index
            for tok in tokens:
                if len(tok) >= 4 and tok not in COMMON_STOPWORDS and len(self.token_index[(country, tok)]) < self.max_posting:
                    self.token_index[(country, tok)].append(eid)

            # Channel 3: First Word Brand Index
            if len(first_word) >= 4 and first_word not in COMMON_STOPWORDS and len(self.first_word_index[(country, first_word)]) < self.max_posting:
                self.first_word_index[(country, first_word)].append(eid)

            # Channel 3B: Second Word Brand Index (for "XYZ Corp" -> index "corp" skipped, real second word)
            tok_list = [t for t in clean_name.split() if t not in COMMON_STOPWORDS and len(t) >= 4]
            if len(tok_list) >= 2:
                second_word = tok_list[1]
                if len(self.second_word_index[(country, second_word)]) < self.max_posting:
                    self.second_word_index[(country, second_word)].append(eid)

            # Channel 4: Phonetic Skeleton Index
            if len(skel) >= 3 and len(self.phonetic_index[(country, skel)]) < self.max_posting:
                self.phonetic_index[(country, skel)].append(eid)

            # Channel 5: Address Number Block (same street number = very likely same business)
            for num in list(addr_nums)[:3]:  # cap at 3 numbers per record
                key = (country, "#" + num)
                if len(self.addr_num_index[key]) < min(self.max_posting, 50):  # tight cap
                    self.addr_num_index[key].append(eid)

            # Channel 6: Character trigram of compact name (catches OCR/typo variants)
            if len(compact_name) >= 5:
                for i in range(min(3, len(compact_name) - 2)):  # only first 3 trigrams
                    tg = compact_name[i:i+3]
                    if len(self.trigram_index[(country, tg)]) < min(self.max_posting, 100):
                        self.trigram_index[(country, tg)].append(eid)

            # Channel 7: Address Word Block
            for word in list(set(addr_words)):
                if len(word) >= 4 and word not in COMMON_STOPWORDS:
                    key = (country, "AW_" + word)
                    if len(self.addr_word_index[key]) < min(self.max_posting, 100):
                        self.addr_word_index[key].append(eid)
                        
            # Channel 8: Addr Num + Word (Composite, essentially unlimited cap because it's rare)
            for num in list(addr_nums)[:3]:
                for word in list(set(addr_words))[:3]:
                    if len(word) >= 4 and word not in COMMON_STOPWORDS:
                        key = (country, num, word)
                        if len(self.addr_num_word_index[key]) < self.max_posting:
                            self.addr_num_word_index[key].append(eid)
                            
            # Channel 9: Addr Word Pair (Composite)
            aw_valid = [w for w in list(set(addr_words)) if len(w) >= 4 and w not in COMMON_STOPWORDS]
            aw_list = sorted(aw_valid)
            import itertools
            for w1, w2 in itertools.combinations(aw_list[:5], 2):
                key = (country, w1, w2)
                if len(self.addr_word_pair_index[key]) < self.max_posting:
                    self.addr_word_pair_index[key].append(eid)

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
        s1_words_set = set(s1_words)
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

        # Channel 1C: Prefix-4 (only add if also shares at least first token)
        prefix4 = compact_name[:4] if len(compact_name) >= 4 else ""
        if prefix4:
            for match_id in self.prefix4_index.get((s1_country, prefix4), []):
                cand_info = self.aux_records[match_id]
                # Confirm with at least one shared meaningful token to reduce false candidates
                if s1_tokens & cand_info["tokens"] or cand_info["compact_name"][:4] == prefix4:
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

        # Channel 3B: Second Word Brand Anchor
        tok_list = [t for t in clean_name.split() if t not in COMMON_STOPWORDS and len(t) >= 4]
        if len(tok_list) >= 2:
            second_word = tok_list[1]
            for match_id in self.second_word_index.get((s1_country, second_word), []):
                cand_info = self.aux_records[match_id]
                # Require first word to also partially match (phonetic or token)
                if s1_tokens & cand_info["tokens"]:
                    candidates.add(match_id)

        # Channel 4: Phonetic Skeleton Hits
        if len(s1_skel) >= 3:
            for match_id in self.phonetic_index.get((s1_country, s1_skel), []):
                cand_info = self.aux_records[match_id]
                if not s1_nums or not cand_info["addr_nums"] or (s1_nums & cand_info["addr_nums"]):
                    candidates.add(match_id)

        # Channel 5: Address Number Block (high precision — same number = likely same location)
        for num in list(s1_nums)[:3]:
            key = (s1_country, "#" + num)
            for match_id in self.addr_num_index.get(key, []):
                cand_info = self.aux_records[match_id]
                # Require at least partial name match OR high address overlap to prevent false address collisions
                if (s1_tokens & cand_info["tokens"] or 
                    s1_first_word == cand_info["first_word"] or 
                    len(s1_words_set & cand_info["addr_tokens"]) >= 2):
                    candidates.add(match_id)

        # Channel 6: Trigram Block (catches OCR errors, abbreviations)
        if len(compact_name) >= 5:
            tg_counts = defaultdict(int)
            for i in range(min(3, len(compact_name) - 2)):
                tg = compact_name[i:i+3]
                for match_id in self.trigram_index.get((s1_country, tg), []):
                    tg_counts[match_id] += 1
            for match_id, tg_count in tg_counts.items():
                if tg_count >= 2:  # must share ≥2 trigrams
                    candidates.add(match_id)

        # Channel 7: Address Word Block (captures completely transliterated/alternate names at same address)
        if len(s1_words_set) >= 2:
            aw_counts = defaultdict(int)
            for word in s1_words_set:
                if len(word) >= 4 and word not in COMMON_STOPWORDS:
                    key = (s1_country, "AW_" + word)
                    for match_id in self.addr_word_index.get(key, []):
                        aw_counts[match_id] += 1
            for match_id, aw_count in aw_counts.items():
                if aw_count >= 2:  # must share ≥2 address words
                    candidates.add(match_id)
                    
        # Channel 8: Addr Num + Word (Composite)
        for num in list(s1_nums)[:3]:
            for word in list(s1_words_set)[:3]:
                if len(word) >= 4 and word not in COMMON_STOPWORDS:
                    key = (s1_country, num, word)
                    for match_id in self.addr_num_word_index.get(key, []):
                        candidates.add(match_id)
                        
        # Channel 9: Addr Word Pair (Composite)
        aw_valid = [w for w in list(s1_words_set) if len(w) >= 4 and w not in COMMON_STOPWORDS]
        aw_list = sorted(aw_valid)
        import itertools
        for w1, w2 in itertools.combinations(aw_list[:5], 2):
            key = (s1_country, w1, w2)
            for match_id in self.addr_word_pair_index.get(key, []):
                candidates.add(match_id)

        cand_list = list(candidates)
        if len(cand_list) > self.max_candidates:
            def cand_rank_key(cid):
                c_info = self.aux_records[cid]
                tok_overlap = len(s1_tokens & c_info["tokens"])
                num_overlap = len(s1_nums & c_info["addr_nums"])
                exact_flag = 10 if c_info["clean_name"] == clean_name else 0
                compact_flag = 5 if c_info["compact_name"] == compact_name else 0
                skel_flag = 3 if c_info["skel"] == s1_skel else 0
                word_overlap = len(s1_words_set & c_info["addr_tokens"])
                return (exact_flag * 10 + compact_flag * 5 + skel_flag * 3 + num_overlap * 4 + word_overlap * 2 + tok_overlap)

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
