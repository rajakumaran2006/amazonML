"""
ML Challenge 2026: Business Entity Resolution
Module: Linguistic Normalization and Preprocessing for US, India, and France
Handles Latin, French accents, domain names, and Indian multilingual transliteration (Devanagari, Telugu, Malayalam, Tamil, etc.).
"""

import re
import unicodedata
from typing import List, Set, Tuple
from unidecode import unidecode
from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS

RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Domain suffixes to strip from web names (e.g., wilfordhancock.com -> wilfordhancock)
RE_DOMAIN_SUFFIX = re.compile(
    r"\b([a-zA-Z0-9\-]+)\.(com|org|net|in|co\.in|fr|io|biz|info|gov|edu)\b",
    re.IGNORECASE,
)
RE_WWW = re.compile(r"\bwww\.", re.IGNORECASE)

# Legal suffixes across US, India, and France (Zero-Shot)
LEGAL_SUFFIX_PATTERNS = [
    # US
    r"\bincorporated\b", r"\binc\b",
    r"\bcorporation\b", r"\bcorp\b",
    r"\blimited liability company\b", r"\bllc\b",
    r"\blimited liability partnership\b", r"\bllp\b",
    r"\bcompany\b", r"\bco\b",
    # India (English + Devanagari + Indic transliterations)
    r"\bprivate limited\b", r"\bpvt ltd\b", r"\bpvt limited\b", r"\bprivate ltd\b",
    r"\bpvt\b", r"\blimited\b", r"\bltd\b",
    r"\bpraiveett limittedd\b", r"\bpraiveta limiteda\b", r"\bpraivrrrr limirrrrdd\b",
    r"\bindustries\b", r"\binddsttriis\b", r"\benterprises\b", r"\binfra\b", r"\binphraa\b",
    r"\bservices\b", r"\bsolutions\b",
    r"\bप्राइवेट लिमिटेड\b", r"\bप्राइवेट लि\b", r"\bलिमिटेड\b",
    r"\bప్రైవేట్ లిమిటెడ్\b", r"\bപ്രൈവറ്റ് ലിമിറ്റഡ്\b",
    # France (Zero-Shot)
    r"\bsarl\b", r"\bsas\b", r"\bs\.a\.s\b", r"\bsa\b", r"\bs\.a\b",
    r"\beurl\b", r"\bsnc\b", r"\bsci\b", r"\bei\b",
    r"\bsociete\b", r"\bste\b",
]

RE_LEGAL_SUFFIXES = re.compile(r"|".join(LEGAL_SUFFIX_PATTERNS), re.IGNORECASE)

# Roadway abbreviations
ROADWAY_MAP = {
    "rd": "road", "rd.": "road",
    "st": "street", "st.": "street",
    "ave": "avenue", "ave.": "avenue", "av": "avenue", "av.": "avenue",
    "blvd": "boulevard", "blvd.": "boulevard", "bd": "boulevard", "bd.": "boulevard",
    "hwy": "highway", "hwy.": "highway",
    "dr": "drive", "dr.": "drive",
    "ln": "lane", "ln.": "lane",
    "r.": "rue", "r": "rue",
    "allée": "allee", "allee": "allee",
}

RE_NUMBERS = re.compile(r"\b\d+\b")
RE_SPACES = re.compile(r"\s+")
RE_DUPLICATE_CHARS = re.compile(r"(.)\1+")
RE_VOWELS = re.compile(r"[aeiouy]")


def is_allowed_char(ch: str) -> bool:
    """
    Returns True if character is a Letter (L), Mark/Vowel sign (M), Number (N), or Space (Z).
    Guarantees that Indic vowel signs and French accents are preserved.
    """
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "M", "N") or cat.startswith("Z")


def transliterate_text(text: str) -> str:
    """
    Converts Indic/Devanagari text to Roman phonetic script.
    """
    if RE_DEVANAGARI.search(text):
        try:
            return transliterate(text, DEVANAGARI, ITRANS).lower()
        except Exception:
            return unidecode(text).lower()
    return unidecode(text).lower()


def normalize_text(text: str) -> str:
    """
    Applies unicode NFKC normalization, transliterates if needed, strips punctuation safely.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFKC", text)

    # Clean domain wrappers if present
    text = RE_WWW.sub("", text)
    text = RE_DOMAIN_SUFFIX.sub(r"\1", text)

    # Replace common symbols with space or word equivalent
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace(".", " ")
    text = text.replace(",", " ")

    # Unidecode directly if non-ascii
    if any(ord(c) > 127 for c in text):
        text = transliterate_text(text)

    # Filter out punctuation and symbols
    cleaned_chars = [ch if is_allowed_char(ch) else " " for ch in text]
    norm = "".join(cleaned_chars)
    norm = RE_SPACES.sub(" ", norm).strip().lower()
    return norm


def clean_business_name(name: str) -> str:
    """
    Produces a canonical business name with legal suffixes stripped and transliterated.
    """
    norm = normalize_text(name)
    if not norm:
        return ""
    cleaned = RE_LEGAL_SUFFIXES.sub("", norm)
    cleaned = RE_SPACES.sub(" ", cleaned).strip()
    return cleaned if cleaned else norm


def phonetic_skeleton(name: str) -> str:
    """
    Generates a phonetic consonant skeleton for cross-lingual / cross-script matching.
    Example: 'Premier' -> 'prmr', 'ప్రీమియర్' -> 'prmr'.
    """
    text = unidecode(name).lower()
    text = re.sub(r"[^a-z]", "", text)
    text = RE_DUPLICATE_CHARS.sub(r"\1", text)
    if not text:
        return ""
    first_char = text[0]
    rest = RE_VOWELS.sub("", text[1:])
    return first_char + rest


def extract_address_tokens(address: str) -> Tuple[List[str], Set[str]]:
    """
    Returns (word_tokens, numeric_tokens) for an address string.
    Standardizes roadway terms and extracts all building/postal numbers.
    """
    norm = normalize_text(address)
    if not norm:
        return [], set()

    raw_tokens = norm.split()
    tokens = [ROADWAY_MAP.get(tok, tok) for tok in raw_tokens if len(tok) >= 3]
    numeric_tokens = set(RE_NUMBERS.findall(norm))

    return tokens, numeric_tokens


def get_character_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    Extracts character n-grams from text for fuzzy matching.
    """
    norm = normalize_text(text).replace(" ", "")
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}
