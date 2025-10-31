import re
from pathlib import Path
from typing import Dict, List

# Simple regex-based PII detectors
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ssn_us": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"),
}


def scan_text_for_pii(text: str) -> Dict[str, List[str]]:
    hits = {}
    if not text:
        return hits
    for label, pat in PII_PATTERNS.items():
        found = pat.findall(text)
        if found:
            hits[label] = found
    return hits


def scan_dataframe_for_pii(df, column="texto"):
    results = {}
    for i, v in enumerate(df.get(column, [])):
        found = scan_text_for_pii(str(v))
        if found:
            results[i] = found
    return results


def redact_text(text: str) -> str:
    red = text
    red = PII_PATTERNS["email"].sub("[REDACTED_EMAIL]", red)
    red = PII_PATTERNS["ssn_us"].sub("[REDACTED_SSN]", red)
    red = PII_PATTERNS["credit_card"].sub("[REDACTED_CC]", red)
    return red
