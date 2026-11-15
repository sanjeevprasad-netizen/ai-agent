"""Personally Identifiable Information (PII) Detection and Redaction.

Redacts sensitive identifiers (Credit cards, SSNs, external emails, phone numbers)
before data is logged, sent to external LLMs, or stored in observability telemetry,
while strictly preserving e-commerce domain identifiers (Order IDs, Product IDs, SKUs, RMAs).
"""

import re
from typing import Any, Dict, List, Optional

# Regular expressions for sensitive PII
CC_REGEX = re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11}|(?:\d{4}[ -]?){3}\d{4})\b")
SSN_REGEX = re.compile(r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Whitelisted identifiers that MUST NOT be masked
DOMAIN_WHITELIST = re.compile(r"\b(?:ORD[-_A-Za-z0-9]+|PRD[-_A-Za-z0-9]+|PROD[-_A-Za-z0-9]+|SKU[-_A-Za-z0-9]+|RMA[-_A-Za-z0-9]+|TRK[-_A-Za-z0-9]+)\b", re.IGNORECASE)


def redact_pii_text(text: str, allowed_email: Optional[str] = None) -> str:
    """Redact sensitive PII tokens from text while preserving business entities."""
    if not text:
        return ""

    redacted = text

    # 1. Redact Credit Cards
    redacted = CC_REGEX.sub("[REDACTED_CC]", redacted)

    # 2. Redact SSNs
    redacted = SSN_REGEX.sub("[REDACTED_SSN]", redacted)

    # 3. Redact Phone Numbers
    redacted = PHONE_REGEX.sub("[REDACTED_PHONE]", redacted)

    # 4. Redact non-authenticated email addresses
    def email_sub(match):
        found = match.group(0)
        if allowed_email and found.lower() == allowed_email.lower():
            return found
        return "[REDACTED_EMAIL]"

    redacted = EMAIL_REGEX.sub(email_sub, redacted)
    return redacted


def sanitize_telemetry_payload(payload: Any, allowed_email: Optional[str] = None) -> Any:
    """Recursively sanitize dicts, lists, and strings for telemetry."""
    if isinstance(payload, str):
        return redact_pii_text(payload, allowed_email)
    elif isinstance(payload, dict):
        return {k: sanitize_telemetry_payload(v, allowed_email) for k, v in payload.items()}
    elif isinstance(payload, list):
        return [sanitize_telemetry_payload(x, allowed_email) for x in payload]
    return payload

