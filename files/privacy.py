"""
privacy.py
==========
PII scrubbing for LOGS ONLY, using Microsoft Presidio (fully open-source).

Design (agreed in Phase 4 planning):
- The REAL user message always goes to the agent untouched — Aura needs the
  actual email/name/etc. to be useful in conversation.
- Only the copy written to server logs gets scrubbed, so log files never
  contain names, emails, phone numbers, credit cards, and so on.

Fail-safe direction: if Presidio can't load (missing spaCy model, etc.) we
redact the ENTIRE message in logs rather than logging raw PII. Privacy beats
log readability.

Requires (already in your requirements.txt / setup):
  pip install presidio-analyzer presidio-anonymizer spacy
  python -m spacy download en_core_web_lg
"""

import logging

log = logging.getLogger(__name__)

_analyzer = None
_anonymizer = None
_available = False

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    # Loads the spaCy NLP model (en_core_web_lg) — takes a few seconds, once,
    # at server startup.
    _analyzer = AnalyzerEngine()
    _anonymizer = AnonymizerEngine()
    _available = True
    log.info("Presidio PII scrubber ready.")
except Exception as e:  # missing model, import error, anything
    log.warning(
        f"Presidio unavailable ({e}) — log messages will be fully redacted "
        f"instead of selectively scrubbed."
    )


def scrub_for_logging(text: str) -> str:
    """
    Return a copy of `text` safe to write to logs:
      "I'm Rahul, call me on 98765-43210"
        → "I'm <PERSON>, call me on <PHONE_NUMBER>"

    Never raises. If scrubbing isn't possible, redacts the whole message.
    """
    if not text:
        return text
    if not _available:
        return "<redacted: PII scrubber unavailable>"
    try:
        # _available is only True when both engines loaded; assert narrows the
        # Optional type so the checker knows they aren't None here.
        assert _analyzer is not None and _anonymizer is not None
        findings = _analyzer.analyze(text=text, language="en")
        # analyzer & anonymizer each declare RecognizerResult in their own module;
        # same object at runtime, so ignore the stub-level type mismatch.
        return _anonymizer.anonymize(text=text, analyzer_results=findings).text  # type: ignore[arg-type]
    except Exception as e:
        log.warning(f"PII scrub failed ({e}) — redacting whole message")
        return "<redacted: scrub failed>"