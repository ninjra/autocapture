from autocapture.text.normalize import normalize_text


def test_normalize_text_is_deterministic():
    raw = "Ticket\u00a0123"
    assert normalize_text(raw) == normalize_text(raw)


def test_normalize_text_collapses_whitespace_and_punctuation():
    raw = "Alpha\u2013Beta  \n  Gamma"
    normalized = normalize_text(raw)
    assert normalized == "Alpha-Beta Gamma"


def test_normalize_text_preserves_hyphen_tokens():
    raw = "ID-123"
    assert normalize_text(raw) == "ID-123"
