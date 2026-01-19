from autocapture.answer.coverage import coverage_metrics


def test_coverage_scoring_normal() -> None:
    text = "First sentence. [E1] Second sentence. [E1]"
    metrics = coverage_metrics(text, {"E1"}, no_evidence_mode=False)
    assert metrics["sentence_coverage"] == 1.0


def test_coverage_scoring_no_evidence_meta_excluded() -> None:
    text = "No evidence found. Try again. [E1]"
    metrics = coverage_metrics(text, {"E1"}, no_evidence_mode=True)
    assert metrics["sentence_coverage"] == 1.0
