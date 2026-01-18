from pathlib import Path

from autocapture.eval_gate import assert_gate, load_baseline, run_retrieval_eval


def test_retrieval_eval_gate():
    dataset = Path("evals/phase0_retrieval.jsonl")
    baseline = Path("evals/phase0_retrieval_baseline.json")
    metrics = run_retrieval_eval(dataset)
    assert_gate(metrics, load_baseline(baseline))
