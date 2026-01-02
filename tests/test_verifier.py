from autocapture.memory.verification import Claim, RulesVerifier


def test_verifier_rejects_missing_evidence() -> None:
    verifier = RulesVerifier()
    claims = [Claim(text="X", evidence_ids=[], entity_tokens=[])]
    errors = verifier.verify(claims, valid_evidence={"E1"}, entity_tokens=set())
    assert "missing evidence" in errors[0]


def test_verifier_rejects_unknown_entity() -> None:
    verifier = RulesVerifier()
    claims = [Claim(text="X", evidence_ids=["E1"], entity_tokens=["ORG_1234"])]
    errors = verifier.verify(claims, valid_evidence={"E1"}, entity_tokens={"ORG_9999"})
    assert "unknown entity" in errors[0]
