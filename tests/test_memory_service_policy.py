from autocapture.config import MemoryServicePolicyConfig
from autocapture.memory_service.policy import MemoryPolicyValidator
from autocapture.memory_service.schemas import (
    EntityRef,
    MemoryProposal,
    PolicyLabels,
    ProvenancePointer,
)


def _base_proposal(
    *,
    content_text: str = "runbook update",
    content_json: dict | None = None,
    entities: list[EntityRef] | None = None,
) -> MemoryProposal:
    return MemoryProposal(
        type="fact",
        content_text=content_text,
        content_json=content_json or {},
        importance=0.5,
        trust=0.5,
        policy=PolicyLabels(audience=["internal"], sensitivity="low"),
        entities=entities or [],
        provenance=[
            ProvenancePointer(
                artifact_version_id="artifact_v1",
                chunk_id="chunk_1",
                start_offset=0,
                end_offset=10,
                excerpt_hash="hash",
            )
        ],
    )


def test_policy_rejects_pii_secret_person_preference() -> None:
    validator = MemoryPolicyValidator(MemoryServicePolicyConfig())

    pii = _base_proposal(content_text="Contact ops@example.com for access.")
    pii_reasons = validator.validate_proposal(pii)
    assert "policy_pii_detected" in pii_reasons

    secret = _base_proposal(content_text="api_key: sk-1234567890")
    secret_reasons = validator.validate_proposal(secret)
    assert "policy_secret_detected" in secret_reasons

    person = _base_proposal(entities=[EntityRef(kind="person", name="Ada")])
    person_reasons = validator.validate_proposal(person)
    assert "policy_person_entity_detected" in person_reasons

    pref = _base_proposal(content_json={"preferences": {"theme": "dark"}})
    pref_reasons = validator.validate_proposal(pref)
    assert "policy_preference_detected" in pref_reasons

    pref_text = _base_proposal(content_text="User preferences include dark mode.")
    pref_text_reasons = validator.validate_proposal(pref_text)
    assert "policy_preference_detected" in pref_text_reasons


def test_policy_allows_safe_proposal() -> None:
    validator = MemoryPolicyValidator(MemoryServicePolicyConfig())
    proposal = _base_proposal(content_text="Deploys run nightly at 02:00 UTC.")
    assert validator.validate_proposal(proposal) == []


def test_policy_person_text_detection_toggle() -> None:
    proposal = _base_proposal(content_text="Employee: Ada Lovelace updated the runbook.")

    relaxed = MemoryPolicyValidator(MemoryServicePolicyConfig())
    assert "policy_person_text_detected" not in relaxed.validate_proposal(proposal)

    strict = MemoryPolicyValidator(MemoryServicePolicyConfig(reject_person_text=True))
    assert "policy_person_text_detected" in strict.validate_proposal(proposal)
