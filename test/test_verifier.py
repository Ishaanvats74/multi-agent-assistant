from multi_agent_assistant.agents.verifier_agent import (
    VerificationResult,
)


def test_verification_result_approved():
    result = VerificationResult(
        approved=True,
        issues=[],
        reason="The response completely satisfies the request.",
    )

    assert result.approved is True
    assert result.issues == []
    assert result.reason


def test_verification_result_rejected():
    result = VerificationResult(
        approved=False,
        issues=[
            "The response does not satisfy the requested constraint."
        ],
        reason="The answer is incomplete.",
    )

    assert result.approved is False
    assert len(result.issues) == 1
    assert result.reason


def test_verification_result_multiple_issues():
    result = VerificationResult(
        approved=False,
        issues=[
            "Missing required information.",
            "Incorrect constraint.",
        ],
        reason="Multiple problems were found.",
    )

    assert result.approved is False
    assert len(result.issues) == 2
