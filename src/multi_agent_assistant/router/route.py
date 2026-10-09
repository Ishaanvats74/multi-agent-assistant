import logging
from laya import Router

logger = logging.getLogger(__name__)
router = Router(preload=True)

ALLOWED_INTENTS = {"technical", "planning", "financial"}

questions = {
    "intent": {
        "type": "choice",
        "instructions": """
        Determine the user's primary intent.

        technical: Explanation, implementation, debugging,
        or analysis of a technical problem.

        planning: A plan, schedule, roadmap, milestones,
        checklist, or organized sequence of actions.

        financial: Budgeting, expenses, savings,
        or financial planning.
        """,
        "criteria": {
            "technical": "Solve or explain a technical problem.",
            "planning": "Create or organize a plan.",
            "financial": "Analyze budgets, expenses, or savings."
        }
    }
}


def laya_router(state: dict) -> dict:
    if not isinstance(state, dict):
        raise ValueError("Router state must be a dictionary.")

    if not state.get("query_input"):
        raise ValueError("query_input is required.")

    try:
        res = router.predict(state, questions)

    except Exception as exc:
        logger.exception("Laya prediction failed")
        raise RuntimeError("Intent routing is temporarily unavailable.") from exc

    if not isinstance(res, dict):
        raise RuntimeError("Laya returned an invalid response.")

    answers = res.get("answers")

    if not isinstance(answers, dict):
        raise RuntimeError("Laya returned no valid answers.")

    intent_result = answers.get("intent")

    if not isinstance(intent_result, dict):
        raise RuntimeError("Laya did not return an intent.")

    intent = intent_result.get("choice")

    if not isinstance(intent, str) or intent not in ALLOWED_INTENTS:
        raise RuntimeError("Laya returned an invalid intent.")

    return {"intent": intent}
