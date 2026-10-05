from laya import Router

router = Router(preload=True)

questions = {
    "intent": {
        "type": "choice",
        "instructions": """Determine what the user primarily wants.

            technical means the user wants an explanation, solution,
            implementation, debugging, or analysis of a technical problem.

            task_planning means the user wants a plan, schedule, roadmap,
            milestones, checklist, or organized sequence of actions.

            financial means the user wants budgeting, expense analysis,
            savings calculations, or financial planning.
            """,
        "criteria": {
            "technical": "Solve or explain a technical problem.",
            "planning": "Create or organize a plan for achieving a goal.",
            "financial": "Analyze or plan money, expenses, savings, or a budget."
        }
    }
}

def laya_router(state: dict):
    res = router.predict(state, questions)

    return {
        "intent": res["answers"]["intent"]["choice"]
    }