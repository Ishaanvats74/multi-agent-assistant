# multi-agent-assistant

A multi-agent personal assistant built with **Laya, LangGraph, LangChain, FastAPI, and Groq API**.

routes user requests to specialized agents, generates responses, and verifies them before returning the final answer.

## How It Works

```text
User Query
    |
    v
Laya Router
    |
    v
Specialist Agent
(Technical / Planning / Finance)
    |
    v
Verification Agent
    |
    +---- Approved ----> Final Response
    |
    +---- Rejected ----> Retry Specialist
                              |
                              v
                         Re-verify
```

## Features

- Intent-based routing with Laya
- Specialized Technical, Planning, and Finance agents
- Workflow orchestration using LangGraph
- Agent creation and structured outputs using LangChain
- Response verification with bounded retries
- LLM inference through Groq API
- REST API powered by FastAPI

## Tech Stack

- **Python** — Backend
- **FastAPI** — API server
- **Laya** — Intent classification and routing
- **LangGraph** — Workflow orchestration
- **LangChain** — LLM integration and agent creation
- **Groq API** — LLM inference
- **Pydantic** — Data validation
- **uv** — Dependency and environment management

## Agents

| Agent | Responsibility |
|---|---|
| Laya Router | Classifies requests and selects a specialist |
| Technical Agent | Programming, debugging, APIs, databases, and architecture |
| Planning Agent | Study plans, roadmaps, schedules, and task breakdowns |
| Finance Agent | Budgeting, savings, expense analysis, and financial concepts |
| Verification Agent | Evaluates responses for relevance, completeness, correctness, and user constraints |

When verification fails, LangGraph can route the response back to the same specialist for revision, subject to the configured retry limit.

## Project Structure

```text
multi-agent-assistant/
├── src/
│   └── multi_agent_assistant/
│       ├── main.py
│       ├── router/
│       ├── agents/
│       ├── graph/
│       ├── models/
│       ├── tools/
│       ├── observability/
│       └── config.py
├── test/
├── .github/
│   └── workflows/
├── pyproject.toml
├── uv.lock
├── .env.example
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ishaanvats74/multi-agent-assistant.git
cd multi-agent-assistant
```

### 2. Install dependencies

Install [uv](https://docs.astral.sh/uv/) if needed, then run:

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file based on `.env.example` and configure your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key,
DIFFBOT_API_TOKEN=your_diffbot_api_key
```

Add any other environment variables required by your configuration. Never commit secrets to Git.

## Run the API

```bash
uv run uvicorn multi_agent_assistant.main:app --app-dir src --reload
```

API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## API Example

**Request**

`POST /chat`

```json
{
  "query_input": "Create a 30-day DSA study plan."
}
```

**Example response**

```json
{
  "agent_response": "Your 30-day DSA study plan...",
  "route": "planning",
  "verified": true,
  "retry_count": 0,
  "verification_issues": []
}
```

The response illustrates the workflow's output; refer to the actual API implementation for the exact schema.

## Testing

Run the test suite:

```bash
uv run pytest -q
```

## License

This project is intended for learning, experimentation.

---

Built by [Ishaan Vats](https://github.com/Ishaanvats74)