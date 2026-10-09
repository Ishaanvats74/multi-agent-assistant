FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml .
COPY README.md .

RUN pip install uv
RUN uv sync --no-install-project

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.multi_agent_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]