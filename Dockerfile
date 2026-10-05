FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml .
COPY README.md .

RUN uv sync

COPY . .

EXPOSE 8000

CMD ["uv","run","fastapi","dev"]