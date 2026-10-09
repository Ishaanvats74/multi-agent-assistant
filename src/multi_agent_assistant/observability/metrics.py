
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from langchain_core.callbacks import BaseCallbackHandler

class ModelMetricsCallback(BaseCallbackHandler):
    def __init__(self, metrics: RequestMetrics):
        self.metrics = metrics

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.metrics.model_calls += 1

    def on_chat_model_start(
        self, serialized, messages, **kwargs
    ):
        self.metrics.model_calls += 1

    def on_llm_end(self, response, **kwargs):
        self.metrics.successful_model_calls += 1

        usage = response.llm_output or {}
        token_usage = usage.get("token_usage") or {}

        self.metrics.input_tokens += token_usage.get(
            "prompt_tokens", 0
        )
        self.metrics.output_tokens += token_usage.get(
            "completion_tokens", 0
        )


@dataclass
class RequestMetrics:
    started_at: float = field(default_factory=perf_counter)

    laya_calls: int = 0
    model_calls: int = 0
    successful_model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    retries: int = 0
    verification_failures: int = 0

    web_search_calls: int = 0
    web_search_successes: int = 0
    web_search_failures: int = 0
    web_search_no_results: int = 0

    stage_latency_ms: dict[str, float] = field(default_factory=dict)

    def start_stage(self) -> float:
        return perf_counter()

    def end_stage(self, name: str, started: float) -> None:
        self.stage_latency_ms[name] = round((perf_counter() - started) * 1000, 2)

    @property
    def total_latency_ms(self) -> float:
        return round((perf_counter() - self.started_at) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_latency_ms": self.total_latency_ms,
            "laya_calls": self.laya_calls,
            "model_calls": self.model_calls,
            "successful_model_calls": self.successful_model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retries": self.retries,
            "verification_failures": self.verification_failures,
            "web_search_calls": self.web_search_calls,
            "web_search_successes": self.web_search_successes,
            "web_search_failures": self.web_search_failures,
            "web_search_no_results": self.web_search_no_results,
            "stage_latency_ms": self.stage_latency_ms,
        }
