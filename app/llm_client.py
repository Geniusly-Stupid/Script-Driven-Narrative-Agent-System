from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests


INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen2.5-7b-instruct"
logger = logging.getLogger(__name__)
_GLOBAL_RETRYABLE_COOLDOWN_UNTIL = 0.0


class RetryableLLMError(requests.exceptions.RequestException):
    def __init__(self, message: str, *, status_code: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(str(value).strip())
    except Exception:
        return None
    return max(0.0, seconds)


def _extract_retry_after_seconds(response: requests.Response) -> float | None:
    return _parse_retry_after_seconds(response.headers.get("Retry-After"))


def _step_max_tokens(step_name: str, default_max_tokens: int) -> int:
    step = str(step_name or "").strip().lower()
    if step in {"check_whether_roll_dice"}:
        return min(default_max_tokens, 256)
    if step in {"generate_retrieval_queries"}:
        return min(default_max_tokens, 512)
    if step in {
        "pre_response_transition_evaluation",
        "plot_completion_evaluation",
        "scene_completion_evaluation",
        "plot_summary_generation",
        "scene_summary_generation",
    }:
        return min(default_max_tokens, 768)
    if step in {"generate_response"}:
        return min(default_max_tokens, 1536)
    return default_max_tokens


def _wait_for_global_retryable_cooldown(step_name: str, prompt_length: int) -> None:
    global _GLOBAL_RETRYABLE_COOLDOWN_UNTIL
    remaining = _GLOBAL_RETRYABLE_COOLDOWN_UNTIL - time.monotonic()
    if remaining <= 0:
        return
    logger.info(
        "LLM global cooldown wait step=%s prompt_length=%s sleep_seconds=%.1f",
        step_name,
        prompt_length,
        remaining,
    )
    time.sleep(remaining)


def _extend_global_retryable_cooldown(seconds: float) -> None:
    global _GLOBAL_RETRYABLE_COOLDOWN_UNTIL
    if seconds <= 0:
        return
    _GLOBAL_RETRYABLE_COOLDOWN_UNTIL = max(_GLOBAL_RETRYABLE_COOLDOWN_UNTIL, time.monotonic() + seconds)


def _retry_backoff_seconds(step_name: str, attempt: int, error: Exception) -> float:
    if isinstance(error, RetryableLLMError) and error.retry_after is not None:
        return min(90.0, max(1.0, float(error.retry_after)))

    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return min(60.0, 5.0 * (2 ** (attempt - 1)))
    if status_code in {408, 409, 425}:
        return min(20.0, 2.0 * (2 ** (attempt - 1)))
    return min(20.0, 1.5 * (2 ** (attempt - 1)))


def _load_api_key() -> str:
    key_path = Path(__file__).resolve().parent.parent.joinpath("api_key.txt")
    if not key_path.exists():
        raise ValueError("api_key.txt not found in project root")
    api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
        raise ValueError("Please set a real API key in api_key.txt")
    return api_key


def call_nvidia_llm(
    prompt: str,
    model=DEFAULT_MODEL,
    *,
    step_name: str = "generation",
    max_retries: int = 3,
    timeout: int | float = 120,
    allow_env_override: bool = True,
) -> str:
    api_key = _load_api_key()

    if allow_env_override:
        model = os.getenv("NVIDIA_MODEL", model)
    max_retries = int(os.getenv("NVIDIA_MAX_RETRIES", str(max_retries)))
    max_tokens = _step_max_tokens(step_name, int(os.getenv("NVIDIA_MAX_TOKENS", "4096")))
    temperature = float(os.getenv("NVIDIA_TEMPERATURE", "0.6"))
    top_p = float(os.getenv("NVIDIA_TOP_P", "0.95"))
    top_k = int(os.getenv("NVIDIA_TOP_K", "20"))
    presence_penalty = float(os.getenv("NVIDIA_PRESENCE_PENALTY", "0"))
    repetition_penalty = float(os.getenv("NVIDIA_REPETITION_PENALTY", "1"))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    enable_thinking = os.getenv("NVIDIA_ENABLE_THINKING", "false").lower() in {"1", "true", "yes", "on"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "nvext": {
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
        },
    }

    prompt_length = len(prompt)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            _wait_for_global_retryable_cooldown(step_name, prompt_length)
            logger.info(
                "LLM request start step=%s attempt=%s/%s prompt_length=%s model=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                model,
            )
            response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=timeout)
            if not response.ok:
                if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                    raise RetryableLLMError(
                        f"retryable status={response.status_code} body={response.text}",
                        status_code=response.status_code,
                        retry_after=_extract_retry_after_seconds(response),
                    )
                raise ValueError(f"LLM request failed ({response.status_code}): {response.text}")
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            logger.warning(
                "LLM request retryable error step=%s attempt=%s/%s prompt_length=%s error=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                exc,
            )
        except RetryableLLMError as exc:
            last_error = exc
            if exc.status_code == 429:
                _extend_global_retryable_cooldown(_retry_backoff_seconds(step_name, attempt, exc))
            logger.warning(
                "LLM request network error step=%s attempt=%s/%s prompt_length=%s error=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                exc,
            )
        except requests.exceptions.Timeout as exc:
            last_error = exc
            logger.warning(
                "LLM request timeout step=%s attempt=%s/%s prompt_length=%s error=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                exc,
            )
        except requests.exceptions.RequestException as exc:
            last_error = exc
            logger.warning(
                "LLM request network error step=%s attempt=%s/%s prompt_length=%s error=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                exc,
            )
        except Exception as exc:
            logger.error(
                "LLM request failed without retry step=%s prompt_length=%s error=%s",
                step_name,
                prompt_length,
                exc,
            )
            raise

        if attempt < max_retries:
            backoff_seconds = _retry_backoff_seconds(step_name, attempt, last_error)
            if getattr(last_error, "status_code", None) == 429:
                _extend_global_retryable_cooldown(backoff_seconds)
            logger.info(
                "LLM request backoff step=%s next_attempt=%s sleep_seconds=%.1f",
                step_name,
                attempt + 1,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)

    assert last_error is not None
    logger.error(
        "LLM request exhausted retries step=%s prompt_length=%s error=%s",
        step_name,
        prompt_length,
        last_error,
    )
    raise last_error


if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = call_nvidia_llm(prompt)
    print(response)
    print("--------------------------------")
    response = call_nvidia_llm(prompt, model="qwen/qwen3.5-397b-a17b")
    print(response)
    
