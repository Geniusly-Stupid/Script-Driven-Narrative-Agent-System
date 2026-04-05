from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests


INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen2.5-7b-instruct"
logger = logging.getLogger(__name__)


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
    max_tokens = int(os.getenv("NVIDIA_MAX_TOKENS", "4096"))
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
                    raise requests.exceptions.RequestException(
                        f"retryable status={response.status_code} body={response.text}"
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
            backoff_seconds = min(8.0, 1.5 * (2 ** (attempt - 1)))
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
    
