from __future__ import annotations

import os
from pathlib import Path

import requests


INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen2.5-7b-instruct"


def _load_api_key() -> str:
    key_path = Path(__file__).resolve().parent.parent.joinpath("api_key.txt")
    if not key_path.exists():
        raise ValueError("api_key.txt not found in project root")
    api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
        raise ValueError("Please set a real API key in api_key.txt")
    return api_key


def call_nvidia_llm(prompt: str, model=DEFAULT_MODEL) -> str:
    api_key = _load_api_key()

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

    response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=120)
    if not response.ok:
        raise ValueError(f"LLM request failed ({response.status_code}): {response.text}")
    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = call_nvidia_llm(prompt)
    print(response)
    print("--------------------------------")
    response = call_nvidia_llm(prompt, model="qwen/qwen3.5-397b-a17b")
    print(response)
    
