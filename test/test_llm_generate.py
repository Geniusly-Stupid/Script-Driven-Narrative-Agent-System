# refer to https://build.nvidia.com/qwen/qwen3.5-397b-a17b

import requests, base64, json
from pathlib import Path

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True

api_key = Path(__file__).resolve().parent.parent.joinpath("api_key.txt").read_text().strip()

if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
    raise ValueError("Please set a real API key in api_key.txt")

def read_b64(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()

headers = {
  "Authorization": f"Bearer {api_key}",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "qwen/qwen3.5-397b-a17b",
  "messages": [{"role":"user","content":"Which number is larger, 9.11 or 9.8?"}],
  "max_tokens": 1024,
  "temperature": 0.60,
  "top_p": 0.95,
  "top_k": 20,
  "presence_penalty": 0,
  "repetition_penalty": 1,
  "stream": stream,
  "chat_template_kwargs": {"enable_thinking":True},
}

response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)

# if stream:
#     for line in response.iter_lines():
#         if line:
#             print(line.decode("utf-8"))
# else:
#     print(response.json())


if stream:
    print("start generating...\n")
    
    reasoning_text = ""
    final_text = ""

    for raw_line in response.iter_lines():
        if not raw_line:
            continue

        line = raw_line.decode("utf-8").strip()

        if line == "data: [DONE]":
            break

        if not line.startswith("data:"):
            continue

        json_str = line[len("data:"):].strip()

        try:
            data = json.loads(json_str)
            delta = data["choices"][0]["delta"]

            # Reasoning
            if "reasoning" in delta and delta["reasoning"]:
                reasoning_text += delta["reasoning"]

            # Final visible answer
            if "content" in delta and delta["content"]:
                final_text += delta["content"]

        except Exception:
            continue

    # =========================
    # 统一打印
    # =========================
    
    if reasoning_text:
        print("==== REASONING ====\n")
        print(reasoning_text.strip())
        print("\n")

    print("==== FINAL ANSWER ====\n")
    print(final_text.strip())
else:
    print(response.json())
