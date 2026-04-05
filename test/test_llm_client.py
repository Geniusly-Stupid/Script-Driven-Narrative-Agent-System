import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm_client import RetryableLLMError, _retry_backoff_seconds, _step_max_tokens


def main() -> int:
    try:
        print('[test_llm_client] case 1: small evaluator steps should stay tightly capped')
        assert _step_max_tokens('check_whether_roll_dice', 4096) == 256
        assert _step_max_tokens('generate_retrieval_queries', 4096) == 512
        assert _step_max_tokens('plot_completion_evaluation', 4096) == 768

        print('[test_llm_client] case 2: narrative generation should no longer request the full 4k budget by default')
        assert _step_max_tokens('generate_response', 4096) == 1536
        assert _step_max_tokens('generate_response', 1200) == 1200

        print('[test_llm_client] case 3: retry-after and 429 backoff should stay conservative but bounded')
        explicit_retry_after = RetryableLLMError('retry later', status_code=429, retry_after=17)
        plain_429 = RetryableLLMError('too many requests', status_code=429, retry_after=None)
        assert _retry_backoff_seconds('generate_response', 1, explicit_retry_after) == 17
        assert _retry_backoff_seconds('generate_response', 1, plain_429) == 5
        assert _retry_backoff_seconds('generate_response', 2, plain_429) == 10

        print('[test_llm_client] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_llm_client] result: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
