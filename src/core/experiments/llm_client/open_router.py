import os
import json
import logging
import time
import datetime
from typing import Optional, Tuple, NamedTuple
from email.utils import parsedate_to_datetime
from dotenv import load_dotenv

from .cache_manager.cache_openrouter import CacheOpenRouter

# Avoid indefinite freeze when API hangs (e.g. after many requests). Fail after this many seconds.
OPENROUTER_REQUEST_TIMEOUT = 20
# Retry these status codes with backoff (transient / rate limit).
RETRYABLE_STATUS_CODES = (408, 429, 500, 502, 503)
MAX_RETRIES = 4
INITIAL_BACKOFF_SEC = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def query_openrouter_chat_model(
    prompt: str,
    model: str,
    max_tokens: int = 4000,
    temperature: int = 0,
) -> Tuple[str, Optional[datetime.datetime]]:
    """
    Sends a prompt to the OpenRouter API and returns the model's response along with the server's timestamp.
    Reads from cache on hit; does not write to cache (callers that parse responses must commit via
    `commit_openrouter_cache` after a successful parse).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    payload_hash = CacheOpenRouter.get_payload_hash(payload)
    cached_response = CacheOpenRouter.get(payload_hash)
    if cached_response is not None:
        return cached_response

    logger.info(f"Cache miss for payload hash: {payload_hash[:8]}... Making API call.")
    response_content, server_datetime = _make_api_call(payload)
    return response_content, server_datetime


class OpenRouterQueryResult(NamedTuple):
    response_content: str
    server_datetime: Optional[datetime.datetime]
    cache_hit: bool
    payload_hash: str


def commit_openrouter_cache(result: OpenRouterQueryResult) -> None:
    """Persist a successfully parsed response. Skips if the response already came from cache."""
    if not result.cache_hit:
        CacheOpenRouter.set(result.payload_hash, result.response_content, result.server_datetime)


def query_openrouter_chat_model_with_cache_info(
    prompt: str,
    model: str,
    max_tokens: int = 4000,
    temperature: int = 0,
) -> OpenRouterQueryResult:
    """
    Same as `query_openrouter_chat_model`, but also returns cache metadata.
    Does not write to cache; call `commit_openrouter_cache` after a successful parse.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    payload_hash = CacheOpenRouter.get_payload_hash(payload)
    cached_response = CacheOpenRouter.get(payload_hash)
    if cached_response is not None:
        response_content, server_datetime = cached_response
        return OpenRouterQueryResult(response_content, server_datetime, True, payload_hash)

    logger.info(f"Cache miss for payload hash: {payload_hash[:8]}... Making API call.")
    response_content, server_datetime = _make_api_call(payload)
    return OpenRouterQueryResult(response_content, server_datetime, False, payload_hash)


def _make_api_call(payload: dict) -> Tuple[str, Optional[datetime.datetime]]:
    """
    Makes an API call to OpenRouter and returns the response.
    Retries on 408, 429, 500, 502, 503 with exponential backoff. Uses a request timeout to avoid freezing.
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

    import requests

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    last_exc = None
    backoff = INITIAL_BACKOFF_SEC

    for attempt in range(MAX_RETRIES):
        logger.warning(f"[openrouter] _make_api_call: attempt {attempt + 1}/{MAX_RETRIES}")
        try:
            response = requests.post(
                url=url,
                headers=headers,
                data=json.dumps(payload),
                timeout=OPENROUTER_REQUEST_TIMEOUT,
            )
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            logger.warning(f"[openrouter] _make_api_call: attempt {attempt + 1} failed: {e}")
            if attempt + 1 == MAX_RETRIES:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
            continue

        logger.warning("[openrouter] _make_api_call: response received")

        if response.status_code in RETRYABLE_STATUS_CODES:
            retry_after = response.headers.get("Retry-After")
            wait = backoff
            if retry_after is not None:
                try:
                    wait = max(wait, float(retry_after))
                except ValueError:
                    pass
            logger.warning(
                f"[openrouter] _make_api_call: status={response.status_code}, retry in {wait:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
            )
            if attempt + 1 == MAX_RETRIES:
                response.raise_for_status()
            time.sleep(wait)
            backoff = min(backoff * 2, 120)
            continue

        response.raise_for_status()
        server_timestamp_str = response.headers.get("Date")
        server_datetime = parsedate_to_datetime(server_timestamp_str) if server_timestamp_str else None
        response_content = response.json()["choices"][0]["message"]["content"]
        return response_content, server_datetime

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("_make_api_call: unexpected exit from retry loop")


def get_openrouter_auth_status():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

    import requests

    response = requests.get(
        url="https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    print(json.dumps(response.json(), indent=2))

