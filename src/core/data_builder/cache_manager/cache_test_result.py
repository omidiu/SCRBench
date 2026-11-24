from functools import lru_cache
import redis
import json
import hashlib
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)


class CacheTestResult:
    """
    Handles Redis-based caching for test execution results.
    Caches the result of running tests against source code.
    """

    CACHE_KEY_PREFIX = "test_result:cache:v2:"

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_redis_client() -> redis.Redis:
        """Get or create Redis client connection."""
        load_dotenv()
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        redis_db = int(os.getenv("REDIS_DB", "0"))
        
        return redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=redis_db,
            decode_responses=False,
            socket_connect_timeout=5,  # 5 second connection timeout
            socket_timeout=5,  # 5 second socket timeout
            retry_on_timeout=True,
            health_check_interval=30
        )

    @staticmethod
    def get_cache_key(source_code: str, tests: list[str], settings: dict) -> str:
        data = {"source_code": source_code, "tests": tests, "settings": settings}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def get(source_code: str, tests: list[str], settings: dict) -> tuple[bool, str | None] | None:
        key = CacheTestResult.CACHE_KEY_PREFIX + CacheTestResult.get_cache_key(source_code, tests, settings)
        try:
            raw = CacheTestResult._get_redis_client().get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            success, error = data["success"], data["error_message"]
            if type(success) is not bool or (error is not None and not isinstance(error, str)):
                raise ValueError("Invalid cached test result.")
            return success, error
        except (redis.RedisError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Could not read test result from Redis: %s", exc)
            return None

    @staticmethod
    def set(
        source_code: str, tests: list[str], settings: dict, success: bool, error_message: str | None
    ) -> None:
        key = CacheTestResult.CACHE_KEY_PREFIX + CacheTestResult.get_cache_key(source_code, tests, settings)
        try:
            CacheTestResult._get_redis_client().set(
                key, json.dumps({"success": success, "error_message": error_message})
            )
        except redis.RedisError as exc:
            logger.warning("Could not cache test result in Redis: %s", exc)
