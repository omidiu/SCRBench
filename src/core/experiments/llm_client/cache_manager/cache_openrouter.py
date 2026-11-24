from typing import Dict, Optional, Tuple
import datetime
import redis
import json
import hashlib
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)


class CacheOpenRouter:
    """
    Handles Redis-based caching for OpenRouter API responses.
    Checks if payload exists in cache before making API calls.
    """

    CACHE_KEY_PREFIX = "openrouter:cache:"

    @staticmethod
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
        )

    @staticmethod
    def get_payload_hash(payload: Dict) -> str:
        """
        Generate a hash for the payload to use as cache key.

        Args:
            payload: The API request payload dictionary

        Returns:
            SHA256 hash of the payload as a hex string
        """
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode()).hexdigest()

    @staticmethod
    def get(payload_hash: str) -> Optional[Tuple[str, Optional[datetime.datetime]]]:
        """
        Load a cached response from Redis if it exists.

        Args:
            payload_hash: The hash of the payload to look up

        Returns:
            Tuple of (response_content, timestamp) if found, None otherwise
        """
        try:
            redis_client = CacheOpenRouter._get_redis_client()
            cache_key = f"{CacheOpenRouter.CACHE_KEY_PREFIX}{payload_hash}"
            cached_data = redis_client.get(cache_key)

            if cached_data:
                data = json.loads(cached_data.decode("utf-8"))
                response_content = data["response"]
                timestamp_str = data.get("timestamp")

                if timestamp_str:
                    timestamp = datetime.datetime.fromisoformat(timestamp_str)
                else:
                    timestamp = None

                logger.info(f"Cache hit for payload hash: {payload_hash[:8]}...")
                return response_content, timestamp
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error: {e}. Cache disabled for this request.")
        except Exception as e:
            logger.warning(f"Failed to load from Redis cache: {e}")

        return None

    @staticmethod
    def set(payload_hash: str, response_content: str, timestamp: Optional[datetime.datetime]) -> None:
        """
        Save a response to Redis cache.

        Args:
            payload_hash: The hash of the payload used as cache key
            response_content: The API response content to cache
            timestamp: Optional server timestamp from the API response
        """
        try:
            redis_client = CacheOpenRouter._get_redis_client()
            cache_key = f"{CacheOpenRouter.CACHE_KEY_PREFIX}{payload_hash}"

            data = {"response": response_content, "timestamp": timestamp.isoformat() if timestamp else None}

            redis_client.set(cache_key, json.dumps(data))
            logger.info(f"Cached response in Redis for payload hash: {payload_hash[:8]}...")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error: {e}. Could not cache response.")
        except Exception as e:
            logger.warning(f"Failed to save to Redis cache: {e}")

    @staticmethod
    def delete(payload_hash: str) -> int:
        """
        Delete a cached response from Redis if it exists.

        Args:
            payload_hash: The hash of the payload used as cache key

        Returns:
            Number of keys deleted (0 or 1 under normal operation).
        """
        try:
            redis_client = CacheOpenRouter._get_redis_client()
            cache_key = f"{CacheOpenRouter.CACHE_KEY_PREFIX}{payload_hash}"
            deleted = int(redis_client.delete(cache_key))
            if deleted:
                logger.info(f"Deleted Redis cache for payload hash: {payload_hash[:8]}...")
            return deleted
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error: {e}. Could not delete cache entry.")
        except Exception as e:
            logger.warning(f"Failed to delete Redis cache entry: {e}")
        return 0

    @staticmethod
    def delete_by_payload(payload: Dict) -> int:
        """
        Convenience wrapper: hash a payload and delete the corresponding cache entry.
        """
        payload_hash = CacheOpenRouter.get_payload_hash(payload)
        return CacheOpenRouter.delete(payload_hash)


