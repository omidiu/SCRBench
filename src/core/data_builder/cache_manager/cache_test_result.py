from typing import Optional, Tuple, List, Union
import redis
import json
import hashlib
from dotenv import load_dotenv
import os
import logging
import ast

logger = logging.getLogger(__name__)


class CacheTestResult:
    """
    Handles Redis-based caching for test execution results.
    Caches the result of running tests against source code.
    """

    CACHE_KEY_PREFIX = "test_result:cache:"

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
            socket_connect_timeout=5,  # 5 second connection timeout
            socket_timeout=5,  # 5 second socket timeout
            retry_on_timeout=True,
            health_check_interval=30
        )

    @staticmethod
    def _normalize_source_code(source_code: str) -> str:
        """
        Normalize source code string to ensure consistent hashing.
        - Strip leading/trailing whitespace
        - Normalize line endings
        - Remove trailing whitespace from each line
        - Remove leading/trailing empty lines
        
        Args:
            source_code: The source code string
            
        Returns:
            Normalized source code string
        """
        # Normalize line endings to \n
        code = source_code.replace('\r\n', '\n').replace('\r', '\n')
        # Split into lines, strip trailing whitespace, and rejoin
        lines = [line.rstrip() for line in code.split('\n')]
        # Remove leading empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        # Remove trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()
        # Rejoin with normalized line endings
        return '\n'.join(lines)

    @staticmethod
    def _normalize_tests(tests: Union[List[str], str]) -> List[str]:
        """
        Normalize tests to a list of strings for consistent hashing.
        
        Args:
            tests: Either a list of test strings or a string representation
            
        Returns:
            List of test strings
        """
        if isinstance(tests, str):
            try:
                tests = ast.literal_eval(tests)
            except (ValueError, SyntaxError):
                # TODO: CHANGE THIS
                tests = [tests]
        
        if not isinstance(tests, list):
            tests = [str(tests)]
        
        # Normalize: ensure all are strings and sort for consistent hashing
        return sorted([str(test) for test in tests])

    @staticmethod
    def get_cache_key(source_code: str, tests: Union[List[str], str]) -> str:
        """
        Generate a cache key based on source code and tests.
        
        Args:
            source_code: The source code string
            tests: List of test strings or string representation
            
        Returns:
            SHA256 hash of the combined source code and tests
        """
        # Normalize both source code and tests to ensure consistent hashing
        normalized_source_code = CacheTestResult._normalize_source_code(source_code)
        normalized_tests = CacheTestResult._normalize_tests(tests)
        
        # Create a dictionary with sorted keys for consistent hashing
        cache_data = {
            'source_code': normalized_source_code,
            'tests': normalized_tests
        }
        
        # Convert to JSON string with sorted keys
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    @staticmethod
    def get(source_code: str, tests: Union[List[str], str]) -> Optional[Tuple[bool, Optional[str]]]:
        """
        Load a cached test result from Redis if it exists.
        
        Args:
            source_code: The source code string
            tests: List of test strings or string representation
            
        Returns:
            Tuple of (success: bool, error_message: Optional[str]) if found, None otherwise
        """
        try:
            redis_client = CacheTestResult._get_redis_client()
            cache_key_hash = CacheTestResult.get_cache_key(source_code, tests)
            cache_key = f"{CacheTestResult.CACHE_KEY_PREFIX}{cache_key_hash}"
            cached_data = redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data.decode('utf-8'))
                success = data['success']
                error_message = data.get('error_message')
                
                logger.debug(f"Cache hit for test result: {cache_key_hash[:8]}...")
                return (success, error_message)
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error: {e}. Cache disabled for this request.")
        except Exception as e:
            logger.warning(f"Failed to load from Redis cache: {e}")
        
        return None

    @staticmethod
    def set(source_code: str, tests: Union[List[str], str], success: bool, error_message: Optional[str] = None) -> None:
        """
        Save a test result to Redis cache.
        
        Args:
            source_code: The source code string
            tests: List of test strings or string representation
            success: Whether the tests passed (True) or failed (False)
            error_message: Optional error message if tests failed
        """
        try:
            redis_client = CacheTestResult._get_redis_client()
            cache_key_hash = CacheTestResult.get_cache_key(source_code, tests)
            cache_key = f"{CacheTestResult.CACHE_KEY_PREFIX}{cache_key_hash}"
            
            data = {
                'success': success,
                'error_message': error_message
            }
            
            redis_client.set(cache_key, json.dumps(data))
            logger.debug(f"Cached test result in Redis: {cache_key_hash[:8]}... (success={success})")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error: {e}. Could not cache response.")
        except Exception as e:
            logger.warning(f"Failed to save to Redis cache: {e}")

