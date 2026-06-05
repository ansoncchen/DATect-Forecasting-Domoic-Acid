"""
Unit tests for backend/redis_cache.py and CacheManager resilience.

Verifies that RedisCacheManager behaves gracefully when Redis is active,
and fails safe (falling back to standard file-based caches) when Redis
connection drops or fails to initialize.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Mock out Redis before importing redis_cache to ensure we can test in any env
with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379/0"}):
    from backend.redis_cache import RedisCacheManager, get_redis_cache
    from backend.cache_manager import CacheManager


class TestRedisCacheResilience:
    @patch("redis.from_url")
    def test_successful_connection(self, mock_from_url):
        """Verify successful connection initializes flags correctly."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_from_url.return_value = mock_client

        manager = RedisCacheManager(redis_url="redis://localhost:6379/0")

        assert manager.is_available is True
        assert manager._client is mock_client
        mock_client.ping.assert_called_once()

    @patch("redis.from_url")
    def test_failed_ping_handling(self, mock_from_url):
        """Verify that connection failure does not crash the server and sets available=False."""
        mock_client = MagicMock()
        mock_client.ping.side_effect = Exception("Connection Refused")
        mock_from_url.return_value = mock_client

        manager = RedisCacheManager(redis_url="redis://localhost:6379/0")

        assert manager.is_available is False
        assert manager._connected is False

    @patch("redis.from_url")
    def test_get_and_set_with_prefix(self, mock_from_url):
        """Verify get/set commands append correct datect: prefix and handle JSON."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = '{"foo": "bar"}'
        mock_from_url.return_value = mock_client

        manager = RedisCacheManager(redis_url="redis://localhost:6379/0", prefix="test:")

        # Test set
        test_val = {"foo": "bar"}
        success = manager.set("mykey", test_val)
        assert success is True
        # Match call: setex(key, ttl, value)
        mock_client.setex.assert_called_once()
        called_key = mock_client.setex.call_args[0][0]
        called_val = mock_client.setex.call_args[0][2]
        assert called_key == "test:mykey"
        assert "bar" in called_val

        # Test get
        retrieved = manager.get("mykey")
        assert retrieved == test_val
        mock_client.get.assert_called_once_with("test:mykey")

    @patch("redis.from_url")
    def test_graceful_fallback_on_runtime_error(self, mock_from_url):
        """Verify get/set fail gracefully (returning None/False) if Redis raises exception mid-execution."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.side_effect = Exception("Redis timed out")
        mock_client.setex.side_effect = Exception("Redis write failed")
        mock_from_url.return_value = mock_client

        manager = RedisCacheManager(redis_url="redis://localhost:6379/0")

        assert manager.is_available is True

        # Mid-execution read failure
        assert manager.get("anykey") is None

        # Mid-execution write failure
        assert manager.set("anykey", "val") is False

    @patch("redis.from_url")
    def test_clear_pattern(self, mock_from_url):
        """Verify clearing pattern deletes matched keys successfully."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.keys.return_value = ["datect:key1", "datect:key2"]
        mock_from_url.return_value = mock_client

        manager = RedisCacheManager(redis_url="redis://localhost:6379/0")
        manager.clear_pattern("key*")

        mock_client.keys.assert_called_once_with("datect:key*")
        mock_client.delete.assert_called_once_with("datect:key1", "datect:key2")


class TestCacheManagerIntegration:
    def test_cache_manager_disabled_by_default_locally(self):
        """Unless ENABLE_PRECOMPUTED_CACHE=true, caching is off locally to ensure live computations."""
        with patch.dict(os.environ, {}, clear=True):
            manager = CacheManager()
            assert manager.enabled is False

    @patch("redis.from_url")
    def test_cache_manager_uses_redis_when_configured(self, mock_from_url):
        """When precomputed cache enabled and REDIS_URL exists, CacheManager reads from Redis."""
        import backend.redis_cache
        backend.redis_cache._redis_cache = None

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = '[{"date": "2023-01-01", "predicted_da": 1.2}]'
        mock_from_url.return_value = mock_client

        with patch.dict(os.environ, {"ENABLE_PRECOMPUTED_CACHE": "true", "REDIS_URL": "redis://ok"}):
            manager = CacheManager()
            assert manager.enabled is True
            assert manager.use_redis is True

            # Get retrospective forecast should query Redis
            res = manager.get_retrospective_forecast("regression", "ensemble")
            assert res is not None
            assert res[0]["predicted_da"] == 1.2
            mock_client.get.assert_called_once_with("datect:retrospective:regression:ensemble")
