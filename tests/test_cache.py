"""
缓存和限流测试
"""

import pytest
import time
from app.utils.cache import LRUCache, CacheManager, generate_cache_key, cached
from app.utils.rate_limiter import (
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    RateLimiterManager,
    rate_limit
)


class TestLRUCache:
    """LRU 缓存测试类"""

    def test_cache_init(self):
        """测试缓存初始化"""
        cache = LRUCache(max_size=100, ttl=60)
        assert cache.max_size == 100
        assert cache.ttl == 60
        assert cache.size() == 0

    def test_cache_set_get(self):
        """测试缓存设置和获取"""
        cache = LRUCache(max_size=100, ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss(self):
        """测试缓存未命中"""
        cache = LRUCache(max_size=100, ttl=60)
        assert cache.get("nonexistent") is None

    def test_cache_ttl(self):
        """测试缓存过期"""
        cache = LRUCache(max_size=100, ttl=1)
        cache.set("key1", "value1")
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_cache_max_size(self):
        """测试缓存最大容量"""
        cache = LRUCache(max_size=2, ttl=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # key1 应该被淘汰
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_cache_delete(self):
        """测试缓存删除"""
        cache = LRUCache(max_size=100, ttl=60)
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_cache_clear(self):
        """测试缓存清空"""
        cache = LRUCache(max_size=100, ttl=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.size() == 0

    def test_cache_update(self):
        """测试缓存更新"""
        cache = LRUCache(max_size=100, ttl=60)
        cache.set("key1", "value1")
        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"


class TestCacheManager:
    """缓存管理器测试类"""

    def test_get_cache(self):
        """测试获取缓存实例"""
        manager = CacheManager()
        cache1 = manager.get_cache("test", max_size=100, ttl=60)
        cache2 = manager.get_cache("test", max_size=100, ttl=60)
        assert cache1 is cache2

    def test_clear_all(self):
        """测试清空所有缓存"""
        manager = CacheManager()
        cache1 = manager.get_cache("test1")
        cache2 = manager.get_cache("test2")

        cache1.set("key1", "value1")
        cache2.set("key2", "value2")

        manager.clear_all()
        assert cache1.size() == 0
        assert cache2.size() == 0


class TestGenerateCacheKey:
    """缓存键生成测试类"""

    def test_generate_key(self):
        """测试生成缓存键"""
        key1 = generate_cache_key("arg1", "arg2")
        key2 = generate_cache_key("arg1", "arg2")
        assert key1 == key2

    def test_generate_key_different(self):
        """测试不同参数生成不同键"""
        key1 = generate_cache_key("arg1", "arg2")
        key2 = generate_cache_key("arg1", "arg3")
        assert key1 != key2

    def test_generate_key_kwargs(self):
        """测试关键字参数生成键"""
        key1 = generate_cache_key("arg1", key1="value1")
        key2 = generate_cache_key("arg1", key1="value1")
        assert key1 == key2


class TestTokenBucketRateLimiter:
    """令牌桶限流器测试类"""

    def test_acquire_success(self):
        """测试获取令牌成功"""
        limiter = TokenBucketRateLimiter(rate=10, capacity=10)
        assert limiter.acquire(1) is True

    def test_acquire_failure(self):
        """测试获取令牌失败"""
        limiter = TokenBucketRateLimiter(rate=1, capacity=1)
        assert limiter.acquire(1) is True
        assert limiter.acquire(1) is False

    def test_token_refill(self):
        """测试令牌补充"""
        limiter = TokenBucketRateLimiter(rate=10, capacity=10)
        limiter.acquire(10)
        time.sleep(0.1)
        assert limiter.acquire(1) is True


class TestSlidingWindowRateLimiter:
    """滑动窗口限流器测试类"""

    def test_is_allowed(self):
        """测试请求允许"""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=1)
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False

    def test_get_remaining(self):
        """测试获取剩余请求数"""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=1)
        assert limiter.get_remaining("user1") == 2
        limiter.is_allowed("user1")
        assert limiter.get_remaining("user1") == 1

    def test_reset(self):
        """测试重置限制"""
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=1)
        limiter.is_allowed("user1")
        assert limiter.is_allowed("user1") is False
        limiter.reset("user1")
        assert limiter.is_allowed("user1") is True


class TestRateLimiterManager:
    """限流管理器测试类"""

    def test_get_token_bucket(self):
        """测试获取令牌桶限流器"""
        manager = RateLimiterManager()
        limiter1 = manager.get_token_bucket("test", rate=10, capacity=10)
        limiter2 = manager.get_token_bucket("test", rate=10, capacity=10)
        assert limiter1 is limiter2

    def test_get_sliding_window(self):
        """测试获取滑动窗口限流器"""
        manager = RateLimiterManager()
        limiter1 = manager.get_sliding_window("test", max_requests=10, window_seconds=60)
        limiter2 = manager.get_sliding_window("test", max_requests=10, window_seconds=60)
        assert limiter1 is limiter2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
