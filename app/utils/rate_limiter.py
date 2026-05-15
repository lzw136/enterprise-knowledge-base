"""
限流模块
提供令牌桶和滑动窗口限流算法
"""

import time
from typing import Dict, Optional
from collections import defaultdict
import threading

from app.core.logger import app_logger


class TokenBucketRateLimiter:
    """
    令牌桶限流器

    允许突发流量，同时限制平均速率
    """

    def __init__(self, rate: float, capacity: int):
        """
        初始化令牌桶

        Args:
            rate: 令牌生成速率（个/秒）
            capacity: 桶容量
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_time = time.time()
        self.lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """
        获取令牌

        Args:
            tokens: 需要的令牌数

        Returns:
            是否获取成功
        """
        with self.lock:
            now = time.time()
            # 添加新令牌
            elapsed = now - self.last_time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_time = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait(self, tokens: int = 1, timeout: float = None) -> bool:
        """
        等待获取令牌

        Args:
            tokens: 需要的令牌数
            timeout: 超时时间（秒）

        Returns:
            是否获取成功
        """
        start_time = time.time()
        while True:
            if self.acquire(tokens):
                return True

            if timeout is not None and time.time() - start_time >= timeout:
                return False

            # 计算需要等待的时间
            wait_time = (tokens - self.tokens) / self.rate
            time.sleep(min(wait_time, 0.1))


class SlidingWindowRateLimiter:
    """
    滑动窗口限流器

    精确限制时间窗口内的请求数
    """

    def __init__(self, max_requests: int, window_seconds: int):
        """
        初始化滑动窗口限流器

        Args:
            max_requests: 窗口内最大请求数
            window_seconds: 窗口大小（秒）
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        """
        检查是否允许请求

        Args:
            key: 请求标识（如用户ID、IP等）

        Returns:
            是否允许
        """
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds

            # 清理过期请求
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > window_start
            ]

            # 检查是否超过限制
            if len(self.requests[key]) >= self.max_requests:
                return False

            # 记录请求
            self.requests[key].append(now)
            return True

    def get_remaining(self, key: str) -> int:
        """
        获取剩余请求数

        Args:
            key: 请求标识

        Returns:
            剩余请求数
        """
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds

            # 清理过期请求
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > window_start
            ]

            return max(0, self.max_requests - len(self.requests[key]))

    def reset(self, key: str):
        """
        重置某个键的限制

        Args:
            key: 请求标识
        """
        with self.lock:
            if key in self.requests:
                del self.requests[key]


class RateLimiterManager:
    """
    限流管理器

    管理多个限流实例
    """

    def __init__(self):
        self.token_buckets: Dict[str, TokenBucketRateLimiter] = {}
        self.sliding_windows: Dict[str, SlidingWindowRateLimiter] = {}

    def get_token_bucket(self, name: str, rate: float, capacity: int) -> TokenBucketRateLimiter:
        """
        获取或创建令牌桶限流器

        Args:
            name: 限流器名称
            rate: 令牌生成速率
            capacity: 桶容量

        Returns:
            令牌桶限流器
        """
        if name not in self.token_buckets:
            self.token_buckets[name] = TokenBucketRateLimiter(rate, capacity)
            app_logger.info(f"创建令牌桶限流器: {name}, 速率: {rate}/s, 容量: {capacity}")
        return self.token_buckets[name]

    def get_sliding_window(self, name: str, max_requests: int, window_seconds: int) -> SlidingWindowRateLimiter:
        """
        获取或创建滑动窗口限流器

        Args:
            name: 限流器名称
            max_requests: 最大请求数
            window_seconds: 窗口大小

        Returns:
            滑动窗口限流器
        """
        if name not in self.sliding_windows:
            self.sliding_windows[name] = SlidingWindowRateLimiter(max_requests, window_seconds)
            app_logger.info(f"创建滑动窗口限流器: {name}, 最大请求: {max_requests}, 窗口: {window_seconds}s")
        return self.sliding_windows[name]


# 全局限流管理器
rate_limiter_manager = RateLimiterManager()


def rate_limit(
    max_requests: int = 100,
    window_seconds: int = 60,
    key_func=None
):
    """
    限流装饰器

    Args:
        max_requests: 最大请求数
        window_seconds: 窗口大小
        key_func: 获取限流键的函数
    """
    def decorator(func):
        limiter = SlidingWindowRateLimiter(max_requests, window_seconds)

        def wrapper(*args, **kwargs):
            # 获取限流键
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = "default"

            if not limiter.is_allowed(key):
                app_logger.warning(f"限流触发: {func.__name__}, key={key}")
                raise Exception(f"请求过于频繁，请稍后再试")

            return func(*args, **kwargs)

        return wrapper
    return decorator
