"""
缓存模块
提供 LRU 缓存和 TTL 缓存功能
"""

import time
from typing import Any, Optional, Dict
from functools import lru_cache
from collections import OrderedDict
import hashlib
import json

from app.core.logger import app_logger


class LRUCache:
    """
    LRU 缓存实现

    支持最大容量限制和 TTL 过期
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        初始化 LRU 缓存

        Args:
            max_size: 最大缓存条目数
            ttl: 缓存过期时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，如果不存在或已过期返回 None
        """
        if key not in self.cache:
            return None

        # 检查是否过期
        if self._is_expired(key):
            self.delete(key)
            return None

        # 移到最前（最近使用）
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: Any):
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
        """
        if key in self.cache:
            # 更新现有值
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            # 删除最久未使用的
            oldest_key = next(iter(self.cache))
            self.delete(oldest_key)

        self.cache[key] = value
        self.timestamps[key] = time.time()

    def delete(self, key: str):
        """
        删除缓存

        Args:
            key: 缓存键
        """
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.timestamps.clear()

    def _is_expired(self, key: str) -> bool:
        """检查是否过期"""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl

    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)


class CacheManager:
    """
    缓存管理器

    提供多个命名缓存实例
    """

    def __init__(self):
        self.caches: Dict[str, LRUCache] = {}

    def get_cache(self, name: str, max_size: int = 1000, ttl: int = 3600) -> LRUCache:
        """
        获取或创建缓存实例

        Args:
            name: 缓存名称
            max_size: 最大容量
            ttl: 过期时间

        Returns:
            缓存实例
        """
        if name not in self.caches:
            self.caches[name] = LRUCache(max_size=max_size, ttl=ttl)
            app_logger.info(f"创建缓存: {name}, 容量: {max_size}, TTL: {ttl}s")
        return self.caches[name]

    def clear_all(self):
        """清空所有缓存"""
        for cache in self.caches.values():
            cache.clear()
        app_logger.info("清空所有缓存")


def generate_cache_key(*args, **kwargs) -> str:
    """
    生成缓存键

    Args:
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        缓存键字符串
    """
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
    key_str = "|".join(key_parts)

    # 使用 MD5 生成固定长度的键
    return hashlib.md5(key_str.encode()).hexdigest()


# 全局缓存管理器
cache_manager = CacheManager()


def cached(cache_name: str, max_size: int = 1000, ttl: int = 3600):
    """
    缓存装饰器

    Args:
        cache_name: 缓存名称
        max_size: 最大容量
        ttl: 过期时间
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = cache_manager.get_cache(cache_name, max_size, ttl)
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)

            # 尝试从缓存获取
            result = cache.get(cache_key)
            if result is not None:
                app_logger.debug(f"缓存命中: {func.__name__}")
                return result

            # 执行函数
            result = func(*args, **kwargs)

            # 存入缓存
            cache.set(cache_key, result)
            app_logger.debug(f"缓存设置: {func.__name__}")

            return result
        return wrapper
    return decorator
