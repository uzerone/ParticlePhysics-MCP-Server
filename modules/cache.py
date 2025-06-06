import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: float
    size_bytes: int

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() > (self.created_at + self.ttl)

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at

    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStatistics:
    """Cache performance statistics."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0
    total_entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self),
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
        }


class MemoryCache:
    """In-memory LRU cache with TTL support and comprehensive statistics."""

    def __init__(self, max_size: int = None, default_ttl: float = None):
        self.max_size = max_size or config.cache.max_size
        self.default_ttl = default_ttl or config.cache.ttl_seconds

        # Use OrderedDict for LRU functionality
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStatistics(max_size=self.max_size)

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        if config.cache.enabled:
            self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task for expired entries."""

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # Cleanup every minute
                    await self._cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def _cleanup_expired(self):
        """Remove expired cache entries."""
        async with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                self._stats.evictions += 1

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _calculate_size(self, value: Any) -> int:
        """Estimate size of cached value in bytes."""
        try:
            # Simple size estimation using JSON serialization
            return len(json.dumps(value, default=str).encode("utf-8"))
        except:
            # Fallback to string representation
            return len(str(value).encode("utf-8"))

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from function arguments."""
        # Create a stable hash from arguments
        key_data = {
            "prefix": prefix,
            "args": args,
            "kwargs": sorted(kwargs.items()) if kwargs else {},
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not config.cache.enabled:
            return None

        async with self._lock:
            self._stats.total_requests += 1

            if key in self._cache:
                entry = self._cache[key]

                # Check if expired
                if entry.is_expired:
                    del self._cache[key]
                    self._stats.cache_misses += 1
                    self._stats.evictions += 1
                    return None

                # Touch entry (LRU update)
                entry.touch()
                # Move to end for LRU
                self._cache.move_to_end(key)

                self._stats.cache_hits += 1
                logger.debug(f"Cache hit for key: {key[:16]}...")
                return entry.value

            self._stats.cache_misses += 1
            logger.debug(f"Cache miss for key: {key[:16]}...")
            return None

    async def set(self, key: str, value: Any, ttl: float = None) -> bool:
        """Set value in cache."""
        if not config.cache.enabled:
            return False

        ttl = ttl or self.default_ttl
        size_bytes = self._calculate_size(value)

        async with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.current_size -= old_entry.size_bytes
                del self._cache[key]

            # Check if we need to evict entries to make space
            while len(self._cache) >= self.max_size:
                # Remove least recently used item
                oldest_key, oldest_entry = self._cache.popitem(last=False)
                self._stats.current_size -= oldest_entry.size_bytes
                self._stats.evictions += 1
                logger.debug(f"Evicted cache entry: {oldest_key[:16]}...")

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=1,
                ttl=ttl,
                size_bytes=size_bytes,
            )

            # Add to cache
            self._cache[key] = entry
            self._stats.current_size += size_bytes
            self._stats.total_entries = len(self._cache)

            logger.debug(f"Cached value for key: {key[:16]}... (TTL: {ttl}s)")
            return True

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._stats.current_size -= entry.size_bytes
                del self._cache[key]
                self._stats.total_entries = len(self._cache)
                return True
            return False

    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._stats.current_size = 0
            self._stats.total_entries = 0
            logger.info("Cache cleared")

    async def get_statistics(self) -> CacheStatistics:
        """Get cache statistics."""
        async with self._lock:
            stats_copy = CacheStatistics(
                total_requests=self._stats.total_requests,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                evictions=self._stats.evictions,
                current_size=self._stats.current_size,
                max_size=self._stats.max_size,
                total_entries=len(self._cache),
            )
            return stats_copy

    async def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        async with self._lock:
            entries_info = []
            for key, entry in list(self._cache.items())[:10]:  # Show top 10
                entries_info.append(
                    {
                        "key": key[:32] + "..." if len(key) > 32 else key,
                        "age_seconds": entry.age_seconds,
                        "access_count": entry.access_count,
                        "size_bytes": entry.size_bytes,
                        "expires_in": max(
                            0, (entry.created_at + entry.ttl) - time.time()
                        ),
                    }
                )

            return {
                "statistics": (await self.get_statistics()).to_dict(),
                "sample_entries": entries_info,
                "total_entries": len(self._cache),
                "enabled": config.cache.enabled,
                "max_size": self.max_size,
                "default_ttl": self.default_ttl,
            }

    def __del__(self):
        """Cleanup when cache is destroyed."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()


class CacheManager:
    """Main cache manager with multiple cache backends."""

    def __init__(self):
        self.memory_cache = MemoryCache()

        # Cache key prefixes for different data types
        self.KEY_PREFIXES = {
            "particle_search": "ps",
            "particle_properties": "pp",
            "mass_measurements": "mm",
            "lifetime_measurements": "lm",
            "width_measurements": "wm",
            "branching_fractions": "bf",
            "decay_products": "dp",
            "summary_values": "sv",
            "database_info": "db",
            "particle_list": "pl",
        }

    def _make_cache_key(self, cache_type: str, *args, **kwargs) -> str:
        """Generate cache key for specific data type."""
        prefix = self.KEY_PREFIXES.get(cache_type, "unknown")
        return self.memory_cache._generate_key(prefix, *args, **kwargs)

    async def get_cached_result(
        self, cache_type: str, *args, **kwargs
    ) -> Optional[Any]:
        """Get cached result for specific data type."""
        key = self._make_cache_key(cache_type, *args, **kwargs)
        return await self.memory_cache.get(key)

    async def cache_result(
        self, cache_type: str, result: Any, ttl: float = None, *args, **kwargs
    ) -> bool:
        """Cache result for specific data type."""
        key = self._make_cache_key(cache_type, *args, **kwargs)
        return await self.memory_cache.set(key, result, ttl)

    async def invalidate_cache_type(self, cache_type: str):
        """Invalidate all cache entries of a specific type."""
        prefix = self.KEY_PREFIXES.get(cache_type, "unknown")

        async with self.memory_cache._lock:
            keys_to_delete = []
            for key in self.memory_cache._cache.keys():
                if key.startswith(prefix):
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                await self.memory_cache.delete(key)

        logger.info(
            f"Invalidated {len(keys_to_delete)} cache entries for type: {cache_type}"
        )

    async def warm_cache(self, api_instance):
        """Pre-populate cache with frequently accessed data."""
        if not config.cache.enabled:
            return

        logger.info("Starting cache warming...")

        try:
            # Cache database info
            await self._warm_database_info(api_instance)

            # Cache common particles
            await self._warm_common_particles(api_instance)

            logger.info("Cache warming completed successfully")
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")

    async def _warm_database_info(self, api_instance):
        """Warm cache with database info."""
        try:
            # This would be implementation-specific based on your API
            # For now, just log the intent
            logger.debug("Warming database info cache...")
        except Exception as e:
            logger.debug(f"Failed to warm database info cache: {e}")

    async def _warm_common_particles(self, api_instance):
        """Warm cache with common particle data."""
        common_particles = [
            "electron",
            "muon",
            "tau-",
            "proton",
            "neutron",
            "pi+",
            "pi-",
            "pi0",
            "K+",
            "K-",
            "K0",
        ]

        for particle_name in common_particles:
            try:
                logger.debug(f"Warming cache for particle: {particle_name}")
                # Implementation would depend on your API structure
                # This is a placeholder for the actual caching logic
            except Exception as e:
                logger.debug(f"Failed to warm cache for {particle_name}: {e}")

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return await self.memory_cache.get_cache_info()


# Global cache manager instance
cache_manager = CacheManager()


def cached(cache_type: str, ttl: float = None, key_args: List[str] = None):
    """Decorator for caching function results."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not config.cache.enabled:
                return await func(*args, **kwargs)

            # Extract relevant arguments for cache key
            cache_args = []
            cache_kwargs = {}

            if key_args:
                # Use specific arguments for cache key
                for i, arg_name in enumerate(key_args):
                    if i < len(args):
                        cache_args.append(args[i])
                    elif arg_name in kwargs:
                        cache_kwargs[arg_name] = kwargs[arg_name]
            else:
                # Use all arguments
                cache_args = args
                cache_kwargs = kwargs

            # Try to get from cache
            cached_result = await cache_manager.get_cached_result(
                cache_type, *cache_args, **cache_kwargs
            )

            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}, executing function")
            result = await func(*args, **kwargs)

            # Cache the result
            await cache_manager.cache_result(
                cache_type, result, ttl, *cache_args, **cache_kwargs
            )

            return result

        return wrapper

    return decorator


async def get_cache_health() -> Dict[str, Any]:
    """Get cache health information."""
    stats = await cache_manager.get_cache_statistics()

    # Determine health status
    hit_rate = stats["statistics"]["hit_rate"]
    if hit_rate >= 0.8:
        health_status = "excellent"
    elif hit_rate >= 0.6:
        health_status = "good"
    elif hit_rate >= 0.4:
        health_status = "fair"
    else:
        health_status = "poor"

    return {
        "status": health_status,
        "hit_rate": hit_rate,
        "total_requests": stats["statistics"]["total_requests"],
        "cache_size": stats["statistics"]["total_entries"],
        "max_size": stats["max_size"],
        "enabled": stats["enabled"],
        "recommendations": _get_cache_recommendations(stats),
    }


def _get_cache_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Generate cache optimization recommendations."""
    recommendations = []

    hit_rate = stats["statistics"]["hit_rate"]
    total_entries = stats["statistics"]["total_entries"]
    max_size = stats["max_size"]

    if hit_rate < 0.5:
        recommendations.append(
            "Consider increasing cache TTL or warming cache with common queries"
        )

    if total_entries >= max_size * 0.9:
        recommendations.append("Cache is near capacity - consider increasing max_size")

    if stats["statistics"]["evictions"] > stats["statistics"]["cache_hits"] * 0.1:
        recommendations.append(
            "High eviction rate - consider increasing cache size or optimizing TTL"
        )

    if not recommendations:
        recommendations.append("Cache performance is optimal")

    return recommendations
