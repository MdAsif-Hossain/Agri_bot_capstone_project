"""In-memory token-bucket rate limiter, keyed by client IP.

Single-node design — sufficient for AgriBot's offline LAN deployment. For
multi-replica deployments, swap the in-process dict for Redis with the same
bucket math.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketRateLimiter:
    """Steady rate of `per_minute` requests per key, with `burst` overflow.

    `acquire(key)` returns (allowed, retry_after_seconds). retry_after is 0
    when allowed, or a small float estimating when the next token will arrive.
    """

    def __init__(self, per_minute: int = 30, burst: int = 10):
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        if burst < 0:
            raise ValueError("burst must be non-negative")
        self.rate_per_sec = per_minute / 60.0
        self.capacity = float(per_minute + burst)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self.capacity, last_refill=now)
                self._buckets[key] = bucket
            else:
                elapsed = now - bucket.last_refill
                if elapsed > 0:
                    bucket.tokens = min(
                        self.capacity, bucket.tokens + elapsed * self.rate_per_sec
                    )
                    bucket.last_refill = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, 0.0

            deficit = cost - bucket.tokens
            retry_after = deficit / self.rate_per_sec if self.rate_per_sec else 60.0
            return False, retry_after

    def reset(self, key: Optional[str] = None) -> None:
        """Clear bucket state for one key (or all keys when None). Test helper."""
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)
