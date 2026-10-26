"""Process-wide and per-key token buckets."""

from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        if rate < 0 or capacity <= 0:
            raise ValueError("rate must be >= 0 and capacity must be > 0")
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()

    def _refill(self, now: float) -> None:
        elapsed = now - self.updated
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.updated = now

    def available(self, now: float) -> float:
        self._refill(now)
        return self.tokens

    def take(self) -> None:
        self.tokens -= 1

    def retry_after(self, now: float) -> float:
        self._refill(now)
        if self.tokens >= 1:
            return 0.0
        if self.rate <= 0:
            return 60.0
        return (1.0 - self.tokens) / self.rate


class RateLimiter:
    """A request is allowed only when both the global bucket and the key bucket have a token."""

    def __init__(self, rate: float, burst: float, max_keys: int = 2048):
        self.rate = rate
        self.burst = burst
        self.max_keys = max_keys
        self._global = TokenBucket(rate, burst)
        self._keys: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._keys.get(key)
            if bucket is None:
                if len(self._keys) >= self.max_keys:
                    self._keys.clear()
                bucket = TokenBucket(self.rate, self.burst)
                self._keys[key] = bucket
            if bucket.available(now) < 1 or self._global.available(now) < 1:
                return False
            bucket.take()
            self._global.take()
            return True

    def retry_after(self, key: str) -> float:
        now = time.monotonic()
        with self._lock:
            bucket = self._keys.get(key)
            per_key = bucket.retry_after(now) if bucket is not None else 0.0
            return max(per_key, self._global.retry_after(now), 0.05)
