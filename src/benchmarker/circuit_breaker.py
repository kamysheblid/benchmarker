"""Async-aware circuit breaker for benchmarker."""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger("benchmarker.circuit_breaker")


class CircuitBreaker:
    """Simple circuit breaker with closed / open / half_open states."""

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._last_failure_time: float = 0.0
        self._state = self.STATE_CLOSED
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        if self._state == self.STATE_OPEN:
            if time.monotonic() - self._last_failure_time > self.recovery_timeout:
                self._state = self.STATE_HALF_OPEN
                logger.info("Circuit breaker transitioned to half_open")
        return self._state

    async def call(self, func, *args, **kwargs):
        """Execute *func* with circuit-breaker protection.

        Args:
            func: Callable (sync or async) to execute.
            *args, **kwargs: Forwarded to *func*.

        Returns:
            Result of *func*.

        Raises:
            RuntimeError: When the breaker is open.
            Exception: Re-raised from *func* on failure.
        """
        async with self._lock:
            current_state = self.state
            if current_state == self.STATE_OPEN:
                logger.warning("Circuit breaker is open; rejecting call")
                raise RuntimeError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            async with self._lock:
                self._failures += 1
                self._last_failure_time = time.monotonic()
                logger.debug(
                    "Circuit breaker recorded failure %d/%d: %s",
                    self._failures,
                    self.failure_threshold,
                    exc,
                )
                if self._failures >= self.failure_threshold:
                    self._state = self.STATE_OPEN
                    logger.error(
                        "Circuit breaker tripped to open after %d failures",
                        self._failures,
                    )
            raise

        async with self._lock:
            if self._state == self.STATE_HALF_OPEN:
                self._state = self.STATE_CLOSED
                self._failures = 0
                logger.info("Circuit breaker recovered; transitioned to closed")

        return result

    def reset(self) -> None:
        """Force the breaker back to closed."""
        self._failures = 0
        self._state = self.STATE_CLOSED
        self._last_failure_time = 0.0
        logger.info("Circuit breaker manually reset to closed")
