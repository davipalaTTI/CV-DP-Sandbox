import logging
import threading
import time
from functools import wraps
from typing import Any, Optional, List, Callable


# ======================== THREADING UTILITIES ========================

class ThreadSafeCounter:
    """Thread-safe counter with lock"""

    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value"""
        with self._lock:
            self._value += amount
            return self._value

    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and return new value"""
        with self._lock:
            self._value -= amount
            return self._value

    def set(self, value: int) -> int:
        """Set counter value and return new value"""
        with self._lock:
            self._value = value
            return self._value

    def get(self) -> int:
        """Get current counter value"""
        with self._lock:
            return self._value

    def reset(self) -> int:
        """Reset counter to zero and return previous value"""
        with self._lock:
            old_value = self._value
            self._value = 0
            return old_value


class RateLimiter:
    """Rate limiter using token bucket algorithm"""

    def __init__(self, max_calls: int, time_window: float):
        """
        Initialize rate limiter

        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self._lock = threading.Lock()

    def is_allowed(self) -> bool:
        """Check if call is allowed under rate limit"""
        with self._lock:
            now = time.time()

            # Remove old calls outside time window
            self.calls = [call_time for call_time in self.calls
                          if now - call_time < self.time_window]

            # Check if under limit
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True

            return False

    def wait_if_needed(self) -> float:
        """Wait if necessary to respect rate limit, return wait time"""
        with self._lock:
            now = time.time()

            # Remove old calls
            self.calls = [call_time for call_time in self.calls
                          if now - call_time < self.time_window]

            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return 0.0

            # Calculate wait time
            oldest_call = min(self.calls)
            wait_time = self.time_window - (now - oldest_call)

            if wait_time > 0:
                time.sleep(wait_time)
                return wait_time

            return 0.0


# ======================== DATA STRUCTURES ========================

class CircularBuffer:
    """Fixed-size circular buffer with thread safety"""

    def __init__(self, capacity: int):
        """
        Initialize circular buffer

        Args:
            capacity: Maximum number of items
        """
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.size = 0
        self._lock = threading.Lock()

    def put(self, item: Any) -> Optional[Any]:
        """
        Add item to buffer

        Returns:
            Evicted item if buffer was full, None otherwise
        """
        with self._lock:
            evicted = None
            if self.size == self.capacity:
                evicted = self.buffer[self.head]

            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.capacity

            if self.size < self.capacity:
                self.size += 1

            return evicted

    def get_all(self) -> List[Any]:
        """Get all items in chronological order"""
        with self._lock:
            if self.size == 0:
                return []

            if self.size < self.capacity:
                # Buffer not full yet
                return [self.buffer[i] for i in range(self.size)]
            else:
                # Buffer is full, need to handle wraparound
                items = []
                for i in range(self.capacity):
                    index = (self.head + i) % self.capacity
                    items.append(self.buffer[index])
                return items

    def get_latest(self, n: int) -> List[Any]:
        """Get n most recent items"""
        with self._lock:
            if self.size == 0:
                return []

            n = min(n, self.size)
            items = []

            for i in range(n):
                index = (self.head - 1 - i) % self.capacity
                items.append(self.buffer[index])

            return items

    def clear(self):
        """Clear the buffer"""
        with self._lock:
            self.buffer = [None] * self.capacity
            self.head = 0
            self.size = 0

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self._lock:
            return self.size == 0

    def is_full(self) -> bool:
        """Check if buffer is full"""
        with self._lock:
            return self.size == self.capacity


# ======================== ERROR HANDLING ========================

class ConfigurationError(Exception):
    """Exception for configuration-related errors"""
    pass


class ProcessingError(Exception):
    """Exception for processing-related errors"""
    pass


class ValidationError(Exception):
    """Exception for validation errors"""
    pass


def handle_exception(func: Callable) -> Callable:
    """
    Decorator for comprehensive exception handling

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with exception handling
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f"{func.__module__}.{func.__name__}")

        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.info(f"Function {func.__name__} interrupted by user")
            raise
        except ConfigurationError as e:
            logger.error(f"Configuration error in {func.__name__}: {e}")
            raise
        except ProcessingError as e:
            logger.error(f"Processing error in {func.__name__}: {e}")
            raise
        except ValidationError as e:
            logger.error(f"Validation error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise ProcessingError(f"Unexpected error in {func.__name__}: {e}") from e

    return wrapper