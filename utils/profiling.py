import cProfile
import pstats
from collections import deque
from contextlib import contextmanager
from functools import wraps
from io import StringIO
from typing import Optional, Callable
import time
import logging
import logging.handlers

import psutil


class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = 0
        self.end_time = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        self.logger.debug(f"{self.name} took {duration:.4f} seconds")

    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        return self.end_time - self.start_time

def run_with_profiling(main_func, **main_kwargs):
    """Run with performance profiling enabled"""
    profiler = cProfile.Profile()
    profiler.enable()

    # We call the main_func that gets passed in, along with its arguments!
    exit_code = main_func(**main_kwargs)

    profiler.disable()

    # Print profiling results
    stats_stream = StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    print("\n" + "=" * 80)
    print("PERFORMANCE PROFILING RESULTS")
    print("=" * 80)
    print(stats_stream.getvalue())

    return exit_code


def run_with_memory_profiling(main_func, **main_kwargs):
    """Run with memory profiling enabled"""
    try:
        from memory_profiler import profile

        # Wrap the passed-in main function with memory profiler
        profiled_main = profile(main_func)
        return profiled_main(**main_kwargs)

    except ImportError:
        print("memory_profiler not installed. Install with: pip install memory-profiler")
        # Fallback to normal execution if the profiler is missing
        return main_func(**main_kwargs)

class FPSCounter:
    """Efficient FPS counter with deque-based moving average"""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.perf_counter()
        self._sum = 0.0

    def update(self) -> float:
        """Update FPS counter and return current FPS"""
        current_time = time.perf_counter()
        frame_time = current_time - self.last_time
        self.last_time = current_time

        if len(self.frame_times) == self.window_size:
            self._sum -= self.frame_times[0]

        self.frame_times.append(frame_time)
        self._sum += frame_time

        if self._sum > 0:
            return len(self.frame_times) / self._sum
        return 0.0

    def get_fps(self) -> float:
        """Get current FPS without updating"""
        if self._sum > 0:
            return len(self.frame_times) / self._sum
        return 0.0

class MovingAverage:
    """Efficient moving average calculator using deque"""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.sum = 0.0

    def update(self, value: float) -> float:
        """Add new value and return current average"""
        if len(self.values) == self.window_size:
            self.sum -= self.values[0]  # Remove oldest from sum

        self.values.append(value)
        self.sum += value
        return self.sum / len(self.values)

    def get_average(self) -> float:
        """Get current average"""
        return self.sum / len(self.values) if self.values else 0.0

    def reset(self):
        """Reset the moving average"""
        self.values.clear()
        self.sum = 0.0

# ======================== PERFORMANCE UTILITIES ========================

def performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor function performance

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with performance monitoring
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f"{func.__module__}.{func.__name__}")

        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

        try:
            result = func(*args, **kwargs)

            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            logger.debug(
                f"Performance: {func.__name__} took {duration:.4f}s, "
                f"memory delta: {memory_delta:+.2f}MB"
            )

            return result

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.error(f"Function {func.__name__} failed after {duration:.4f}s: {e}")
            raise

    return wrapper


@contextmanager
def memory_monitor(operation_name: str = "Operation"):
    """
    Context manager to monitor memory usage

    Args:
        operation_name: Name of the operation being monitored
    """
    logger = logging.getLogger(__name__)
    process = psutil.Process()

    start_memory = process.memory_info().rss / (1024 * 1024)  # MB
    peak_memory = start_memory

    try:
        yield
    finally:
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        peak_memory = max(peak_memory, end_memory)

        memory_delta = end_memory - start_memory
        peak_delta = peak_memory - start_memory

        logger.debug(
            f"Memory usage for {operation_name}: "
            f"delta={memory_delta:+.2f}MB, peak_delta={peak_delta:+.2f}MB"
        )
