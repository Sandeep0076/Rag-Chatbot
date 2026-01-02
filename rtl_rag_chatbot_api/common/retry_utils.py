"""
Retry utilities for handling transient API failures with exponential backoff.
"""
import logging
import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator for retrying function calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    error_str = str(e).lower()

                    # Check if this is an overload error
                    is_overload = (
                        "overloaded" in error_str
                        or "overloaded_error" in error_str
                        or "503" in error_str
                        or "rate limit" in error_str
                        or "too many requests" in error_str
                    )

                    if not is_overload or attempt == max_retries:
                        # Not an overload error or last attempt, raise immediately
                        logger.error(
                            f"{func.__name__} failed on attempt {attempt + 1}: "
                            f"{str(e)}"
                        )
                        raise

                    # Calculate delay for next retry
                    current_delay = min(delay, max_delay)

                    logger.warning(
                        f"{func.__name__} overloaded (attempt {attempt + 1}/"
                        f"{max_retries + 1}). Retrying in {current_delay:.1f}s..."
                    )

                    time.sleep(current_delay)
                    delay *= exponential_base

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__name__} failed after {max_retries} retries")

        return wrapper

    return decorator


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable (transient API failures).

    Args:
        error: Exception to check

    Returns:
        bool: True if error is retryable, False otherwise
    """
    error_str = str(error).lower()
    retryable_patterns = [
        "overloaded",
        "overloaded_error",
        "503",
        "service unavailable",
        "rate limit",
        "too many requests",
        "temporarily unavailable",
        "timeout",
    ]

    return any(pattern in error_str for pattern in retryable_patterns)
