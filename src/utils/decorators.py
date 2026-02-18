"""Instrumentation decorators for logging and timing."""

from __future__ import annotations

import time
import functools
import logging
from typing import Any, Callable, Optional, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def _resolve_logger(args: tuple[Any, ...]) -> Optional[logging.Logger]:
    """
    Retrieve a logger from the first positional argument if available.

    Args:
        args: Positional arguments passed to the wrapped callable.

    Returns:
        A ``logging.Logger`` instance if present, otherwise ``None``.
    """
    if not args:
        return None

    candidate = args[0]
    return getattr(candidate, "logger", None)


def log_call(func: F) -> F:
    """
    Log function entry and exit using ``self.logger`` when available.

    Args:
        func: Callable to wrap.

    Returns:
        Wrapped callable with identical behaviour.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = _resolve_logger(args)
        if logger:
            logger.debug("[CALL] → %s()", func.__name__)
        result = func(*args, **kwargs)
        if logger:
            logger.debug("[CALL] ← %s()", func.__name__)
        return result

    return cast(F, wrapper)


def timer(func: F) -> F:
    """
    Measure execution time and log duration in milliseconds when a logger is available.

    Args:
        func: Callable to wrap.

    Returns:
        Wrapped callable preserving original behaviour.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = _resolve_logger(args)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if logger:
            logger.debug("[TIMER] %s took %.2f ms", func.__name__, elapsed_ms)
        return result

    return cast(F, wrapper)