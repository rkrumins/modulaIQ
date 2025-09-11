"""Error handling utilities for the supervisor system."""

import logging
from typing import Dict, Any, Optional, Callable, Type, Union
from functools import wraps
import asyncio
import time

logger = logging.getLogger(__name__)


class SupervisorError(Exception):
    """Base exception for supervisor-related errors."""
    pass


class LLMError(SupervisorError):
    """Exception raised when LLM operations fail."""
    pass


class AgentError(SupervisorError):
    """Exception raised when agent operations fail."""
    pass


class QueryAnalysisError(SupervisorError):
    """Exception raised when query analysis fails."""
    pass


class ReActError(SupervisorError):
    """Exception raised when ReAct operations fail."""
    pass


class ErrorHandler:
    """Centralized error handling for the supervisor system."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_counts: Dict[str, int] = {}
    
    def handle_error(
        self, 
        error: Exception, 
        context: str = "", 
        operation: str = "",
        should_retry: bool = True
    ) -> Dict[str, Any]:
        """
        Handle an error and return error information.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            operation: The operation that failed
            should_retry: Whether this error should be retried
            
        Returns:
            Dictionary with error information
        """
        error_key = f"{operation}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "operation": operation,
            "error_count": self.error_counts[error_key],
            "should_retry": should_retry and self.error_counts[error_key] < self.max_retries,
            "timestamp": time.time()
        }
        
        # Log the error
        log_level = logging.ERROR if self.error_counts[error_key] == 1 else logging.WARNING
        logger.log(
            log_level, 
            f"Error in {operation} ({context}): {error_info['error_message']} "
            f"(count: {self.error_counts[error_key]})"
        )
        
        return error_info
    
    def should_retry(self, error: Exception, operation: str) -> bool:
        """Check if an operation should be retried based on error type and count."""
        error_key = f"{operation}:{type(error).__name__}"
        error_count = self.error_counts.get(error_key, 0)
        
        # Don't retry if we've exceeded max retries
        if error_count >= self.max_retries:
            return False
        
        # Don't retry certain types of errors
        non_retryable_errors = (ValueError, TypeError, KeyError, AttributeError)
        if isinstance(error, non_retryable_errors):
            return False
        
        return True
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "unique_errors": len(self.error_counts)
        }
    
    def reset_error_counts(self) -> None:
        """Reset error counts."""
        self.error_counts.clear()
        logger.info("Error counts reset")


def with_error_handling(
    error_handler: ErrorHandler,
    context: str = "",
    operation: str = "",
    fallback_value: Any = None,
    raise_on_failure: bool = False
):
    """
    Decorator for adding error handling to functions.
    
    Args:
        error_handler: ErrorHandler instance
        context: Context for error logging
        operation: Operation name for error tracking
        fallback_value: Value to return on error
        raise_on_failure: Whether to raise the error after handling
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_info = error_handler.handle_error(e, context, operation)
                    
                    if raise_on_failure:
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed, returning fallback value")
                    return fallback_value
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_info = error_handler.handle_error(e, context, operation)
                    
                    if raise_on_failure:
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed, returning fallback value")
                    return fallback_value
            
            return sync_wrapper
    
    return decorator


async def retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    **kwargs
) -> Any:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        backoff_factor: Factor to multiply delay by on each retry
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Result of the function call
        
    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    delay = base_delay
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt == max_retries:
                logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                break
            
            logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
            logger.info(f"Retrying in {delay:.2f} seconds...")
            
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
    
    raise last_exception


class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN - operation blocked")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = asyncio.run(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                logger.info("Circuit breaker transitioning to CLOSED")
            
            self.failure_count = 0
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout
        }
