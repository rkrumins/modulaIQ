"""LLM service for handling all LLM interactions with caching and rate limiting."""

import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional, Union
import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel

from core.config import Config
from supervisor.error_handler import ErrorHandler, LLMError, retry_with_backoff

logger = logging.getLogger(__name__)


class LLMService:
    """Service for managing LLM interactions with caching and rate limiting."""
    
    def __init__(self, llm: BaseLanguageModel, config: Config):
        self.llm = llm
        self.config = config
        
        # Rate limiting configuration
        self.last_llm_call_time = 0
        self.min_llm_call_interval = getattr(
            config.supervisor, 'rate_limiting', {}
        ).get('min_llm_call_interval', 1.0)
        self.max_requests_per_minute = getattr(
            config.supervisor, 'rate_limiting', {}
        ).get('max_requests_per_minute', 30)
        
        # Response cache
        self.response_cache: Dict[str, str] = {}
        self.cache_max_size = 200
        
        # Error handling
        self.error_handler = ErrorHandler(max_retries=3, retry_delay=1.0)
        
    def _get_cache_key(self, messages: List[BaseMessage], call_type: str) -> str:
        """Generate a cache key for LLM calls."""
        content = "".join([msg.content for msg in messages if hasattr(msg, 'content')])
        return hashlib.md5(f"{call_type}:{content}".encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available."""
        return self.response_cache.get(cache_key)
    
    def _cache_response(self, cache_key: str, response: str) -> None:
        """Cache a response."""
        if len(self.response_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        self.response_cache[cache_key] = response
    
    async def _apply_rate_limiting(self) -> None:
        """Apply rate limiting to LLM calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_llm_call_time
        
        if time_since_last_call < self.min_llm_call_interval:
            sleep_time = self.min_llm_call_interval - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
    
    async def call_llm(
        self, 
        messages: List[BaseMessage], 
        call_type: str = "general",
        timeout: float = 30.0
    ) -> str:
        """
        Call LLM with caching, rate limiting, and error handling.
        
        Args:
            messages: List of messages to send to LLM
            call_type: Type of call for caching purposes
            timeout: Timeout in seconds
            
        Returns:
            LLM response content
            
        Raises:
            Exception: If LLM call fails
        """
        # Check cache first
        cache_key = self._get_cache_key(messages, call_type)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            logger.debug(f"Using cached response for {call_type} LLM call")
            return cached_response
        
        # Apply rate limiting
        await self._apply_rate_limiting()
        
        logger.debug(f"Making {call_type} LLM call")
        start_time = time.time()
        
        try:
            # Use retry with backoff for LLM calls
            response = await retry_with_backoff(
                self.llm.ainvoke,
                messages,
                max_retries=2,
                base_delay=1.0,
                max_delay=5.0
            )
            
            call_duration = time.time() - start_time
            self.last_llm_call_time = time.time()
            
            # Cache the response
            self._cache_response(cache_key, response.content)
            
            logger.debug(f"LLM {call_type} call completed in {call_duration:.2f}s")
            return response.content
            
        except asyncio.TimeoutError as e:
            call_duration = time.time() - start_time
            error_info = self.error_handler.handle_error(
                e, f"LLM {call_type} call", "llm_call_timeout"
            )
            raise LLMError(f"LLM call timed out after {timeout}s")
        except Exception as e:
            call_duration = time.time() - start_time
            error_info = self.error_handler.handle_error(
                e, f"LLM {call_type} call", "llm_call_failed"
            )
            raise LLMError(f"LLM call failed: {str(e)}")
    
    async def call_llm_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        call_type: str = "general",
        timeout: float = 30.0
    ) -> str:
        """
        Call LLM with system and user prompts.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            call_type: Type of call for caching purposes
            timeout: Timeout in seconds
            
        Returns:
            LLM response content
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        return await self.call_llm(messages, call_type, timeout)
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self.response_cache.clear()
        logger.info("LLM response cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.response_cache),
            "cache_max_size": self.cache_max_size,
            "cache_utilization": len(self.response_cache) / self.cache_max_size
        }
