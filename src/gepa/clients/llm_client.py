"""LLM client with retry logic and rate limiting."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI, OpenAI, RateLimitError

from ..config import Settings
from .base import BaseLLMClient

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0
BACKOFF_MULTIPLIER = 2.0
CHARS_PER_TOKEN_ESTIMATE = 4


class LLMClient(BaseLLMClient):
    """OpenAI API client with automatic retry and exponential backoff."""

    def __init__(self, settings: Settings):
        """Initialize LLM client with OpenAI credentials."""
        self.settings = settings
        client_kwargs: Dict[str, Any] = {"api_key": settings.api_key or "local"}
        if settings.base_url:
            client_kwargs["base_url"] = settings.base_url
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
        self.model = settings.model
        self.temperature = settings.temperature

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        """Send synchronous chat completion request with retry logic."""
        temp = temperature if temperature is not None else self.temperature

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        retry_delay = INITIAL_RETRY_DELAY

        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(**kwargs)
                latency = (time.time() - start_time) * 1000

                content = response.choices[0].message.content or ""
                tokens_used = response.usage.total_tokens if response.usage else 0

                logger.debug(
                    f"LLM response: {len(content)} chars, "
                    f"{tokens_used} tokens, {latency:.0f}ms"
                )

                return content

            except RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Rate limit hit, retrying in {retry_delay}s... "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= BACKOFF_MULTIPLIER
                else:
                    logger.error(f"Rate limit exceeded after {MAX_RETRIES} attempts")
                    raise

            except Exception as e:
                logger.error(f"LLM request failed: {e}")
                raise

        raise RuntimeError("Unexpected end of retry loop")

    async def achat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        """Send asynchronous chat completion request with retry logic."""
        temp = temperature if temperature is not None else self.temperature

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        retry_delay = INITIAL_RETRY_DELAY

        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                response = await self.async_client.chat.completions.create(**kwargs)
                latency = (time.time() - start_time) * 1000

                content = response.choices[0].message.content or ""
                tokens_used = response.usage.total_tokens if response.usage else 0

                logger.debug(
                    f"LLM async response: {len(content)} chars, "
                    f"{tokens_used} tokens, {latency:.0f}ms"
                )

                return content

            except RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Rate limit hit, retrying in {retry_delay}s... "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= BACKOFF_MULTIPLIER
                else:
                    logger.error(f"Rate limit exceeded after {MAX_RETRIES} attempts")
                    raise

            except Exception as e:
                logger.error(f"Async LLM request failed: {e}")
                raise

        raise RuntimeError("Unexpected end of retry loop")

    def count_tokens(self, text: str) -> int:
        """Estimate token count using character-based approximation."""
        return len(text) // CHARS_PER_TOKEN_ESTIMATE

    def validate_json_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate JSON response from LLM."""
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {response[:200]}")
            raise ValueError(f"LLM returned invalid JSON: {e}")
