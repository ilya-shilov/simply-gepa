"""Base LLM client interface for GEPA."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

CHARS_PER_TOKEN_ESTIMATE = 4


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients used in GEPA optimization."""

    @abstractmethod
    async def achat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        """Send asynchronous chat completion request."""
        pass

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        """Send synchronous chat completion request."""
        pass

    def count_tokens(self, text: str) -> int:
        """Estimate token count using character-based approximation."""
        return len(text) // CHARS_PER_TOKEN_ESTIMATE
