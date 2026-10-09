"""
LLM Wrapper - Interface for generating responses.

Uses OpenAI-compatible API (GAPGPT, OpenRouter, etc.)
"""

import logging
from typing import Optional

from openai import OpenAI
from settings import RAGConfig

logger = logging.getLogger(__name__)


class LLMWrapper:
    """
    Wrapper for LLM API calls.
    
    Features:
    - Configurable model and base URL
    - Temperature and max tokens control
    - System prompt support
    - Runtime config updates for UI
    """
    
    def __init__(
        self,
        config: RAGConfig,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize LLM wrapper.
        
        Args:
            config: RAGConfig instance with all settings
            model: Override model name (optional)
            base_url: Override API base URL (optional)
            api_key: Override API key (optional)
            temperature: Override temperature (optional)
            max_tokens: Override max tokens (optional)
            system_prompt: Override system prompt (optional)
        """
        self.config = config
        
        # Use overrides or fall back to config
        self.model = model or config.llm_model
        self.base_url = base_url or config.llm_base_url
        self.api_key = api_key or config.llm_api_key
        self.temperature = temperature if temperature is not None else config.temperature
        self.max_tokens = max_tokens if max_tokens is not None else config.max_tokens
        self.system_prompt = system_prompt or config.system_prompt
        
        # Initialize client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        logger.info(f"LLM initialized with model: {self.model}")
        logger.info(f"Temperature: {self.temperature}, Max tokens: {self.max_tokens}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The full prompt including context and question
            
        Returns:
            LLM response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            result = response.choices[0].message.content.strip()
            logger.debug(f"LLM response: {result[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "متأسفانه خطایی در پردازش پاسخ رخ داد. لطفاً دوباره تلاش کنید."
    
    def generate_stream(self, prompt: str):
        """
        Generate a streaming response from the LLM.
        
        Args:
            prompt: The full prompt including context and question
            
        Yields:
            Chunks of the answer
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield "متأسفانه خطایی در پردازش پاسخ رخ داد."
    
    def update_config(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Update LLM parameters at runtime (for UI).
        
        Args:
            temperature: New temperature value (0-1)
            max_tokens: New max tokens value
            system_prompt: New system prompt
        """
        if temperature is not None:
            if 0 <= temperature <= 1:
                self.temperature = temperature
                logger.info(f"Updated temperature to {temperature}")
            else:
                logger.warning(f"Invalid temperature: {temperature}. Must be 0-1.")
        
        if max_tokens is not None:
            if max_tokens > 0:
                self.max_tokens = max_tokens
                logger.info(f"Updated max_tokens to {max_tokens}")
        
        if system_prompt is not None:
            self.system_prompt = system_prompt
            logger.info("Updated system prompt")
    
    def health_check(self) -> bool:
        """
        Check if LLM API is reachable.
        
        Returns:
            True if API responds, False otherwise
        """
        try:
            # Simple test call with minimal tokens
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                temperature=0.0,
            )
            return bool(response.choices)
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False