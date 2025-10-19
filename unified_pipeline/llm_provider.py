"""
LLM Provider Abstraction Layer
Supports both Ollama (local) and OpenAI API
"""

import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(
            self,
            prompt: str,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from a prompt"""
        pass

    @abstractmethod
    def chat(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using chat with system/user/assistant messages"""
        pass

    @abstractmethod
    def generate_json(
            self,
            prompt: str,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate structured JSON output"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(
            self,
            prompt: str,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""

    def chat(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using chat with system/user/assistant messages"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI chat error: {e}")
            return ""

    def generate_json(
            self,
            prompt: str,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate structured JSON output from OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            print(f"OpenAI JSON generation error: {e}")
            return {}


class OllamaProvider(LLMProvider):
    """Ollama local provider"""

    def __init__(self, model: str = "llama3.1:8b"):
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package not installed. Run: pip install ollama")

        self.model = model
        self.client = ollama

        # Test connection
        try:
            self.client.list()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama. Is it running? Error: {e}")

    def generate(
            self,
            prompt: str,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from Ollama"""
        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens:
                options["num_predict"] = max_tokens

            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options=options
            )
            return response['response'].strip()
        except Exception as e:
            print(f"Ollama generation error: {e}")
            return ""

    def chat(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            stop: Optional[List[str]] = None
    ) -> str:
        """Generate text using chat with system/user/assistant messages"""
        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens:
                options["num_predict"] = max_tokens

            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Ollama chat error: {e}")
            return ""

    def generate_json(
            self,
            prompt: str,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate structured JSON output from Ollama"""
        # Add explicit JSON formatting instruction
        json_prompt = f"{prompt}\n\nRespond ONLY with valid JSON. No other text."

        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens:
                options["num_predict"] = max_tokens

            response = self.client.generate(
                model=self.model,
                prompt=json_prompt,
                format="json",
                options=options
            )

            content = response['response'].strip()
            # Try to extract JSON if wrapped in markdown
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except Exception as e:
            print(f"Ollama JSON generation error: {e}")
            # Return empty structure as fallback
            return {}


def create_llm_provider(
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None
) -> LLMProvider:
    """Factory function to create LLM provider"""

    if provider.lower() == "openai":
        model = model or "gpt-4o-mini"
        return OpenAIProvider(model=model, api_key=api_key)

    elif provider.lower() == "ollama":
        model = model or "llama3.1:8b"
        return OllamaProvider(model=model)

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'ollama'")
