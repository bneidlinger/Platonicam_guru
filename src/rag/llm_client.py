"""
Ollama LLM Client for chat completions.

Handles communication with local Ollama instance for response generation.
"""
from typing import Generator, Optional

import ollama

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import Settings


class OllamaLLM:
    """
    LLM client for generating responses using Ollama.
    """

    def __init__(
        self,
        model: str = Settings.CHAT_MODEL,
        host: str = Settings.OLLAMA_HOST,
        temperature: float = Settings.TEMPERATURE,
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """Lazy-load Ollama client."""
        if self._client is None:
            self._client = ollama.Client(host=self.host)
        return self._client

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response for a single prompt.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            temperature: Override default temperature.

        Returns:
            Generated response text.
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature or self.temperature,
            },
        )

        return response["message"]["content"]

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response for a conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.

        Returns:
            Generated response text.
        """
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature or self.temperature,
            },
        )

        return response["message"]["content"]

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a response token by token.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            temperature: Override default temperature.

        Yields:
            Response tokens as they're generated.
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                "temperature": temperature or self.temperature,
            },
        )

        for chunk in stream:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]

    def check_model_available(self) -> bool:
        """
        Check if the chat model is available in Ollama.
        """
        try:
            models = self.client.list()
            model_names = [m["name"] for m in models.get("models", [])]
            return (
                self.model in model_names or
                f"{self.model}:latest" in model_names
            )
        except Exception as e:
            print(f"Error checking Ollama models: {e}")
            return False

    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        """
        try:
            return self.client.show(self.model)
        except Exception:
            return {}


if __name__ == "__main__":
    print("Testing OllamaLLM...")
    print("-" * 40)

    llm = OllamaLLM()

    print(f"Model: {llm.model}")
    print(f"Host: {llm.host}")
    print(f"Temperature: {llm.temperature}")

    if llm.check_model_available():
        print("Model is available!")

        # Test generation
        response = llm.generate(
            "What is PoE+ in one sentence?",
            system="You are a helpful assistant for security system engineers.",
        )
        print(f"\nTest response:\n{response}")
    else:
        print(f"Model '{llm.model}' not found!")
        print(f"Run: ollama pull {llm.model}")
