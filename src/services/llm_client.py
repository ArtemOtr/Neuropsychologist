import logging
from collections.abc import AsyncGenerator
from typing import Any

from langfuse.openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from src.langfuse_client import langfuse_client
from src.schemas.openai_compat import OpenAIMessage
from src.settings import settings


logger = logging.getLogger(__name__)


def _strip_model_prefix(model: str) -> str:
    return model.removeprefix("openai/")


class LLMClient:
    def __init__(self) -> None:
        self._langfuse_client = langfuse_client
        self._client = AsyncOpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            max_retries=0,
        )

    async def generate(
        self,
        *,
        messages: list[OpenAIMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        request_timeout: float | None = None,
        **extra_params: Any,
    ) -> ChatCompletion:
        response = await self._client.chat.completions.create(
            model=_strip_model_prefix(model or settings.llm.model),
            messages=[self._serialize_message(message) for message in messages],
            temperature=temperature if temperature is not None else settings.llm.temperature,
            max_tokens=max_tokens if max_tokens is not None else settings.llm.max_tokens,
            timeout=request_timeout if request_timeout is not None else settings.llm.timeout,
            stream=False,
            **extra_params,
        )
        self._validate_response(response)
        logger.info("LLM non-stream request completed with model=%s", response.model)
        return response

    async def generate_text(
        self,
        *,
        messages: list[OpenAIMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        request_timeout: float | None = None,
        **extra_params: Any,
    ) -> str:
        response = await self.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            **extra_params,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self,
        *,
        messages: list[OpenAIMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        request_timeout: float | None = None,
        **extra_params: Any,
    ) -> AsyncGenerator[str, None]:
        stream = await self._client.chat.completions.create(
            model=_strip_model_prefix(model or settings.llm.model),
            messages=[self._serialize_message(message) for message in messages],
            temperature=temperature if temperature is not None else settings.llm.temperature,
            max_tokens=max_tokens if max_tokens is not None else settings.llm.max_tokens,
            timeout=request_timeout if request_timeout is not None else settings.llm.timeout,
            stream=True,
            **extra_params,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

        logger.info("LLM stream request completed with model=%s", model or settings.llm.model)

    @staticmethod
    def _serialize_message(message: OpenAIMessage) -> dict[str, str]:
        return {
            "role": message.role,
            "content": message.content,
        }

    @staticmethod
    def _validate_response(response: ChatCompletion) -> None:
        if not response.choices:
            raise ValueError("LLM returned no choices")

        message = response.choices[0].message.content
        if message is None or message == "":
            raise ValueError("LLM returned empty message content")


llm_client = LLMClient()
