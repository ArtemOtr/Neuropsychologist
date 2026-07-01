import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Protocol

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from src.schemas.openai_compat import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelListResponse,
    ModelObject,
    OpenAIMessage,
)


logger = logging.getLogger(__name__)


class ChatPipeline(Protocol):
    def process(self, user_query: str) -> AsyncGenerator[str, None]:
        pass


class OpenAICompatController:
    def __init__(self, pipeline: ChatPipeline) -> None:
        self._pipeline = pipeline

    async def chat_completions(
        self,
        body: ChatCompletionRequest,
    ) -> ChatCompletionResponse | StreamingResponse:
        user_query = self._extract_user_query(body)

        if body.stream:
            return StreamingResponse(
                self._stream_response(body=body, user_query=user_query),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return await self._generate_full_answer(body=body, user_query=user_query)

    def list_models(self) -> ModelListResponse:
        return ModelListResponse(
            data=[
                ModelObject(id="neuropsychologist-agent"),
            ]
        )

    async def _generate_full_answer(
        self,
        *,
        body: ChatCompletionRequest,
        user_query: str,
    ) -> ChatCompletionResponse:
        completion_id = self._completion_id()
        created = int(time.time())
        answer_parts: list[str] = []

        try:
            async for token in self._pipeline.process(user_query):
                answer_parts.append(token)
        except Exception as exc:
            logger.exception("OpenAI-compatible chat completion failed")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate chat completion",
            ) from exc

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=body.model,
            choices=[
                ChatCompletionChoice(
                    message=OpenAIMessage(
                        role="assistant",
                        content="".join(answer_parts),
                    )
                )
            ],
        )

    async def _stream_response(
        self,
        *,
        body: ChatCompletionRequest,
        user_query: str,
    ) -> AsyncGenerator[str, None]:
        completion_id = self._completion_id()
        created = int(time.time())

        yield self._make_chunk(
            completion_id=completion_id,
            created=created,
            model=body.model,
            role="assistant",
        )

        try:
            async for token in self._pipeline.process(user_query):
                if token:
                    yield self._make_chunk(
                        completion_id=completion_id,
                        created=created,
                        model=body.model,
                        content=token,
                    )
        except Exception:
            logger.exception("OpenAI-compatible streaming chat completion failed")
            yield self._make_chunk(
                completion_id=completion_id,
                created=created,
                model=body.model,
                content="Не удалось сформировать ответ. Попробуйте повторить запрос позже.",
            )

        yield self._make_chunk(
            completion_id=completion_id,
            created=created,
            model=body.model,
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"

    @staticmethod
    def _extract_user_query(body: ChatCompletionRequest) -> str:
        for message in reversed(body.messages):
            if message.role == "user" and message.content.strip():
                return message.content.strip()

        raise HTTPException(
            status_code=400,
            detail="messages must contain at least one non-empty user message",
        )

    @staticmethod
    def _make_chunk(
        *,
        completion_id: str,
        created: int,
        model: str,
        content: str | None = None,
        role: str | None = None,
        finish_reason: str | None = None,
    ) -> str:
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(
                        role=role,
                        content=content,
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    @staticmethod
    def _completion_id() -> str:
        return f"chatcmpl-{uuid.uuid4().hex[:29]}"
